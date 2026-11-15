"""Measure dense embeddings per second through the running REST API (stdlib only)."""

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID, uuid4

METRICS = ("embeddings_per_second", "responses_per_second", "submit_seconds", "seconds")


class BenchmarkError(Exception):
    def __init__(self, message, *, http_status=None, detail=None, stop=False):
        super().__init__(message)
        self.http_status = http_status
        self.detail = detail
        self.stop = stop


def positive_int(value):
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return result


def nonnegative_int(value):
    result = int(value)
    if result < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return result


def positive_float(value):
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return result


def describe(values):
    if not values:
        return dict.fromkeys(("mean", "median", "min", "max"))
    return {
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def format_stats(stats, digits=3):
    return [
        f"{stats[key]:.{digits}f}" if stats[key] is not None else "n/a"
        for key in ("mean", "median", "min", "max")
    ]


def tab_row(values, widths):
    cells = []
    for index, value in enumerate(values):
        cell = str(value)
        cells.append(cell)
        if index < len(widths):
            cells.append("\t" * max(1, widths[index] // 8 - len(cell) // 8))
    return "".join(cells)


def summarize(batch):
    successful = [run for run in batch["runs"] if run["status"] == "ok"]
    batch["successful_repeats"] = len(successful)
    batch["failed_repeats"] = len(batch["runs"]) - len(successful)
    for metric in METRICS:
        batch[metric] = describe([run[metric] for run in successful])
    batch["aggregate_embeddings_per_second"] = (
        len(successful) * batch["batch_size"] / sum(run["seconds"] for run in successful)
        if successful
        else None
    )


def save(report, output):
    for batch in report["batches"]:
        summarize(batch)
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(output.with_suffix(".json"))
    columns = [
        "batch_size",
        "status",
        "batch_total_seconds",
        "successful_repeats",
        "failed_repeats",
    ]
    columns += [
        f"{metric}_{stat}" for metric in METRICS for stat in ("mean", "median", "min", "max")
    ]
    columns += ["aggregate_embeddings_per_second"]
    with output.with_suffix(".csv.tmp").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for batch in report["batches"]:
            row = {
                key: batch[key]
                for key in (
                    "batch_size",
                    "status",
                    "batch_total_seconds",
                    "successful_repeats",
                    "failed_repeats",
                    "aggregate_embeddings_per_second",
                )
            }
            row.update(
                {
                    f"{metric}_{stat}": value
                    for metric in METRICS
                    for stat, value in batch[metric].items()
                }
            )
            writer.writerow(row)
    output.with_suffix(".csv.tmp").replace(output.with_suffix(".csv"))


def request(base, path, deadline, request_timeout, payload=None):
    remaining = deadline - time.perf_counter()
    if remaining <= 0:
        raise BenchmarkError("POST deadline exceeded; task state uncertain", stop=True)
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(
        base + path,
        data=body,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
    )
    try:
        response = urllib.request.urlopen(req, timeout=min(remaining, request_timeout))
    except urllib.error.HTTPError as exc:
        response = exc
    except (OSError, urllib.error.URLError) as exc:
        raise BenchmarkError(
            f"HTTP transport failed; delivery/completion uncertain: {exc}", stop=True
        ) from exc
    try:
        with response:
            raw = response.read()
            code = response.status
        data = json.loads(raw)
    except (ValueError, OSError) as exc:
        raise BenchmarkError(
            "Incomplete or invalid JSON response; task state uncertain", stop=True
        ) from exc
    if not isinstance(data, dict):
        raise BenchmarkError("Expected JSON object; task state uncertain", stop=True)
    return code, data, len(raw)


def check_result(result, size):
    result_id = str(UUID(result["task_id"]))
    if result.get("model") != "BAAI/bge-m3" or result.get("embedding_types") != ["dense"]:
        raise ValueError("Unexpected model or embedding types")
    data = result.get("data")
    if not isinstance(data, list) or len(data) != size:
        raise ValueError("Result text count mismatch")
    for index, item in enumerate(data):
        dense = item.get("dense")
        if item.get("index") != index or not isinstance(dense, list) or len(dense) != 1024:
            raise ValueError("Invalid index or dense vector dimension")
        if not all(isinstance(x, (int, float)) and math.isfinite(x) for x in dense):
            raise ValueError("Nonfinite or nonnumeric dense vector")
        if abs(sum(x * x for x in dense) - 1) > 0.03:
            raise ValueError("Dense vector is not normalized")
    return result_id


def error_detail(data):
    # Pydantic's 422 response can echo all 2048 texts in `input`.
    def compact(value):
        if isinstance(value, dict):
            return {key: compact(item) for key, item in value.items() if key != "input"}
        if isinstance(value, list):
            return [compact(item) for item in value]
        return value

    detail = compact(data)
    serialized = json.dumps(detail, ensure_ascii=False)
    return detail if len(serialized) <= 4000 else {"truncated_response": serialized[:4000]}


def attempt(args, size, text, record):
    started = time.perf_counter()
    deadline = started + args.timeout
    record.update(status="in_progress", started_at=datetime.now(UTC).isoformat())
    try:
        code, data, byte_count = request(
            args.base_url,
            "/api/v1/embeddings",
            deadline,
            args.request_timeout,
            {"model": "baai/bge-m3", "input": [text] * size, "embedding_types": ["dense"]},
        )
        record["submit_http_status"] = code
        record["submit_seconds"] = time.perf_counter() - started
        record["final_http_status"] = code
        if code == 202:
            record["task_id"] = data.get("task_id")
            raise BenchmarkError(
                "Expected HTTP 200 with embeddings; received 202. No polling; task may still run",
                http_status=code,
                detail=error_detail(data),
                stop=True,
            )
        if code != 200:
            # 422 is a terminal validation/inference error. Other responses may leave work alive.
            raise BenchmarkError(
                f"REST API returned HTTP {code}",
                http_status=code,
                detail=error_detail(data),
                stop=code != 422,
            )
        elapsed = record["submit_seconds"]
        record["task_id"] = check_result(data, size)
        record.update(
            status="ok",
            seconds=elapsed,
            embeddings_per_second=size / elapsed,
            responses_per_second=1 / elapsed,
            result_json_bytes=byte_count,
        )
    except Exception as exc:
        uncertain = record.get("final_http_status") != 200
        record.update(
            status="error",
            error_type=type(exc).__name__,
            error=str(exc)[:4000],
            failed_attempt_wall_seconds=time.perf_counter() - started,
            stop_benchmark=exc.stop if isinstance(exc, BenchmarkError) else uncertain,
        )
        if isinstance(exc, BenchmarkError):
            record["http_status"] = exc.http_status
            record["error_response"] = exc.detail


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-url",
        "--base_url",
        default=os.environ.get("DUPER_TEST_API_URL", "http://localhost:8000"),
    )
    parser.add_argument(
        "--batch-sizes",
        nargs="+",
        default=["1", "128", "512", "1024", "2048"],
        help='Batch sizes, e.g. 1 2 or "1 2"',
    )
    parser.add_argument("--repeats", type=positive_int, default=5)
    parser.add_argument("--warmups", type=nonnegative_int, default=2)
    parser.add_argument(
        "--timeout", type=positive_float, default=600, help="Total seconds per attempt"
    )
    parser.add_argument("--request-timeout", type=positive_float, default=90)
    parser.add_argument("--text-file", type=Path, default=Path(__file__).with_name("text.txt"))
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).with_name("results"))
    args = parser.parse_args()
    try:
        args.batch_sizes = [
            positive_int(item) for group in args.batch_sizes for item in group.split()
        ]
    except (ValueError, argparse.ArgumentTypeError):
        parser.error("--batch-sizes expects positive integers separated by spaces")
    if not args.batch_sizes:
        parser.error("--batch-sizes must contain at least one size")
    args.base_url = args.base_url.rstrip("/")
    url = urllib.parse.urlsplit(args.base_url)
    if url.scheme not in {"http", "https"} or not url.netloc or url.query or url.fragment:
        parser.error("--base-url must be an HTTP(S) base URL without query/fragment")
    text = args.text_file.read_text(encoding="utf-8-sig").rstrip("\r\n")
    if not text.strip() or len(set(args.batch_sizes)) != len(args.batch_sizes):
        parser.error("Text must be nonempty; batch sizes must not contain duplicates")
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / f"rest-batch-embeddings-{run_id}"
    report = {
        "schema_version": 3,
        "transport": "REST",
        "run_id": run_id,
        "started_at": datetime.now(UTC).isoformat(),
        "status": "running",
        "base_url": args.base_url,
        "endpoint": "/api/v1/embeddings",
        "embedding_type": "dense",
        "embedding_dimension": 1024,
        "batch_sizes": args.batch_sizes,
        "repeats": args.repeats,
        "warmups": args.warmups,
        "timeout_seconds": args.timeout,
        "request_timeout_seconds": args.request_timeout,
        "text": text,
        "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "scope": "One synchronous REST POST through HTTP 200 with embeddings; no polling",
        "batches": [],
    }
    save(report, output)
    batch = None
    batch_started = None
    try:
        print(f"REST API: {args.base_url}", flush=True)
        print(
            tab_row(["texts/request", "ok/errors", "batch time, s", "status"], [16, 16, 16]),
            flush=True,
        )
        for size in args.batch_sizes:
            batch_started = None
            batch = {
                "batch_size": size,
                "status": "warming_up",
                "warmups": [],
                "runs": [],
                "batch_total_seconds": 0.0,
                "started_at": None,
            }
            report["batches"].append(batch)
            for phase, count in (("warmups", args.warmups), ("runs", args.repeats)):
                batch["status"] = "warming_up" if phase == "warmups" else "measuring"
                if phase == "runs":
                    batch_started = time.perf_counter()
                    batch["started_at"] = datetime.now(UTC).isoformat()
                for iteration in range(count):
                    record = {"iteration": iteration + 1, "status": "in_progress"}
                    batch[phase].append(record)
                    attempt(args, size, text, record)
                    if batch_started is not None:
                        batch["batch_total_seconds"] = time.perf_counter() - batch_started
                    save(report, output)
                    if record.get("stop_benchmark"):
                        batch["status"] = "stopped_task_state_uncertain"
                        raise RuntimeError(
                            "Stopped to avoid overlapping unfinished tasks; see last attempt"
                        )
            all_attempts = batch["warmups"] + batch["runs"]
            batch["status"] = "ok" if all(r["status"] == "ok" for r in all_attempts) else "errors"
            batch["batch_total_seconds"] = time.perf_counter() - batch_started
            batch["finished_at"] = datetime.now(UTC).isoformat()
            save(report, output)
            print(
                tab_row(
                    [
                        size,
                        f"{batch['successful_repeats']}/{batch['failed_repeats']}",
                        f"{batch['batch_total_seconds']:.2f}",
                        batch["status"],
                    ],
                    [16, 16, 16],
                ),
                flush=True,
            )
            print(tab_row(["metric", "mean", "median", "min", "max"], [24, 8, 8, 8]), flush=True)
            for label, metric in (
                ("embeddings/s", "embeddings_per_second"),
                ("POST responses/s", "responses_per_second"),
            ):
                print(tab_row([label, *format_stats(batch[metric])], [24, 8, 8, 8]), flush=True)
            print(flush=True)
            failures = [r for r in all_attempts if r["status"] == "error"]
            if failures:
                print(f"  {failures[0]['error']}: {failures[0].get('error_response')}", flush=True)
        report["status"] = (
            "ok" if all(b["status"] == "ok" for b in report["batches"]) else "completed_with_errors"
        )
    except (Exception, KeyboardInterrupt) as exc:
        report["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "stopped"
        report["error_type"] = type(exc).__name__
        report["error"] = str(exc)[:4000]
    finally:
        if batch is not None and "finished_at" not in batch:
            if batch_started is not None:
                batch["batch_total_seconds"] = time.perf_counter() - batch_started
            batch["finished_at"] = datetime.now(UTC).isoformat()
        report["finished_at"] = datetime.now(UTC).isoformat()
        save(report, output)
    print(f"Status: {report['status']}; reports: {output}.json / .csv", flush=True)
    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
