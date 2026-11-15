"""Explicitly invoked benchmark of the real Celery broker and AMD worker (not pytest)."""

import argparse
import itertools
import json
import math
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from threading import local
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.celery_app import TASK_NAME, create_celery_app
from common.config import Settings
from common.jobs import JobStore, create_redis
from common.schemas import EmbeddingResult, TaskStatus
from common.tokenization import load_tokenizer

TYPES = ("dense", "sparse", "colbert")
TERMINAL = {"succeeded", "failed"}


def positive(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be finite and greater than zero")
    return number


def integers(value):
    numbers = [int(item) for item in value.split(",")]
    if not numbers or any(item < 1 for item in numbers):
        raise argparse.ArgumentTypeError("expected positive comma-separated integers")
    return sorted(set(numbers))


def quantiles(values):
    values = sorted(values)
    if not values:
        return None
    return {
        name: values[min(len(values) - 1, math.ceil(len(values) * fraction) - 1)]
        for name, fraction in (("p50", 0.5), ("p95", 0.95), ("p99", 0.99), ("max", 1))
    }


def bounded_text(tokenizer, budget, seed="hello "):
    """Find the longest repeated seed fitting the token budget; report actual tokens."""
    low, high = 0, budget * 2 + 1
    while low + 1 < high:
        middle = (low + high) // 2
        if len(tokenizer.encode(seed * middle).ids) <= budget:
            low = middle
        else:
            high = middle
    text = seed * low
    count = len(tokenizer.encode(text).ids)
    if not text or count > budget:
        raise ValueError(f"Cannot construct nonempty text within {budget} tokens")
    return text, count


def cases(settings, tokenizer, args):
    lengths = args.tokens or sorted(
        {
            min(32, settings.max_input_tokens),
            min(512, settings.max_input_tokens),
            settings.max_input_tokens,
        }
    )
    batches = args.batches or sorted({1, min(8, settings.max_texts), settings.max_texts})
    if max(lengths) > settings.max_input_tokens or max(batches) > settings.max_texts:
        raise ValueError("--tokens/--batches exceed configured worker limits")
    combinations = [
        list(combo) for size in range(1, 4) for combo in itertools.combinations(TYPES, size)
    ]
    # A focused default suite; --matrix opts into the expensive Cartesian product.
    specs = (
        list(itertools.product(lengths, batches, combinations))
        if args.matrix
        else (
            [(length, 1, ["dense"]) for length in lengths]
            + [(lengths[0], batch, ["dense"]) for batch in batches if batch != 1]
            + [(lengths[0], 1, types) for types in combinations if types != ["dense"]]
        )
    )
    for budget, batch, types in specs:
        text, actual = bounded_text(tokenizer, budget)
        yield {
            "name": f"tokens-{budget}_batch-{batch}_{'+'.join(types)}",
            "payload": {"model": "baai/bge-m3", "input": [text] * batch, "embedding_types": types},
            "text_count": batch,
            "input_tokens": actual * batch,
            "tokens_per_text": actual,
            "requested_token_budget": budget,
        }
    texts = ["Привет, мир!", "Search across documents", "你好，世界", "مرحبا بالعالم", "🙂"]
    texts = [
        text for text in texts if len(tokenizer.encode(text).ids) <= settings.max_input_tokens
    ][: settings.max_texts]
    if texts:
        yield {
            "name": "multilingual",
            "payload": {"model": "baai/bge-m3", "input": texts, "embedding_types": list(TYPES)},
            "text_count": len(texts),
            "input_tokens": sum(len(tokenizer.encode(text).ids) for text in texts),
        }


def boundary_cases(settings, tokenizer):
    yield (
        "empty_batch",
        {"model": "baai/bge-m3", "input": [], "embedding_types": ["dense"]},
        "succeeded",
    )
    yield (
        "empty_string",
        {"model": "baai/bge-m3", "input": [""], "embedding_types": list(TYPES)},
        "succeeded",
    )
    text, actual = bounded_text(tokenizer, settings.max_input_tokens)
    # Do not label a shorter tokenizer-dependent fixture an exact boundary.
    if actual != settings.max_input_tokens:
        raise ValueError(f"Exact token-limit fixture unavailable: got {actual} tokens")
    yield (
        "exact_token_limit",
        {"model": "baai/bge-m3", "input": [text], "embedding_types": ["dense"]},
        "succeeded",
    )
    over = text + " hello"
    while len(tokenizer.encode(over).ids) <= settings.max_input_tokens:
        over += " hello"
    yield (
        "over_token_limit",
        {"model": "baai/bge-m3", "input": [over], "embedding_types": ["dense"]},
        "failed",
    )
    yield (
        "exact_text_limit",
        {
            "model": "baai/bge-m3",
            "input": ["hi"] * settings.max_texts,
            "embedding_types": ["dense"],
        },
        "succeeded",
    )
    yield (
        "over_text_limit",
        {"model": "baai/bge-m3", "input": ["hi"] * (settings.max_texts + 1)},
        "failed",
    )
    for name, payload in (
        ("missing_input", {"model": "baai/bge-m3"}),
        ("wrong_text_type", {"model": "baai/bge-m3", "input": [123]}),
        (
            "unknown_embedding",
            {"model": "baai/bge-m3", "input": ["hi"], "embedding_types": ["unknown"]},
        ),
        (
            "duplicate_embedding",
            {"model": "baai/bge-m3", "input": ["hi"], "embedding_types": ["dense", "dense"]},
        ),
        ("empty_embedding_types", {"model": "baai/bge-m3", "input": ["hi"], "embedding_types": []}),
        ("extra_field", {"model": "baai/bge-m3", "input": ["hi"], "unexpected": True}),
    ):
        yield name, payload, "failed"


class Publisher:
    def __init__(self, settings):
        self.settings = settings
        self.context = local()

    def send(self, task_id, payload):
        if not hasattr(self.context, "app"):
            self.context.app = create_celery_app(self.settings)
            self.context.store = JobStore(
                create_redis(self.settings), self.settings.result_ttl_seconds
            )
        started = time.perf_counter()
        try:
            self.context.store.create(task_id)
            self.context.app.send_task(
                TASK_NAME,
                args=[payload],
                task_id=task_id,
                queue=self.settings.task_queue,
                retry=False,
            )
            return {
                "publish_seconds": time.perf_counter() - started,
                "publish_finished_monotonic": time.perf_counter(),
                "publish_error": None,
            }
        except Exception as exc:
            # A transport error can occur AFTER delivery. Keep this ID under observation.
            return {
                "publish_seconds": time.perf_counter() - started,
                "publish_finished_monotonic": time.perf_counter(),
                "publish_error": f"{type(exc).__name__}: {exc}",
            }


def validate_result(raw, task_id, payload):
    result = EmbeddingResult.model_validate_json(raw)
    if str(result.task_id) != task_id or result.model != "BAAI/bge-m3":
        raise ValueError("Result identity mismatch")
    if result.embedding_types != payload.get("embedding_types", ["dense"]):
        raise ValueError("Result embedding types mismatch")
    if len(result.data) != len(payload["input"]):
        raise ValueError("Result text count mismatch")
    selected = {str(item) for item in result.embedding_types}
    for index, item in enumerate(result.data):
        if item.index != index:
            raise ValueError("Result order mismatch")
        for kind in TYPES:
            if (getattr(item, "embedding" if kind == "dense" else kind) is not None) != (
                kind in selected
            ):
                raise ValueError(f"Unexpected/missing {kind}")
        if item.embedding is not None:
            if len(item.embedding) != 1024 or not all(math.isfinite(x) for x in item.embedding):
                raise ValueError("Invalid dense vector")
            if abs(sum(x * x for x in item.embedding) - 1) > 0.03:
                raise ValueError("Dense vector is not normalized")
        if item.sparse is not None:
            if any(not k.isdigit() or not math.isfinite(v) for k, v in item.sparse.items()):
                raise ValueError("Invalid sparse vector")
        if item.colbert is not None:
            if any(
                len(row) != 1024 or not all(math.isfinite(x) for x in row) for row in item.colbert
            ):
                raise ValueError("Invalid ColBERT vector")


def stage(
    settings,
    client,
    broker,
    publisher,
    pool,
    case,
    window,
    args,
    report,
    expected="succeeded",
    single=False,
):
    started = time.perf_counter()
    records, futures, pending = {}, {}, set()
    errors = Counter()
    peak_queue = 0
    peak_memory = 0
    publish_open = True
    drain_deadline = None
    sampled = False
    completed_samples = []
    payload = case["payload"]
    total = 1 if single else args.messages
    stage_report = {
        "case": {key: value for key, value in case.items() if key != "payload"},
        "window": window,
        "expected_state": expected,
        "payload_json_bytes": len(json.dumps(payload, ensure_ascii=False).encode()),
        "tasks": records,
        "errors": {},
        "complete": False,
    }
    report["stages"].append(stage_report)
    while True:
        now = time.perf_counter()
        if now - started >= args.stage_seconds or len(records) >= total or errors:
            publish_open = False
        if not publish_open and drain_deadline is None:
            drain_deadline = now + args.drain_timeout
        while publish_open and len(pending) + len(futures) < window and len(records) < total:
            task_id = str(uuid4())
            records[task_id] = {"state": "publishing", "submitted_offset_seconds": now - started}
            futures[pool.submit(publisher.send, task_id, payload)] = task_id
        for future in list(futures):
            if not future.done():
                continue
            task_id = futures.pop(future)
            records[task_id].update(future.result())
            records[task_id]["publish_acknowledged_offset_seconds"] = (
                records[task_id].pop("publish_finished_monotonic") - started
            )
            records[task_id]["state"] = "awaiting_status"
            pending.add(task_id)
            if records[task_id]["publish_error"]:
                errors["publish_error_delivery_uncertain"] += 1
        ids = list(pending)
        for offset in range(0, len(ids), 128):
            chunk = ids[offset : offset + 128]
            statuses = client.mget([JobStore.status_key(task_id) for task_id in chunk])
            for task_id, raw in zip(chunk, statuses, strict=True):
                if raw is None:
                    continue
                status = TaskStatus.model_validate_json(raw)
                records[task_id]["state"] = str(status.status)
                if status.status not in TERMINAL:
                    continue
                pending.remove(task_id)
                record = records[task_id]
                record["status"] = status.model_dump(mode="json")
                record["observed_total_seconds"] = (
                    time.perf_counter() - started - record["submitted_offset_seconds"]
                )
                if status.started_at and status.finished_at:
                    record["queue_seconds"] = (
                        status.started_at - status.submitted_at
                    ).total_seconds()
                    record["worker_seconds"] = (
                        status.finished_at - status.started_at
                    ).total_seconds()
                if status.status != expected:
                    errors[f"unexpected_{status.status}"] += 1
                if status.status == "failed":
                    record["worker_error"] = status.error.model_dump() if status.error else None
                    if expected == "failed" and (
                        status.error is None or status.error.code != "inference_failed"
                    ):
                        errors["unexpected_failure_code"] += 1
                else:
                    size = client.strlen(JobStore.result_key(task_id))
                    record["result_bytes"] = size
                    if not size:
                        errors["missing_result"] += 1
                    if not sampled and size:
                        if size <= args.validate_max_mib * 1024 * 1024:
                            try:
                                validate_result(
                                    client.get(JobStore.result_key(task_id)), task_id, payload
                                )
                                stage_report["sample_result_validation"] = "passed"
                            except Exception as exc:
                                errors["invalid_result"] += 1
                                record["result_error"] = f"{type(exc).__name__}: {exc}"
                        else:
                            stage_report["sample_result_validation"] = "skipped_size_limit"
                        sampled = True
                    if status.finished_at:
                        completed_samples.append(status.finished_at.timestamp())
        peak_queue = max(peak_queue, broker.llen(settings.task_queue))
        peak_memory = max(peak_memory, client.info("memory")["used_memory"])
        stage_report["errors"] = dict(errors)
        if not futures and not pending and (not publish_open or len(records) >= total):
            break
        if drain_deadline is not None and time.perf_counter() >= drain_deadline:
            errors["unresolved_at_deadline"] += len(pending) + len(futures)
            break
        time.sleep(args.poll_interval)
    elapsed = time.perf_counter() - started
    successes = [record for record in records.values() if record["state"] == "succeeded"]
    terminal = sum(record["state"] in TERMINAL for record in records.values())
    publish_ok = sum(
        "publish_seconds" in record and not record["publish_error"] for record in records.values()
    )
    stage_report.update(
        elapsed_seconds=elapsed,
        scheduled=len(records),
        published=publish_ok,
        succeeded=len(successes),
        failed=sum(r["state"] == "failed" for r in records.values()),
        unresolved=len(records) - terminal,
        errors=dict(errors),
        complete=terminal == len(records) and not errors,
        successful_tasks_per_second=len(successes) / elapsed,
        successful_texts_per_second=len(successes) * case.get("text_count", 0) / elapsed,
        successful_input_tokens_per_second=len(successes) * case.get("input_tokens", 0) / elapsed,
        peak_sampled_queue_depth=peak_queue,
        peak_sampled_redis_used_memory_bytes=peak_memory,
        total_result_bytes=sum(record.get("result_bytes", 0) for record in records.values()),
        latency_seconds={
            key: quantiles([r[key] for r in records.values() if key in r])
            for key in (
                "publish_seconds",
                "queue_seconds",
                "worker_seconds",
                "observed_total_seconds",
            )
        },
    )
    first_burst = list(records.values())[: min(window, total)]
    acknowledged = [
        r
        for r in first_burst
        if "publish_acknowledged_offset_seconds" in r and not r["publish_error"]
    ]
    if len(acknowledged) == len(first_burst) and acknowledged:
        burst_seconds = max(r["publish_acknowledged_offset_seconds"] for r in acknowledged)
        stage_report["initial_burst"] = {
            "published": len(acknowledged),
            "elapsed_seconds": burst_seconds,
            "published_tasks_per_second": len(acknowledged) / burst_seconds,
        }
    # Worker timestamps avoid inflating the rate when many completions share one poll.
    completed_samples.sort()
    if len(completed_samples) > 1 and completed_samples[-1] > completed_samples[0]:
        stage_report["completion_interval_tasks_per_second"] = (len(completed_samples) - 1) / (
            completed_samples[-1] - completed_samples[0]
        )
    print(
        json.dumps({k: v for k, v in stage_report.items() if k != "tasks"}, ensure_ascii=False),
        flush=True,
    )
    return stage_report


def save(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def summarize(report):
    summary = {}
    for item in report["stages"]:
        if item["case"]["name"].startswith("boundary_") or item["case"]["name"] in {
            "warmup",
            "post_error_recovery",
        }:
            continue
        entry = summary.setdefault(
            item["case"]["name"],
            {
                "largest_passed_window": None,
                "first_failed_window": None,
                "best_successful_tasks_per_second": None,
                "tested_windows": [],
            },
        )
        entry["tested_windows"].append(item["window"])
        if item["complete"]:
            entry["largest_passed_window"] = item["window"]
            entry["best_successful_tasks_per_second"] = max(
                entry["best_successful_tasks_per_second"] or 0,
                item["successful_tasks_per_second"],
            )
        elif entry["first_failed_window"] is None:
            entry["first_failed_window"] = item["window"]
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--windows", type=integers, default=[1, 8, 32, 128])
    parser.add_argument("--messages", type=int, default=256, help="Maximum tasks per stage")
    parser.add_argument("--publishers", type=int, default=4)
    parser.add_argument("--stage-seconds", type=positive, default=60)
    parser.add_argument("--drain-timeout", type=positive, default=600)
    parser.add_argument("--poll-interval", type=positive, default=0.05)
    parser.add_argument("--validate-max-mib", type=positive, default=32)
    parser.add_argument("--tokens", type=integers)
    parser.add_argument("--batches", type=integers)
    parser.add_argument("--matrix", action="store_true")
    parser.add_argument("--case", action="append", help="Exact case name; repeat to select several")
    parser.add_argument("--skip-boundaries", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("/app/super_test/results"))
    args = parser.parse_args()
    if args.messages < max(args.windows) or args.publishers < 1:
        parser.error("--messages must be >= max(--windows); --publishers must be >= 1")
    settings = Settings()
    if settings.inference_device != "amd":
        parser.error("Requires INFERENCE_DEVICE=amd and the real celery-amd worker")
    client = create_redis(settings)
    broker = client.__class__.from_url(
        settings.redis_broker_url,
        socket_timeout=settings.redis_socket_timeout_seconds,
        socket_connect_timeout=settings.redis_socket_timeout_seconds,
    )
    client.ping()
    broker.ping()
    if broker.llen(settings.task_queue) or broker.hlen("unacked"):
        parser.error("Queue/unacked is nonempty; use an idle dedicated benchmark stack")
    tokenizer = load_tokenizer(settings)
    selected = list(cases(settings, tokenizer, args))
    if args.case:
        names = {case["name"] for case in selected}
        unknown = set(args.case) - names
        if unknown:
            parser.error(f"Unknown case(s): {sorted(unknown)}. Available: {sorted(names)}")
        selected = [case for case in selected if case["name"] in args.case]
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
    output = args.output / f"celery-amd-{run_id}.json"
    report = {
        "run_id": run_id,
        "started_at": datetime.now(UTC).isoformat(),
        "configuration": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "worker_configuration": settings.model_dump(
            exclude={"redis_broker_url", "redis_results_url"}
        ),
        "scope": "Direct Celery JSON tasks; HTTP validation/latency excluded",
        "redis_memory_at_start_bytes": client.info("memory")["used_memory"],
        "stages": [],
        "outcome": "running",
    }
    save(output, report)
    publisher = Publisher(settings)
    # Shutdown does not cancel already published work. Finite socket timeouts bound producers.
    pool = ThreadPoolExecutor(max_workers=args.publishers)
    try:
        warmup = {
            "name": "warmup",
            "payload": {"model": "baai/bge-m3", "input": ["hi"], "embedding_types": ["dense"]},
        }
        result = stage(
            settings, client, broker, publisher, pool, warmup, 1, args, report, single=True
        )
        save(output, report)
        if not result["complete"]:
            raise RuntimeError("Warmup failed: worker may be unavailable or not ready")
        for case in selected:
            for window in args.windows:
                result = stage(
                    settings, client, broker, publisher, pool, case, window, args, report
                )
                save(output, report)
                if not result["complete"]:
                    raise RuntimeError(
                        f"Capacity boundary/error in {case['name']} at window={window}"
                    )
        if not args.skip_boundaries:
            for name, payload, expected in boundary_cases(settings, tokenizer):
                result = stage(
                    settings,
                    client,
                    broker,
                    publisher,
                    pool,
                    {"name": f"boundary_{name}", "payload": payload},
                    1,
                    args,
                    report,
                    expected=expected,
                    single=True,
                )
                save(output, report)
                if not result["complete"]:
                    raise RuntimeError(f"Unexpected boundary behavior: {name}")
            # Verify valid work still succeeds after malformed messages.
            stage_result = stage(
                settings,
                client,
                broker,
                publisher,
                pool,
                {**warmup, "name": "post_error_recovery"},
                1,
                args,
                report,
                single=True,
            )
            if not stage_result["complete"]:
                raise RuntimeError("Worker did not recover after invalid messages")
        report["outcome"] = "configured_range_passed_maximum_not_proven"
    except (Exception, KeyboardInterrupt) as exc:
        report["outcome"] = "stopped_on_error_or_interrupt"
        report["stop_reason"] = f"{type(exc).__name__}: {exc}"
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
        report["summary_by_case"] = summarize(report)
        report["finished_at"] = datetime.now(UTC).isoformat()
        save(output, report)
        client.close()
        broker.close()
    print(f"Report: {output}; outcome={report['outcome']}", flush=True)
    return 0 if report["outcome"] == "configured_range_passed_maximum_not_proven" else 1


if __name__ == "__main__":
    raise SystemExit(main())
