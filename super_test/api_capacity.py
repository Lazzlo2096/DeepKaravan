"""Opt-in concurrent HTTP benchmark of the single-container IPC inference service."""

import argparse
import itertools
import json
import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from smoke import request, wait_result  # noqa: E402

TYPES = ("dense", "sparse", "colbert")


def run_case(base, timeout, kinds, batch):
    started = time.monotonic()
    code, task = request(
        base,
        "/api/v1/embeddings/async",
        {
            "model": "baai/bge-m3",
            "input": ["Concurrent inference test"] * batch,
            "embedding_types": kinds,
        },
        timeout=timeout,
    )
    if code != 202:
        return {
            "types": kinds,
            "batch": batch,
            "status": code,
            "seconds": time.monotonic() - started,
        }
    result = wait_result(base, task["result_url"], timeout)
    assert result["embedding_types"] == kinds
    assert len(result["data"]) == batch
    expected = {"object", "index", *("embedding" if k == "dense" else k for k in kinds)}
    for index, item in enumerate(result["data"]):
        assert item["index"] == index and item["object"] == "embedding"
        assert set(item) == expected
        if "dense" in kinds:
            vector = item["embedding"]
            assert len(vector) == 1024 and all(math.isfinite(x) for x in vector)
            assert abs(sum(x * x for x in vector) - 1) < 0.03
        if "sparse" in kinds:
            assert all(k.isdigit() and math.isfinite(v) for k, v in item["sparse"].items())
        if "colbert" in kinds:
            assert all(
                len(row) == 1024 and all(math.isfinite(v) for v in row) for row in item["colbert"]
            )
    return {
        "types": kinds,
        "batch": batch,
        "status": 200,
        "task_id": task["task_id"],
        "seconds": time.monotonic() - started,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--output", type=Path, default=ROOT / "test-results" / "api-capacity.json")
    args = parser.parse_args()
    if min(args.concurrency, args.repeats, args.batch_size, args.timeout) <= 0:
        parser.error("concurrency, repeats, batch-size and timeout must be positive")
    kinds = [list(c) for count in range(1, 4) for c in itertools.combinations(TYPES, count)]
    jobs = kinds * args.repeats
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [
            pool.submit(run_case, args.base_url.rstrip("/"), args.timeout, k, args.batch_size)
            for k in jobs
        ]
        results = [future.result() for future in futures]
    report = {
        "at": datetime.now(UTC).isoformat(),
        "seconds": time.monotonic() - started,
        "concurrency": args.concurrency,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return int(any(row["status"] != 200 for row in results))


if __name__ == "__main__":
    raise SystemExit(main())
