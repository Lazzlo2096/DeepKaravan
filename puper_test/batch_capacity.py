"""Opt-in benchmark of real BGE-M3 internal batches; no API or IPC."""

import argparse
import gc
import hashlib
import json
import logging
import math
import platform
import statistics
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

# Also support direct execution from the project root without PYTHONPATH setup.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class ForwardFailure(Exception):
    """Escape FlagEmbedding's catch(RuntimeError) automatic batch shrinking."""


def strict_forward(model, observed):
    original = model.forward

    def forward(*args, **kwargs):
        features = args[0] if args else kwargs["features"]
        observed.append(list(features["input_ids"].shape))
        try:
            return original(*args, **kwargs)
        except RuntimeError as exc:
            raise ForwardFailure(str(exc)) from exc

    return forward


def positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("Must be positive")
    return number


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--batch-sizes", nargs="+", type=positive, default=[1, 2, 4, 8, 16, 32])
    result.add_argument("--tokens", nargs="+", type=positive, default=[32, 128, 512, 2048, 8192])
    result.add_argument(
        "--types", nargs="+", choices=["dense", "sparse", "colbert"], default=["dense"]
    )
    result.add_argument("--warmups", type=positive, default=2)
    result.add_argument("--repeats", type=positive, default=5)
    result.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "results")
    return result


def make_text(tokenizer, budget):
    """Find a repeatable synthetic text fitting the budget, including special tokens."""
    seed = " Проверка скорости модели: поиск документов и обработка текста."
    low, high = 0, budget * 2
    while low < high:
        middle = (low + high + 1) // 2
        if len(tokenizer.encode(seed * middle).ids) <= budget:
            low = middle
        else:
            high = middle - 1
    if not low:
        raise ValueError(f"Token budget {budget} is too small for the synthetic seed")
    text = seed * low
    return text, len(tokenizer.encode(text).ids)


def summarize(rows):
    summaries = []
    for budget in sorted({row["token_budget"] for row in rows}):
        group = [row for row in rows if row["token_budget"] == budget]
        successful = [row for row in group if row["status"] == "ok"]
        if not successful:
            summaries.append({"token_budget": budget, "status": "no_successful_batch"})
            continue
        best = max(successful, key=lambda row: row["texts_per_second"])
        near_best = min(
            (
                row
                for row in successful
                if row["texts_per_second"] >= 0.95 * best["texts_per_second"]
            ),
            key=lambda row: row["batch_size"],
        )
        summaries.append(
            {
                "token_budget": budget,
                "tokens_per_text": best["tokens_per_text"],
                "fastest_tested_batch": best["batch_size"],
                "fastest_texts_per_second": best["texts_per_second"],
                "smallest_batch_within_5_percent_of_peak": near_best["batch_size"],
                "largest_successful_tested_batch": max(row["batch_size"] for row in successful),
                "first_failed_batch": next(
                    (row["batch_size"] for row in group if row["status"] == "oom"), None
                ),
                "absolute_maximum_proven": False,
            }
        )
    return summaries


def save_report(path, report):
    report["summary"] = summarize(report["cases"])
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def validate_output(output, types, size):
    import numpy as np

    for kind, key in [
        ("dense", "dense_vecs"),
        ("sparse", "lexical_weights"),
        ("colbert", "colbert_vecs"),
    ]:
        if kind not in types:
            continue
        values = output[key]
        if len(values) != size:
            raise ValueError(f"{key}: expected {size} results, got {len(values)}")
        if kind == "sparse":
            finite = all(math.isfinite(float(v)) for item in values for v in item.values())
        else:
            finite = all(np.isfinite(item).all() for item in values)
        if not finite:
            raise ValueError(f"{key}: non-finite result")


def measure(engine, torch, text, actual_tokens, budget, batch, args, gpu):
    texts = [text] * batch
    observed = []
    model = engine.model.model
    original_forward = model.forward
    model.forward = strict_forward(model, observed)
    row = {
        "token_budget": budget,
        "tokens_per_text": actual_tokens,
        "batch_size": batch,
        "embedding_types": args.types,
        "status": "running",
        "seconds": [],
        "forward_shapes": [],
        "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
    }

    def sync():
        if gpu:
            torch.cuda.synchronize()

    def encode():
        return engine.model.encode(
            texts,
            batch_size=batch,
            max_length=engine.settings.max_input_tokens,
            return_dense="dense" in args.types,
            return_sparse="sparse" in args.types,
            return_colbert_vecs="colbert" in args.types,
        )

    try:
        gc.collect()
        if gpu:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        for _ in range(args.warmups):
            output = encode()
            sync()
            validate_output(output, args.types, batch)
            del output
        if gpu:
            torch.cuda.reset_peak_memory_stats()
        for _ in range(args.repeats):
            observed.clear()
            sync()
            start = time.perf_counter()
            output = encode()
            sync()
            row["seconds"].append(time.perf_counter() - start)
            row["forward_shapes"].append(list(observed))
            # Checking actual forwards also catches unexpected library-side splitting.
            if not observed or any(shape[0] != batch for shape in observed):
                raise ValueError(f"Requested batch {batch}, actual forward shapes: {observed}")
            validate_output(output, args.types, batch)
            del output
        elapsed = sum(row["seconds"])
        row.update(
            status="ok",
            texts_per_second=batch * args.repeats / elapsed,
            input_tokens_per_second=actual_tokens * batch * args.repeats / elapsed,
            median_batch_seconds=statistics.median(row["seconds"]),
            min_batch_seconds=min(row["seconds"]),
            max_batch_seconds=max(row["seconds"]),
        )
    except (ForwardFailure, RuntimeError) as exc:
        cause = exc.__cause__ or exc
        is_oom = (
            isinstance(cause, torch.cuda.OutOfMemoryError) or "out of memory" in str(cause).lower()
        )
        if not is_oom:
            raise
        row.update(status="oom", error_type=type(cause).__name__, error=str(cause))
    finally:
        model.forward = original_forward
        if gpu:
            row["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
            row["peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
    return row


def main():
    cli = parser()
    args = cli.parse_args()
    from common.config import Settings

    settings = Settings()
    if max(args.tokens) > settings.max_input_tokens:
        cli.error("--tokens exceeds MAX_INPUT_TOKENS")
    if max(args.batch_sizes) > settings.max_texts:
        cli.error("--batch-sizes exceeds MAX_TEXTS")
    args.types = sorted(set(args.types))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = datetime.now(UTC)
    path = args.output_dir / f"batch-capacity-{started:%Y%m%dT%H%M%S%fZ}.json"
    report = {
        "status": "loading",
        "started_at": started.isoformat(),
        "scope": "model.encode including tokenization and CPU outputs; no queue or JSON",
        "arguments": {**vars(args), "output_dir": str(args.output_dir)},
        "device": settings.inference_device,
        "fp16_requested": settings.use_fp16,
        "model_id": settings.model_id,
        "model_revision": settings.model_revision,
        "max_input_tokens": settings.max_input_tokens,
        "platform": platform.platform(),
        "cases": [],
    }
    print(f"Report: {path}", flush=True)
    save_report(path, report)
    try:
        import torch

        from gpu_worker.inference import InferenceEngine

        report["torch_version"] = torch.__version__
        report["hip_version"] = torch.version.hip
        engine = InferenceEngine(settings)
        gpu = settings.inference_device != "cpu"
        report["model_dtype"] = str(next(engine.model.model.parameters()).dtype)
        if gpu:
            properties = torch.cuda.get_device_properties(0)
            report["gpu"] = {"name": properties.name, "total_memory_bytes": properties.total_memory}
        report["status"] = "running"
        save_report(path, report)
        for budget in sorted(set(args.tokens)):
            text, actual = make_text(engine.validator.tokenizer, budget)
            exhausted = False
            for batch in sorted(set(args.batch_sizes)):
                if exhausted:
                    row = {
                        "token_budget": budget,
                        "tokens_per_text": actual,
                        "batch_size": batch,
                        "status": "skipped_after_oom",
                    }
                else:
                    row = measure(engine, torch, text, actual, budget, batch, args, gpu)
                    exhausted = row["status"] == "oom"
                report["cases"].append(row)
                save_report(path, report)
                print(json.dumps(row, ensure_ascii=False), flush=True)
                gc.collect()
                if gpu:
                    torch.cuda.empty_cache()
        report["status"] = "completed_tested_range"
    except KeyboardInterrupt:
        report["status"] = "interrupted"
        return 130
    except Exception as exc:
        report.update(status="error", error_type=type(exc).__name__, error=str(exc))
        logging.exception("Benchmark stopped")
        return 1
    finally:
        report["finished_at"] = datetime.now(UTC).isoformat()
        save_report(path, report)
        print(json.dumps(report["summary"], ensure_ascii=False, indent=2), flush=True)
        print(f"Report saved: {path}", flush=True)
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    raise SystemExit(main())
