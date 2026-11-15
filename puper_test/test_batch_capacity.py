"""Run explicitly: python -m pytest puper_test/test_batch_capacity.py."""

from types import SimpleNamespace

import numpy as np
import pytest

from puper_test.batch_capacity import ForwardFailure, measure, summarize


class FakeModel:
    def __init__(self, fail=False):
        self.model = self
        self.fail = fail
        self.calls = 0

    def forward(self, features):
        self.calls += 1
        if self.fail:
            raise RuntimeError("out of memory")

    def encode(self, texts, **kwargs):
        # Reproduce the library's catch of RuntimeError. Our wrapper must escape it.
        try:
            self.forward({"input_ids": np.zeros((len(texts), 32))})
        except RuntimeError:
            pytest.fail("Library swallowed the failure and would reduce the batch")
        return {"dense_vecs": np.zeros((len(texts), 1024))}


def run_measure(model):
    engine = SimpleNamespace(model=model, settings=SimpleNamespace(max_input_tokens=8192))
    torch = SimpleNamespace(cuda=SimpleNamespace(OutOfMemoryError=type("OOM", (RuntimeError,), {})))
    args = SimpleNamespace(types=["dense"], warmups=1, repeats=3)
    return measure(engine, torch, "text", 32, 32, 4, args, gpu=False)


def test_real_batch_sizes_and_forward_restored():
    model = FakeModel()
    original = model.forward
    row = run_measure(model)
    assert row["status"] == "ok"
    assert row["forward_shapes"] == [[[4, 32]]] * 3
    assert len(row["seconds"]) == 3
    assert row["texts_per_second"] == pytest.approx(12 / sum(row["seconds"]))
    assert model.calls == 4
    assert model.forward == original


def test_oom_cannot_be_swallowed_by_library():
    model = FakeModel(fail=True)
    original = model.forward
    row = run_measure(model)
    assert row["status"] == "oom"
    assert model.calls == 1
    assert model.forward == original
    assert not issubclass(ForwardFailure, RuntimeError)


def test_summary_uses_throughput_and_excludes_failed_batches():
    rows = [
        {
            "token_budget": 32,
            "tokens_per_text": 30,
            "batch_size": batch,
            "status": "ok",
            "texts_per_second": rate,
        }
        for batch, rate in [(1, 10), (2, 96), (4, 100)]
    ]
    rows.append({"token_budget": 32, "batch_size": 8, "status": "oom"})
    summary = summarize(rows)[0]
    assert summary["fastest_tested_batch"] == 4
    assert summary["smallest_batch_within_5_percent_of_peak"] == 2
    assert summary["largest_successful_tested_batch"] == 4
    assert summary["first_failed_batch"] == 8
