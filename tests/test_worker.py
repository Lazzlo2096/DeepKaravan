from types import SimpleNamespace
from uuid import uuid4

import numpy as np
import pytest

from common.schemas import EmbeddingRequest
from common.tokenization import TextValidator
from gpu_worker.inference import InferenceEngine, resolve_device


@pytest.mark.parametrize(
    "selected",
    [
        ["dense"],
        ["sparse"],
        ["colbert"],
        ["dense", "sparse"],
        ["dense", "colbert"],
        ["sparse", "colbert"],
        ["dense", "sparse", "colbert"],
    ],
)
def test_model_flags_and_output_serialization(selected, settings, tokenizer, engine):
    inference = InferenceEngine.__new__(InferenceEngine)
    inference.settings = settings
    inference.validator = TextValidator(tokenizer, settings)
    inference.model = engine
    engine.encode.return_value = {
        "dense_vecs": np.array([[0.1, 0.2]]),
        "lexical_weights": [{"3": np.float32(0.5)}],
        "colbert_vecs": [np.array([[0.3, 0.4]])],
    }
    request = EmbeddingRequest(model="baai/bge-m3", input=["text"], embedding_types=selected)
    result = inference.encode(str(uuid4()), request)
    flags = engine.encode.call_args.kwargs
    assert flags["max_length"] == settings.max_input_tokens
    assert flags["return_dense"] == ("dense" in selected)
    assert flags["return_sparse"] == ("sparse" in selected)
    assert flags["return_colbert_vecs"] == ("colbert" in selected)
    data = result.model_dump(mode="json", exclude_none=True)["data"][0]
    expected_fields = {"embedding" if kind == "dense" else kind for kind in selected}
    assert set(data) == {"object", "index", *expected_fields}
    assert data["object"] == "embedding"
    if "dense" in selected:
        assert data["embedding"] == [0.1, 0.2]


def test_empty_batch_skips_model(settings, tokenizer, engine):
    inference = InferenceEngine.__new__(InferenceEngine)
    inference.settings = settings
    inference.validator = TextValidator(tokenizer, settings)
    inference.model = engine
    assert (
        inference.encode(str(uuid4()), EmbeddingRequest(model="baai/bge-m3", input=[])).data == []
    )
    engine.encode.assert_not_called()


def test_explicit_cpu_never_auto_selects_gpu(settings):
    assert resolve_device(settings, SimpleNamespace()) == "cpu"


@pytest.mark.parametrize("device", ["amd", "amd-wsl"])
def test_amd_uses_rocm_cuda_namespace(settings, device):
    settings.inference_device = device
    torch = SimpleNamespace(
        version=SimpleNamespace(hip="6.4"), cuda=SimpleNamespace(is_available=lambda: True)
    )
    assert resolve_device(settings, torch) == "cuda:0"


@pytest.mark.parametrize("device", ["amd", "amd-wsl"])
def test_amd_refuses_silent_cpu_or_nvidia_fallback(settings, device):
    settings.inference_device = device
    torch = SimpleNamespace(
        version=SimpleNamespace(hip=None), cuda=SimpleNamespace(is_available=lambda: True)
    )
    with pytest.raises(RuntimeError, match="ROCm"):
        resolve_device(settings, torch)
