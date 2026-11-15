from types import SimpleNamespace
from uuid import uuid4

import numpy as np
import pytest
from celery.exceptions import Reject
from redis.exceptions import ConnectionError

from common.celery_app import create_celery_app
from common.schemas import EmbeddingRequest, EmbeddingResult
from common.tokenization import TextValidator
from worker import tasks
from worker.inference import InferenceEngine, resolve_device


def run_task(task_id, payload):
    tasks.embed.push_request(id=str(task_id))
    try:
        return tasks.embed.run(payload)
    finally:
        tasks.embed.pop_request()


@pytest.fixture
def configured_worker(monkeypatch, store, engine):
    monkeypatch.setattr(tasks, "get_store", lambda: store)
    monkeypatch.setattr(tasks, "get_engine", lambda: engine)
    return engine


def test_worker_persists_result_before_completion(store, configured_worker):
    task_id = uuid4()
    store.create(task_id)
    configured_worker.encode.return_value = EmbeddingResult(
        task_id=task_id, model="BAAI/bge-m3", embedding_types=["dense"], data=[]
    )
    run_task(task_id, {"input": [], "embedding_types": ["dense"]})
    assert store.get(task_id).status == "succeeded"
    assert store.get_result(task_id).task_id == task_id


def test_interrupted_running_task_is_executed_again(store, configured_worker):
    task_id = uuid4()
    store.create(task_id)
    store.start(task_id)
    configured_worker.encode.return_value = EmbeddingResult(
        task_id=task_id, model="BAAI/bge-m3", embedding_types=["dense"], data=[]
    )
    run_task(task_id, {"input": []})
    configured_worker.encode.assert_called_once()
    assert store.get(task_id).status == "succeeded"


def test_ack_redelivery_does_not_recompute_or_extend_ttl(store, configured_worker):
    task_id = uuid4()
    store.create(task_id)
    configured_worker.encode.return_value = EmbeddingResult(
        task_id=task_id, model="BAAI/bge-m3", embedding_types=["dense"], data=[]
    )
    run_task(task_id, {"input": []})
    expiry = store.get(task_id).expires_at
    run_task(task_id, {"input": []})
    configured_worker.encode.assert_called_once()
    assert store.get(task_id).expires_at == expiry


def test_inference_exception_is_terminal_failure(store, configured_worker):
    task_id = uuid4()
    store.create(task_id)
    configured_worker.encode.side_effect = RuntimeError("sensitive internal details")
    run_task(task_id, {"input": ["text"]})
    task = store.get(task_id)
    assert task.status == "failed"
    assert "RuntimeError" in task.error.message
    assert "sensitive" not in task.error.message


def test_storage_failure_requeues_without_ack(store, configured_worker, monkeypatch):
    task_id = uuid4()
    store.create(task_id)
    configured_worker.encode.return_value = EmbeddingResult(
        task_id=task_id, model="BAAI/bge-m3", embedding_types=["dense"], data=[]
    )

    def unavailable(*args, **kwargs):
        raise ConnectionError("offline")

    monkeypatch.setattr(store, "finish", unavailable)
    with pytest.raises(Reject) as error:
        run_task(task_id, {"input": []})
    assert error.value.requeue is True
    assert store.get(task_id).status == "running"


@pytest.mark.parametrize(
    "selected", [["dense"], ["sparse"], ["colbert"], ["dense", "sparse", "colbert"]]
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
    request = EmbeddingRequest(input=["text"], embedding_types=selected)
    result = inference.encode(str(uuid4()), request)
    flags = engine.encode.call_args.kwargs
    assert flags["max_length"] == settings.max_input_tokens
    assert flags["return_dense"] == ("dense" in selected)
    assert flags["return_sparse"] == ("sparse" in selected)
    assert flags["return_colbert_vecs"] == ("colbert" in selected)
    data = result.model_dump(mode="json", exclude_none=True)["data"][0]
    assert set(data) == {"index", *selected}


def test_empty_batch_skips_model(settings, tokenizer, engine):
    inference = InferenceEngine.__new__(InferenceEngine)
    inference.settings = settings
    inference.validator = TextValidator(tokenizer, settings)
    inference.model = engine
    assert inference.encode(str(uuid4()), EmbeddingRequest(input=[])).data == []
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


def test_queue_has_late_ack_and_no_prefetch(settings):
    app = create_celery_app(settings)
    assert app.conf.task_acks_late
    assert app.conf.task_reject_on_worker_lost
    assert app.conf.worker_concurrency == 1
    assert app.conf.worker_pool == "solo"
    assert app.conf.worker_prefetch_multiplier == 1
    assert app.conf.broker_transport_options["priority_steps"] == [0]
