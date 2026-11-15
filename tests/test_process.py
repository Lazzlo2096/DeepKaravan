import os
import time
from functools import partial

import pytest

from api.service import QueueUnavailable, TaskService
from common.schemas import EmbeddingRequest, EmbeddingResult, TextEmbedding
from gpu_worker.process import run_worker


class FakeEngine:
    def __init__(self, settings):
        pass

    def encode(self, task_id, request):
        if request.input == ["crash"]:
            os._exit(7)
        if request.input == ["fail"]:
            raise RuntimeError("sensitive details")
        if request.input == ["slow"]:
            time.sleep(30)
        if request.input == ["nan"]:
            return EmbeddingResult(
                task_id=task_id,
                model="BAAI/bge-m3",
                embedding_types=["dense"],
                data=[TextEmbedding(index=0, embedding=[float("nan")])],
            )
        return EmbeddingResult(
            task_id=task_id,
            model="BAAI/bge-m3",
            embedding_types=request.embedding_types,
            data=[
                TextEmbedding(index=i, embedding=[float(os.getpid())] * 2048)
                for i in range(len(request.input))
            ],
        )


def broken_factory(settings):
    raise RuntimeError("startup failure")


def stalled_worker(settings, requests, results, stop):
    results.put(("ready", None, None))
    stop.wait(30)


def never_ready_worker(settings, requests, results, stop):
    stop.wait(30)


def wait_terminal(service, task_id):
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        task = service.status(task_id)
        if task.status in {"succeeded", "failed"}:
            return task
        time.sleep(0.01)
    pytest.fail("Task did not complete")


@pytest.fixture
def runtime(settings):
    service = TaskService(settings, worker_target=partial(run_worker, engine_factory=FakeEngine))
    service.start()
    yield service
    service.close()
    assert not service._process.is_alive()
    assert not service._collector.is_alive()


def request(text):
    return EmbeddingRequest(model="baai/bge-m3", input=[text])


def test_spawn_fifo_and_large_results(runtime):
    jobs = [runtime.submit(request(str(i))) for i in range(12)]
    for job in jobs:
        assert wait_terminal(runtime, job.task_id).status == "succeeded"
        result = runtime.store.get_result(job.task_id)
        assert result.data[0].embedding[0] == runtime._process.pid
        assert result.data[0].embedding[0] != os.getpid()
    starts = [runtime.status(job.task_id).started_at for job in jobs]
    assert starts == sorted(starts)


def test_task_failure_does_not_stop_worker(runtime):
    failed = runtime.submit(request("fail"))
    task = wait_terminal(runtime, failed.task_id)
    assert task.error.code == "inference_failed"
    assert "sensitive" not in task.error.message
    success = runtime.submit(request("ok"))
    assert wait_terminal(runtime, success.task_id).status == "succeeded"


def test_invalid_result_does_not_stop_collector(runtime):
    failed = runtime.submit(request("nan"))
    assert wait_terminal(runtime, failed.task_id).error.code == "invalid_result"
    success = runtime.submit(request("ok"))
    assert wait_terminal(runtime, success.task_id).status == "succeeded"


def test_worker_crash_fails_waiting_tasks(runtime):
    jobs = [runtime.submit(request("crash")), runtime.submit(request("ok"))]
    for job in jobs:
        assert wait_terminal(runtime, job.task_id).error.code == "worker_exited"
    assert not runtime.ready
    with pytest.raises(QueueUnavailable):
        runtime.submit(request("ok"))


def test_startup_failure_is_propagated_and_process_cleaned(settings):
    service = TaskService(
        settings, worker_target=partial(run_worker, engine_factory=broken_factory)
    )
    with pytest.raises(RuntimeError):
        service.start()
    assert not service._process.is_alive()
    assert not service._collector.is_alive()


def test_startup_timeout_cleans_up(settings):
    settings.worker_startup_timeout_seconds = 0.1
    service = TaskService(settings, worker_target=never_ready_worker)
    with pytest.raises(TimeoutError):
        service.start()
    service.close()  # Shutdown is idempotent, including failed startup.
    assert not service._process.is_alive()


def test_real_api_to_spawn_process(monkeypatch, settings, tokenizer):
    from fastapi.testclient import TestClient

    from api.main import create_app

    monkeypatch.setattr("api.main.load_tokenizer", lambda settings: tokenizer)
    service = TaskService(settings, worker_target=partial(run_worker, engine_factory=FakeEngine))
    with TestClient(create_app(settings, service)) as client:
        body = {"model": "BAAI/BGE-M3", "input": ["text"]}
        response = client.post("/api/v1/embeddings", json=body)
        assert response.status_code == 200
        assert response.json()["data"][0]["embedding"][0] == service._process.pid
        accepted = client.post("/api/v1/embeddings/async", json=body)
        assert accepted.status_code == 202
        task_id = accepted.json()["task_id"]
        wait_terminal(service, task_id)
        result = client.get(accepted.json()["result_url"])
        assert result.status_code == 200
        assert result.json()["data"][0]["object"] == "embedding"
        response = client.post("/api/v1/embeddings", json={**body, "input": ["crash"]})
        assert response.status_code == 422
        assert response.json()["detail"]["error"]["code"] == "worker_exited"
        assert client.get("/health/ready").status_code == 503
        assert client.get("/health/live").status_code == 200


def test_queue_full_does_not_create_orphan_job(settings):
    settings.queue_capacity = 1
    service = TaskService(settings, worker_target=stalled_worker)
    service.start()
    try:
        job = service.submit(request("ok"))
        with pytest.raises(QueueUnavailable):
            service.submit(request("ok"))
        assert len(service.store._statuses) == 1
    finally:
        service.close()
    assert service.status(job.task_id).status == "failed"


def test_forced_shutdown_is_bounded(settings):
    settings.worker_shutdown_timeout_seconds = 0.1
    service = TaskService(settings, worker_target=partial(run_worker, engine_factory=FakeEngine))
    service.start()
    job = service.submit(request("slow"))
    deadline = time.monotonic() + 5
    while service.status(job.task_id).status == "queued" and time.monotonic() < deadline:
        time.sleep(0.01)
    service.close()
    assert not service._process.is_alive()
    assert service.status(job.task_id).status == "failed"
