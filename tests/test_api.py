from concurrent.futures import ThreadPoolExecutor
from threading import Event
from time import sleep
from uuid import UUID, uuid4

import pytest
from kombu.exceptions import OperationalError
from redis.exceptions import ConnectionError

from common.schemas import EmbeddingResult, TaskError, TextEmbedding


def complete(store, task_id, payload):
    store.start(task_id)
    store.finish(
        task_id,
        result=EmbeddingResult(
            task_id=UUID(task_id),
            model="BAAI/bge-m3",
            embedding_types=payload["embedding_types"],
            data=[TextEmbedding(index=i, dense=[0.1, 0.2]) for i in range(len(payload["input"]))],
        ),
    )


def test_async_accepts_and_uses_shared_queue(client, publisher, store):
    response = client.post("/v1/embeddings/async", json={"input": ["text"]})
    assert response.status_code == 202
    body = response.json()
    assert body["status"] == "queued"
    assert response.headers["location"] == body["status_url"]
    assert response.headers["retry-after"] == "1"
    assert store.get(body["task_id"]).status == "queued"
    assert publisher.sent[0]["args"][0]["embedding_types"] == ["dense"]
    assert client.get(body["result_url"]).status_code == 202


@pytest.mark.parametrize("embedding_types", [None, ["dense"]])
@pytest.mark.parametrize("model", [None, "baai/bge-m3"])
def test_sync_returns_result_when_ready(client, publisher, store, embedding_types, model):
    publisher.callback = lambda task_id, payload: complete(store, task_id, payload)
    payload = {"input": ["text"]}
    if model is not None:
        payload["model"] = model
    if embedding_types is not None:
        payload["embedding_types"] = embedding_types
    response = client.post("/v1/embeddings", json=payload)
    assert response.status_code == 200
    assert response.json()["embedding_types"] == ["dense"]
    assert publisher.sent[0]["args"][0] == {
        "input": ["text"],
        "model": "baai/bge-m3",
        "embedding_types": ["dense"],
    }
    assert response.json()["data"] == [{"index": 0, "dense": [0.1, 0.2]}]


@pytest.mark.parametrize("settings", [{"sync_timeout_seconds": 0.01}], indirect=True)
@pytest.mark.parametrize("failed", [False, True])
def test_sync_waits_through_queue_and_inference_without_timeout(client, publisher, store, failed):
    # A legacy timeout setting must not make the API return a task_id early.
    published = Event()
    publisher.callback = lambda task_id, payload: published.set()
    with ThreadPoolExecutor(max_workers=1) as executor:
        response_future = executor.submit(client.post, "/v1/embeddings", json={"input": ["text"]})
        assert published.wait(timeout=5)
        task_id = publisher.sent[0]["task_id"]
        try:
            sleep(0.05)
            assert not response_future.done(), "Request returned while the task was queued"
            store.start(task_id)
            sleep(0.05)
            assert not response_future.done(), "Request returned while inference was running"
            assert client.get("/health/live").status_code == 200
        finally:
            if failed:
                store.finish(
                    task_id, error=TaskError(code="inference_failed", message="test failure")
                )
            else:
                complete(store, task_id, publisher.sent[0]["args"][0])
        response = response_future.result(timeout=5)
    assert response.status_code == (422 if failed else 200)
    assert "location" not in response.headers
    assert "retry-after" not in response.headers
    if failed:
        assert response.json()["detail"]["error"]["code"] == "inference_failed"
    else:
        assert response.json()["data"] == [{"index": 0, "dense": [0.1, 0.2]}]


def test_async_task_can_still_be_polled_until_complete(client, publisher, store):
    response = client.post("/v1/embeddings/async", json={"input": ["text"]})
    assert response.status_code == 202
    body = response.json()
    assert client.get(body["status_url"]).json()["status"] == "queued"
    complete(store, body["task_id"], publisher.sent[0]["args"][0])
    assert client.get(body["status_url"]).json()["status"] == "succeeded"
    result = client.get(body["result_url"])
    assert result.status_code == 200
    assert result.json()["task_id"] == body["task_id"]


@pytest.mark.parametrize("path", ["/v1/embeddings", "/v1/embeddings/async"])
def test_long_text_rejected_before_enqueue(client, publisher, path):
    response = client.post(path, json={"input": ["text", "text text text text text"]})
    assert response.status_code == 422
    assert response.json()["detail"] == {
        "code": "text_too_long",
        "indices": [1],
        "max_input_tokens": 6,
    }
    assert publisher.sent == []


def test_exact_token_limit_allowed(client):
    response = client.post("/v1/embeddings/async", json={"input": ["text text text text"]})
    assert response.status_code == 202


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"texts": ["text"]},
        {"input": ["text"], "texts": ["text"]},
        {"input": "text"},
        {"input": [5]},
        {"input": [None]},
        {"input": ["text"], "embedding_types": []},
        {"input": ["text"], "embedding_types": None},
        {"input": ["text"], "embedding_types": ["dense", "dense"]},
        {"input": ["text"], "embedding_types": ["unknown"]},
        {"input": ["text"] * 1025},
        {"input": ["text"], "unexpected": True},
    ],
)
@pytest.mark.parametrize("path", ["/v1/embeddings", "/v1/embeddings/async"])
def test_invalid_request_rejected(client, publisher, payload, path):
    assert client.post(path, json=payload).status_code == 422
    assert publisher.sent == []


@pytest.mark.parametrize("path", ["/v1/embeddings", "/v1/embeddings/async"])
@pytest.mark.parametrize("model", ["other-model", "BAAI/bge-m3", "", None, 123, ["baai/bge-m3"]])
def test_unsupported_model_rejected_before_enqueue(client, publisher, path, model):
    response = client.post(path, json={"input": ["text"], "model": model})
    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == ["body", "model"]
    assert publisher.sent == []


def test_async_accepts_explicit_model(client, publisher):
    response = client.post("/v1/embeddings/async", json={"input": ["text"], "model": "baai/bge-m3"})
    assert response.status_code == 202
    assert publisher.sent[0]["args"][0]["model"] == "baai/bge-m3"


def test_empty_batch_accepted(client, publisher, store):
    publisher.callback = lambda task_id, payload: complete(store, task_id, payload)
    response = client.post("/v1/embeddings", json={"input": []})
    assert response.status_code == 200
    assert response.json()["data"] == []


@pytest.mark.parametrize("suffix", ["", "/result"])
def test_unknown_or_expired_task_is_404(client, suffix):
    assert client.get(f"/v1/tasks/{uuid4()}{suffix}").status_code == 404


def test_invalid_task_id(client):
    assert client.get("/v1/tasks/not-a-uuid").status_code == 422


def test_failed_task_has_status_and_error(client, store, publisher):
    def fail(task_id, payload):
        store.finish(task_id, error=TaskError(code="inference_failed", message="test failure"))

    publisher.callback = fail
    response = client.post("/v1/embeddings", json={"input": ["text"]})
    assert response.status_code == 422
    task_id = response.json()["detail"]["task_id"]
    assert client.get(f"/v1/tasks/{task_id}").json()["status"] == "failed"
    assert client.get(f"/v1/tasks/{task_id}/result").status_code == 422


def test_broker_outage_returns_503(client, publisher):
    def fail(task_id, payload):
        raise OperationalError("broker is offline")

    publisher.callback = fail
    assert client.post("/v1/embeddings/async", json={"input": []}).status_code == 503


def test_redis_outage_returns_503(client, monkeypatch):
    def fail(*args, **kwargs):
        raise ConnectionError("Redis is offline")

    monkeypatch.setattr(client.app.state.service.store.redis, "get", fail)
    assert client.get(f"/v1/tasks/{uuid4()}").status_code == 503


def test_health_and_openapi(client):
    assert client.get("/health/live").status_code == 200
    assert client.get("/health/ready").status_code == 200
    schema = client.get("/openapi.json").json()
    request_schema = schema["components"]["schemas"]["EmbeddingRequest"]
    assert request_schema["required"] == ["input"]
    assert request_schema["properties"]["model"]["const"] == "baai/bge-m3"
    assert request_schema["properties"]["model"]["default"] == "baai/bge-m3"
    assert "texts" not in request_schema["properties"]
    assert "202" not in schema["paths"]["/v1/embeddings"]["post"]["responses"]
    assert "202" in schema["paths"]["/v1/embeddings/async"]["post"]["responses"]
    assert "202" in schema["paths"]["/v1/tasks/{task_id}/result"]["get"]["responses"]
