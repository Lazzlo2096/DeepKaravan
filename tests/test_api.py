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
            data=[TextEmbedding(index=i, dense=[0.1, 0.2]) for i in range(len(payload["texts"]))],
        ),
    )


def test_async_accepts_and_uses_shared_queue(client, publisher, store):
    response = client.post("/v1/embeddings/async", json={"texts": ["text"]})
    assert response.status_code == 202
    body = response.json()
    assert body["status"] == "queued"
    assert response.headers["location"] == body["status_url"]
    assert response.headers["retry-after"] == "1"
    assert store.get(body["task_id"]).status == "queued"
    assert publisher.sent[0]["args"][0]["embedding_types"] == ["dense", "sparse", "colbert"]
    assert client.get(body["result_url"]).status_code == 202


def test_sync_returns_result_when_ready(client, publisher, store):
    publisher.callback = lambda task_id, payload: complete(store, task_id, payload)
    response = client.post("/v1/embeddings", json={"texts": ["text"], "embedding_types": ["dense"]})
    assert response.status_code == 200
    assert response.json()["data"] == [{"index": 0, "dense": [0.1, 0.2]}]


def test_sync_timeout_keeps_task_and_can_poll_result(client, publisher, store):
    response = client.post("/v1/embeddings", json={"texts": ["text"]})
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
    response = client.post(path, json={"texts": ["text", "text text text text text"]})
    assert response.status_code == 422
    assert response.json()["detail"] == {
        "code": "text_too_long",
        "indices": [1],
        "max_input_tokens": 6,
    }
    assert publisher.sent == []


def test_exact_token_limit_allowed(client):
    response = client.post("/v1/embeddings/async", json={"texts": ["text text text text"]})
    assert response.status_code == 202


@pytest.mark.parametrize(
    "payload",
    [
        {"texts": "text"},
        {"texts": [5]},
        {"texts": [None]},
        {"texts": ["text"], "embedding_types": []},
        {"texts": ["text"], "embedding_types": ["dense", "dense"]},
        {"texts": ["text"], "embedding_types": ["unknown"]},
        {"texts": ["text"] * 1025},
        {"texts": ["text"], "unexpected": True},
    ],
)
def test_invalid_request_rejected(client, publisher, payload):
    assert client.post("/v1/embeddings/async", json=payload).status_code == 422
    assert publisher.sent == []


def test_empty_batch_accepted(client, publisher, store):
    publisher.callback = lambda task_id, payload: complete(store, task_id, payload)
    response = client.post("/v1/embeddings", json={"texts": []})
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
    response = client.post("/v1/embeddings", json={"texts": ["text"]})
    assert response.status_code == 422
    task_id = response.json()["detail"]["task_id"]
    assert client.get(f"/v1/tasks/{task_id}").json()["status"] == "failed"
    assert client.get(f"/v1/tasks/{task_id}/result").status_code == 422


def test_broker_outage_returns_503(client, publisher):
    def fail(task_id, payload):
        raise OperationalError("broker is offline")

    publisher.callback = fail
    assert client.post("/v1/embeddings/async", json={"texts": []}).status_code == 503


def test_redis_outage_returns_503(client, monkeypatch):
    def fail(*args, **kwargs):
        raise ConnectionError("Redis is offline")

    monkeypatch.setattr(client.app.state.service.store.redis, "get", fail)
    assert client.get(f"/v1/tasks/{uuid4()}").status_code == 503


def test_health_and_openapi(client):
    assert client.get("/health/live").status_code == 200
    assert client.get("/health/ready").status_code == 200
    schema = client.get("/openapi.json").json()
    assert "202" in schema["paths"]["/v1/embeddings"]["post"]["responses"]
