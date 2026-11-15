from collections.abc import Callable
from dataclasses import dataclass, field
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing

from api.service import QueueUnavailable
from common.config import Settings
from common.jobs import JobStore


@pytest.fixture
def settings(request):
    return Settings(
        _env_file=None,
        poll_interval_seconds=0.005,
        max_input_tokens=6,
        worker_startup_timeout_seconds=10,
        worker_shutdown_timeout_seconds=0.2,
        **getattr(request, "param", {}),
    )


@pytest.fixture
def store(settings):
    return JobStore(settings.result_ttl_seconds)


@pytest.fixture
def tokenizer():
    tokenizer = Tokenizer(WordLevel({"<unk>": 0, "<s>": 1, "</s>": 2, "text": 3}, "<unk>"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.post_processor = TemplateProcessing(
        single="<s> $A </s>", special_tokens=[("<s>", 1), ("</s>", 2)]
    )
    return tokenizer


@dataclass
class Publisher:
    callback: Callable | None = None
    sent: list = field(default_factory=list)

    ready: bool = True
    store: object = None

    def start(self):
        pass

    def submit(self, request):
        from uuid import uuid4

        if not self.ready:
            raise QueueUnavailable("GPU worker is unavailable")
        task_id = str(uuid4())
        payload = request.model_dump(mode="json")
        task = self.store.create(task_id)
        self.sent.append({"args": [payload], "task_id": task_id})
        if self.callback:
            self.callback(task_id, payload)
        return task

    def status(self, task_id):
        return self.store.get(task_id)

    def close(self):
        pass


@pytest.fixture
def publisher(store):
    return Publisher(store=store)


@pytest.fixture
def client(monkeypatch, settings, tokenizer, publisher):
    import api.main as main

    monkeypatch.setattr(main, "load_tokenizer", lambda settings: tokenizer)
    with TestClient(main.create_app(settings, service=publisher)) as client:
        yield client


@pytest.fixture
def engine():
    return Mock()
