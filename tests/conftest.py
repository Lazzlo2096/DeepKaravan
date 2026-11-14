from collections.abc import Callable
from dataclasses import dataclass, field
from unittest.mock import Mock

import fakeredis
import pytest
from fastapi.testclient import TestClient
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing

from common.config import Settings
from common.jobs import JobStore


@pytest.fixture
def settings():
    return Settings(
        _env_file=None,
        sync_timeout_seconds=0.04,
        poll_interval_seconds=0.005,
        max_input_tokens=6,
    )


@pytest.fixture
def redis():
    return fakeredis.FakeRedis(decode_responses=True)


@pytest.fixture
def store(redis, settings):
    return JobStore(redis, settings.result_ttl_seconds)


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

    def send_task(self, name, args, task_id, retry):
        self.sent.append({"name": name, "args": args, "task_id": task_id})
        if self.callback:
            self.callback(task_id, args[0])

    def close(self):
        pass


@pytest.fixture
def publisher():
    return Publisher()


@pytest.fixture
def client(monkeypatch, settings, redis, tokenizer, publisher):
    import api.main as main

    monkeypatch.setattr(main, "create_redis", lambda settings: redis)
    monkeypatch.setattr(main, "create_celery_app", lambda settings: publisher)
    monkeypatch.setattr(main, "load_tokenizer", lambda settings: tokenizer)
    with TestClient(main.create_app(settings)) as client:
        yield client


@pytest.fixture
def engine():
    return Mock()
