from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    redis_broker_url: str = "redis://localhost:6379/0"
    redis_results_url: str = "redis://localhost:6379/1"
    task_queue: str = "embeddings"
    result_ttl_seconds: int = Field(default=86400, ge=1)
    poll_interval_seconds: float = Field(default=0.25, gt=0)
    redis_socket_timeout_seconds: float = Field(default=5, gt=0)
    # Must exceed the longest expected inference. Abrupt recovery can take this long.
    visibility_timeout_seconds: int = Field(default=3600, ge=1)

    model_id: Literal["BAAI/bge-m3"] = "BAAI/bge-m3"
    model_revision: str = "5617a9f61b028005a4858fdac845db406aefb181"
    inference_device: Literal["cpu", "amd", "amd-wsl"] = "cpu"
    inference_batch_size: int = Field(default=8, ge=1, le=1024)
    use_fp16: bool = False
    max_texts: int = Field(default=1024, ge=1, le=1024)
    max_input_tokens: int = Field(default=8192, ge=2, le=8192)


@lru_cache
def get_settings() -> Settings:
    return Settings()
