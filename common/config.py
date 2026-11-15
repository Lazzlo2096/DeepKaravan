from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    result_ttl_seconds: int = Field(default=86400, ge=1)
    poll_interval_seconds: float = Field(default=0.01, gt=0)
    queue_capacity: int = Field(default=128, ge=1)
    worker_startup_timeout_seconds: float = Field(default=600, gt=0)
    worker_shutdown_timeout_seconds: float = Field(default=30, gt=0)

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
