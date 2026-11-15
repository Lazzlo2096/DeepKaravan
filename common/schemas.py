from datetime import datetime
from enum import StrEnum
from typing import Annotated
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, StrictStr, field_validator


class EmbeddingType(StrEnum):
    DENSE = "dense"
    SPARSE = "sparse"
    COLBERT = "colbert"


class EmbeddingRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input: Annotated[list[StrictStr], Field(max_length=1024)]
    embedding_types: Annotated[list[EmbeddingType], Field(min_length=1, max_length=3)] = Field(
        default_factory=lambda: [EmbeddingType.DENSE]
    )

    @field_validator("embedding_types")
    @classmethod
    def unique_types(cls, value: list[EmbeddingType]) -> list[EmbeddingType]:
        if len(value) != len(set(value)):
            raise ValueError("embedding_types must not contain duplicates")
        return value


class TaskState(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


TERMINAL_STATES = {TaskState.SUCCEEDED, TaskState.FAILED}


class TaskError(BaseModel):
    code: str
    message: str


class TaskStatus(BaseModel):
    task_id: UUID
    status: TaskState
    submitted_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    expires_at: datetime | None = None
    error: TaskError | None = None


class TaskAccepted(TaskStatus):
    status_url: str
    result_url: str


class TextEmbedding(BaseModel):
    index: int
    dense: list[float] | None = None
    # Keys are tokenizer token IDs, not decoded words.
    sparse: dict[str, float] | None = None
    colbert: list[list[float]] | None = None


class EmbeddingResult(BaseModel):
    task_id: UUID
    model: str
    embedding_types: list[EmbeddingType]
    data: list[TextEmbedding]
