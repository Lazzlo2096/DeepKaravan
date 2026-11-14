from datetime import UTC, datetime, timedelta
from uuid import UUID

from redis import Redis

from common.config import Settings
from common.schemas import EmbeddingResult, TaskError, TaskState, TaskStatus


def create_redis(settings: Settings) -> Redis:
    return Redis.from_url(
        settings.redis_results_url,
        decode_responses=True,
        socket_connect_timeout=settings.redis_socket_timeout_seconds,
        socket_timeout=settings.redis_socket_timeout_seconds,
        health_check_interval=30,
    )


class JobStore:
    """Queue/running metadata has no TTL. Terminal metadata and results expire together."""

    def __init__(self, redis: Redis, ttl: int):
        self.redis = redis
        self.ttl = ttl

    @staticmethod
    def status_key(task_id: str | UUID) -> str:
        return f"deepkaravan:task:{task_id}"

    @staticmethod
    def result_key(task_id: str | UUID) -> str:
        return f"deepkaravan:result:{task_id}"

    def create(self, task_id: str | UUID) -> TaskStatus:
        status = TaskStatus(
            task_id=UUID(str(task_id)), status=TaskState.QUEUED, submitted_at=datetime.now(UTC)
        )
        self.redis.set(self.status_key(task_id), status.model_dump_json(), nx=True)
        return status

    def get(self, task_id: str | UUID) -> TaskStatus | None:
        raw = self.redis.get(self.status_key(task_id))
        return TaskStatus.model_validate_json(raw) if raw else None

    def start(self, task_id: str | UUID) -> TaskStatus:
        status = self.get(task_id)
        if status is None:
            raise ValueError(f"Unknown task {task_id}")
        status.status = TaskState.RUNNING
        status.started_at = datetime.now(UTC)
        # SET removes any old TTL; a restarted task must not expire while running.
        self.redis.set(self.status_key(task_id), status.model_dump_json())
        return status

    def finish(
        self,
        task_id: str | UUID,
        *,
        result: EmbeddingResult | None = None,
        error: TaskError | None = None,
    ) -> TaskStatus:
        if (result is None) == (error is None):
            raise ValueError("Provide exactly one of result or error")
        status = self.get(task_id)
        if status is None:
            raise ValueError(f"Unknown task {task_id}")
        status.status = TaskState.FAILED if error else TaskState.SUCCEEDED
        status.finished_at = datetime.now(UTC)
        status.expires_at = status.finished_at + timedelta(seconds=self.ttl)
        status.error = error
        # Atomic terminal transition: clients never see success without a result.
        with self.redis.pipeline(transaction=True) as pipe:
            if result is not None:
                pipe.set(
                    self.result_key(task_id), result.model_dump_json(exclude_none=True), ex=self.ttl
                )
            pipe.set(self.status_key(task_id), status.model_dump_json(), ex=self.ttl)
            pipe.execute()
        return status

    def get_result(self, task_id: str | UUID) -> EmbeddingResult | None:
        raw = self.redis.get(self.result_key(task_id))
        return EmbeddingResult.model_validate_json(raw) if raw else None
