from datetime import UTC, datetime, timedelta
from threading import RLock
from uuid import UUID

from common.schemas import TERMINAL_STATES, EmbeddingResult, TaskError, TaskState, TaskStatus


class JobStore:
    """In-memory task state owned by the API. Restarting the API clears all jobs."""

    def __init__(self, ttl: int):
        self.ttl = ttl
        self._statuses: dict[str, TaskStatus] = {}
        self._results: dict[str, EmbeddingResult] = {}
        self._lock = RLock()

    def prune(self) -> None:
        now = datetime.now(UTC)
        with self._lock:
            for key, task in list(self._statuses.items()):
                if task.expires_at is not None and task.expires_at <= now:
                    self.discard(key)

    def create(self, task_id: str | UUID) -> TaskStatus:
        task = TaskStatus(
            task_id=UUID(str(task_id)), status=TaskState.QUEUED, submitted_at=datetime.now(UTC)
        )
        with self._lock:
            self._statuses[str(task_id)] = task
        return task.model_copy(deep=True)

    def get(self, task_id: str | UUID) -> TaskStatus | None:
        with self._lock:
            task = self._statuses.get(str(task_id))
            if task is not None and task.expires_at and task.expires_at <= datetime.now(UTC):
                self.discard(task_id)
                return None
            return task.model_copy(deep=True) if task else None

    def discard(self, task_id: str | UUID) -> None:
        with self._lock:
            self._statuses.pop(str(task_id), None)
            self._results.pop(str(task_id), None)

    def start(self, task_id: str | UUID) -> TaskStatus:
        with self._lock:
            task = self._statuses[str(task_id)]
            if task.status not in TERMINAL_STATES:
                task.status = TaskState.RUNNING
                task.started_at = datetime.now(UTC)
            return task.model_copy(deep=True)

    def finish(
        self,
        task_id: str | UUID,
        *,
        result: EmbeddingResult | None = None,
        error: TaskError | None = None,
    ) -> TaskStatus:
        if (result is None) == (error is None):
            raise ValueError("Provide exactly one of result or error")
        with self._lock:
            task = self._statuses[str(task_id)]
            if task.status in TERMINAL_STATES:
                return task.model_copy(deep=True)
            if result is not None:
                self._results[str(task_id)] = result.model_copy(deep=True)
            task.status = TaskState.FAILED if error else TaskState.SUCCEEDED
            task.error = error
            task.finished_at = datetime.now(UTC)
            task.expires_at = task.finished_at + timedelta(seconds=self.ttl)
            return task.model_copy(deep=True)

    def fail_pending(self, code: str, message: str) -> None:
        with self._lock:
            for task in list(self._statuses.values()):
                if task.status not in TERMINAL_STATES:
                    self.finish(task.task_id, error=TaskError(code=code, message=message))

    def get_result(self, task_id: str | UUID) -> EmbeddingResult | None:
        with self._lock:
            if self.get(task_id) is None:
                return None
            result = self._results.get(str(task_id))
            return result.model_copy(deep=True) if result else None
