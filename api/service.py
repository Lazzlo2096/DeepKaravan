from uuid import UUID, uuid4

from celery import Celery
from kombu.exceptions import OperationalError
from redis.exceptions import RedisError

from common.celery_app import TASK_NAME
from common.jobs import JobStore
from common.schemas import EmbeddingRequest, TaskError, TaskStatus


class QueueUnavailable(RuntimeError):
    pass


class TaskService:
    def __init__(self, celery: Celery, store: JobStore):
        self.celery = celery
        self.store = store

    def submit(self, request: EmbeddingRequest) -> TaskStatus:
        task_id = uuid4()
        status = self.store.create(task_id)
        try:
            self.celery.send_task(
                TASK_NAME,
                args=[request.model_dump(mode="json")],
                task_id=str(task_id),
                retry=False,
            )
        except (OperationalError, RedisError, OSError) as exc:
            self.store.finish(
                task_id,
                error=TaskError(code="queue_unavailable", message="Could not publish task"),
            )
            raise QueueUnavailable("Task queue is unavailable") from exc
        return status

    def status(self, task_id: UUID) -> TaskStatus | None:
        return self.store.get(task_id)
