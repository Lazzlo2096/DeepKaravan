import logging
from functools import lru_cache

from celery.exceptions import Reject
from redis.exceptions import RedisError

from common.celery_app import TASK_NAME, create_celery_app
from common.config import get_settings
from common.jobs import JobStore, create_redis
from common.schemas import TERMINAL_STATES, EmbeddingRequest, TaskError
from worker.inference import get_engine

logger = logging.getLogger(__name__)
settings = get_settings()
app = create_celery_app(settings)


@lru_cache(maxsize=1)
def get_store() -> JobStore:
    return JobStore(create_redis(settings), settings.result_ttl_seconds)


@app.task(bind=True, name=TASK_NAME, ignore_result=True)
def embed(self, payload: dict) -> None:
    task_id = self.request.id
    store = get_store()
    try:
        existing = store.get(task_id)
        # A crash after persisting the result but before ACK may redeliver the
        # message. Preserve its original result and expiration in that case.
        if existing is not None and existing.status in TERMINAL_STATES:
            return
        if existing is None:
            store.create(task_id)
        store.start(task_id)
        try:
            request = EmbeddingRequest.model_validate(payload)
            result = get_engine().encode(task_id, request)
        except Exception as exc:
            logger.exception("Inference failed for task %s", task_id)
            store.finish(
                task_id,
                error=TaskError(
                    code="inference_failed",
                    message=f"Inference failed ({type(exc).__name__}); see worker logs",
                ),
            )
        else:
            store.finish(task_id, result=result)
    except RedisError as exc:
        # Never ACK an inference whose result could not be persisted.
        raise Reject("Redis unavailable; requeue task", requeue=True) from exc
