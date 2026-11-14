from celery import Celery

from common.config import Settings

TASK_NAME = "deepkaravan.embed"


def create_celery_app(settings: Settings) -> Celery:
    app = Celery("deepkaravan", broker=settings.redis_broker_url)
    app.conf.update(
        task_default_queue=settings.task_queue,
        task_serializer="json",
        accept_content=["json"],
        # Status and embeddings are stored by JobStore in Redis DB 1. This avoids
        # storing a second copy of potentially large ColBERT results in Celery.
        task_ignore_result=True,
        task_acks_late=True,
        task_reject_on_worker_lost=True,
        worker_concurrency=1,
        worker_pool="solo",
        worker_prefetch_multiplier=1,
        broker_connection_retry_on_startup=True,
        broker_connection_timeout=settings.redis_socket_timeout_seconds,
        broker_transport_options={
            "visibility_timeout": settings.visibility_timeout_seconds,
            "priority_steps": [0],
            "socket_timeout": settings.redis_socket_timeout_seconds,
            "socket_connect_timeout": settings.redis_socket_timeout_seconds,
        },
        task_publish_retry=False,
        enable_utc=True,
        timezone="UTC",
    )
    return app
