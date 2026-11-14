import asyncio
import logging
from contextlib import asynccontextmanager
from time import monotonic
from uuid import UUID

from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.responses import JSONResponse
from redis.exceptions import RedisError
from starlette.concurrency import run_in_threadpool

from api.service import QueueUnavailable, TaskService
from common.celery_app import create_celery_app
from common.config import Settings, get_settings
from common.jobs import JobStore, create_redis
from common.schemas import (
    EmbeddingRequest,
    EmbeddingResult,
    TaskAccepted,
    TaskState,
    TaskStatus,
)
from common.tokenization import InputTooLong, TextValidator, load_tokenizer

logger = logging.getLogger(__name__)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        redis = create_redis(settings)
        celery = create_celery_app(settings)
        try:
            await run_in_threadpool(redis.ping)
            tokenizer = await run_in_threadpool(load_tokenizer, settings)
            app.state.validator = TextValidator(tokenizer, settings)
            app.state.service = TaskService(celery, JobStore(redis, settings.result_ttl_seconds))
            yield
        finally:
            celery.close()
            await run_in_threadpool(redis.close)

    app = FastAPI(
        title="DeepKaravan BGE-M3",
        version="0.1.0",
        description="One FIFO queue for synchronous and asynchronous BGE-M3 inference.",
        lifespan=lifespan,
    )

    @app.exception_handler(RedisError)
    async def redis_error_handler(request: Request, exc: RedisError):
        logger.warning("Redis unavailable: %s", type(exc).__name__)
        return JSONResponse(status_code=503, content={"detail": "Redis is unavailable"})

    @app.exception_handler(QueueUnavailable)
    async def queue_error_handler(request: Request, exc: QueueUnavailable):
        return JSONResponse(status_code=503, content={"detail": str(exc)})

    async def submit(request: Request, payload: EmbeddingRequest) -> TaskStatus:
        try:
            await run_in_threadpool(request.app.state.validator.validate, payload.texts)
        except InputTooLong as exc:
            raise HTTPException(
                status_code=422,
                detail={
                    "code": "text_too_long",
                    "indices": exc.indices,
                    "max_input_tokens": exc.limit,
                },
            ) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return await run_in_threadpool(request.app.state.service.submit, payload)

    def accepted(task: TaskStatus, response: Response) -> TaskAccepted:
        status_url = f"/v1/tasks/{task.task_id}"
        result_url = f"{status_url}/result"
        response.status_code = status.HTTP_202_ACCEPTED
        response.headers["Location"] = status_url
        response.headers["Retry-After"] = "1"
        return TaskAccepted(**task.model_dump(), status_url=status_url, result_url=result_url)

    async def get_status(request: Request, task_id: UUID) -> TaskStatus:
        task = await run_in_threadpool(request.app.state.service.status, task_id)
        if task is None:
            raise HTTPException(status_code=404, detail="Task not found or expired")
        return task

    async def completed_result(request: Request, task: TaskStatus) -> EmbeddingResult:
        if task.status == TaskState.FAILED:
            raise HTTPException(
                status_code=422,
                detail={"task_id": str(task.task_id), "error": task.error.model_dump()},
            )
        result = await run_in_threadpool(request.app.state.service.store.get_result, task.task_id)
        if result is None:
            raise HTTPException(status_code=404, detail="Result expired")
        return result

    @app.get("/health/live", tags=["health"])
    async def live():
        return {"status": "ok"}

    @app.get("/health/ready", tags=["health"])
    async def ready(request: Request):
        await run_in_threadpool(request.app.state.service.store.redis.ping)
        return {"status": "ok"}

    @app.post(
        "/v1/embeddings/async",
        response_model=TaskAccepted,
        status_code=202,
        tags=["embeddings"],
    )
    async def embed_async(payload: EmbeddingRequest, request: Request, response: Response):
        task = await submit(request, payload)
        return accepted(task, response)

    @app.post(
        "/v1/embeddings",
        response_model=EmbeddingResult,
        response_model_exclude_none=True,
        responses={202: {"model": TaskAccepted}},
        tags=["embeddings"],
    )
    async def embed_sync(payload: EmbeddingRequest, request: Request, response: Response):
        # The deadline includes validation, publication and time spent in the queue.
        deadline = monotonic() + settings.sync_timeout_seconds
        task = await submit(request, payload)
        while True:
            task = await get_status(request, task.task_id)
            if task.status in {TaskState.SUCCEEDED, TaskState.FAILED}:
                return await completed_result(request, task)
            remaining = deadline - monotonic()
            if remaining <= 0:
                pending = accepted(task, response)
                # A Response bypasses the 200 response-model validation on the 202 path.
                return JSONResponse(
                    status_code=202,
                    content=pending.model_dump(mode="json"),
                    headers=dict(response.headers),
                )
            await asyncio.sleep(min(settings.poll_interval_seconds, remaining))

    @app.get("/v1/tasks/{task_id}", response_model=TaskStatus, tags=["tasks"])
    async def task_status(task_id: UUID, request: Request):
        return await get_status(request, task_id)

    @app.get(
        "/v1/tasks/{task_id}/result",
        response_model=EmbeddingResult,
        response_model_exclude_none=True,
        responses={202: {"model": TaskAccepted}},
        tags=["tasks"],
    )
    async def task_result(task_id: UUID, request: Request, response: Response):
        task = await get_status(request, task_id)
        if task.status in {TaskState.SUCCEEDED, TaskState.FAILED}:
            return await completed_result(request, task)
        pending = accepted(task, response)
        return JSONResponse(
            status_code=202,
            content=pending.model_dump(mode="json"),
            headers=dict(response.headers),
        )

    return app


app = create_app()
