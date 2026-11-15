import multiprocessing as mp
import time
from queue import Empty, Full
from threading import Event, RLock, Thread
from uuid import uuid4

from pydantic import ValidationError

from common.config import Settings
from common.jobs import JobStore
from common.schemas import EmbeddingRequest, EmbeddingResult, TaskError, TaskStatus
from gpu_worker.process import run_worker


class QueueUnavailable(RuntimeError):
    pass


class TaskService:
    def __init__(self, settings: Settings, *, worker_target=run_worker):
        self.settings = settings
        self.store = JobStore(settings.result_ttl_seconds)
        self._context = mp.get_context("spawn")
        self._requests = self._context.Queue(maxsize=settings.queue_capacity)
        self._results = self._context.Queue(maxsize=2)
        self._worker_stop = self._context.Event()
        self._process = self._context.Process(
            target=worker_target,
            args=(settings, self._requests, self._results, self._worker_stop),
            name="gpu-worker",
            daemon=True,
        )
        self._ready = Event()
        self._collector_stop = Event()
        self._lock = RLock()
        self._accepting = False
        self._closed = False
        self._startup_error = None
        self._collector = Thread(target=self._collect, name="gpu-results", daemon=True)

    @property
    def ready(self) -> bool:
        return (
            self._accepting
            and self._ready.is_set()
            and self._process.is_alive()
            and self._collector.is_alive()
        )

    def start(self) -> None:
        try:
            self._process.start()
            self._collector.start()
            deadline = time.monotonic() + self.settings.worker_startup_timeout_seconds
            while not self._ready.wait(0.1):
                if self._startup_error or not self._process.is_alive():
                    raise RuntimeError(self._startup_error or "GPU worker exited during startup")
                if time.monotonic() >= deadline:
                    raise TimeoutError("GPU worker startup timed out")
            with self._lock:
                self._accepting = True
        except BaseException:
            self.close()
            raise

    def submit(self, request: EmbeddingRequest) -> TaskStatus:
        with self._lock:
            if not self.ready:
                raise QueueUnavailable("GPU worker is unavailable")
            task_id = uuid4()
            task = self.store.create(task_id)
            try:
                self._requests.put_nowait((str(task_id), request.model_dump(mode="json")))
            except (Full, OSError, ValueError) as exc:
                self.store.discard(task_id)
                raise QueueUnavailable("Inference queue is full or unavailable") from exc
            return task

    def status(self, task_id):
        # Also detect exit here: a killed process may leave a partial Queue frame
        # and block the collector thread inside recv(), despite get(timeout=...).
        if self._process.exitcode is not None:
            self.store.fail_pending("worker_exited", "GPU worker exited; restart service")
        elif self._process.pid is not None and not self._collector.is_alive() and not self._closed:
            self.store.fail_pending("worker_exited", "GPU result collector exited; restart service")
        return self.store.get(task_id)

    def _collect(self):
        next_prune = time.monotonic()
        while not self._collector_stop.is_set():
            try:
                kind, task_id, payload = self._results.get(timeout=0.1)
            except (EOFError, OSError, ValueError):
                with self._lock:
                    self._accepting = False
                    self._ready.clear()
                    self.store.fail_pending("worker_exited", "GPU result channel closed")
                return
            except Empty:
                if not self._process.is_alive():
                    with self._lock:
                        self._accepting = False
                        self._ready.clear()
                        self.store.fail_pending(
                            "worker_exited", "GPU worker exited; restart service"
                        )
                    return
            else:
                with self._lock:
                    if kind == "ready":
                        self._ready.set()
                    elif kind == "startup_failed":
                        self._startup_error = f"GPU worker initialization failed ({payload})"
                    elif kind == "running":
                        self.store.start(task_id)
                    elif kind == "succeeded":
                        try:
                            result = EmbeddingResult.model_validate_json(payload)
                        except ValidationError:
                            self.store.finish(
                                task_id,
                                error=TaskError(
                                    code="invalid_result",
                                    message="GPU worker returned an invalid result",
                                ),
                            )
                        else:
                            self.store.finish(task_id, result=result)
                    elif kind == "failed":
                        self.store.finish(
                            task_id,
                            error=TaskError(
                                code="inference_failed",
                                message=f"Inference failed ({payload}); see worker logs",
                            ),
                        )
            if time.monotonic() >= next_prune:
                self.store.prune()
                next_prune = time.monotonic() + 1

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._accepting = False
        if self._process.pid is not None:
            # Drain results while joining so large Queue payloads cannot deadlock exit.
            try:
                self._requests.put_nowait(None)
            except (Full, OSError, ValueError):
                self._worker_stop.set()
            self._process.join(timeout=self.settings.worker_shutdown_timeout_seconds)
            if self._process.is_alive():
                self._process.terminate()
                self._process.join(timeout=5)
            if self._process.is_alive():
                self._process.kill()
                self._process.join(timeout=5)
        self._collector_stop.set()
        if self._collector.ident is not None:
            self._collector.join(timeout=2)
        self._ready.clear()
        self.store.fail_pending("service_stopped", "Service stopped before task completion")
        for queue in (self._requests, self._results):
            queue.cancel_join_thread()
            queue.close()
