import logging
from queue import Empty

from common.schemas import EmbeddingRequest

logger = logging.getLogger(__name__)


def run_worker(settings, requests, results, stop, engine_factory=None):
    """Spawn entry point: only this process imports torch and owns the model."""
    logging.basicConfig(level=logging.INFO)
    try:
        if engine_factory is None:
            from gpu_worker.inference import InferenceEngine

            engine_factory = InferenceEngine
        engine = engine_factory(settings)
    except Exception as exc:
        logger.exception("Model initialization failed")
        results.put(("startup_failed", None, type(exc).__name__))
        return
    results.put(("ready", None, None))
    while not stop.is_set():
        try:
            message = requests.get(timeout=0.1)
        except Empty:
            continue
        if message is None:
            return
        task_id, payload = message
        results.put(("running", task_id, None))
        try:
            request = EmbeddingRequest.model_validate(payload)
            result = engine.encode(task_id, request)
            # Serialize before put: feeder-thread errors must not lose results silently.
            encoded = result.model_dump_json(exclude_none=True)
        except Exception as exc:
            logger.exception("Inference failed for %s", task_id)
            results.put(("failed", task_id, type(exc).__name__))
        else:
            results.put(("succeeded", task_id, encoded))
