from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest

from common.schemas import EmbeddingResult, TaskError


def test_completed_result_and_status_expire_together(store):
    task_id = uuid4()
    store.create(task_id)
    store.start(task_id)
    final = store.finish(
        task_id,
        result=EmbeddingResult(
            task_id=task_id, model="BAAI/bge-m3", embedding_types=["dense"], data=[]
        ),
    )
    assert final.status == "succeeded"
    assert (final.expires_at - final.finished_at).total_seconds() == 86400
    assert store.get_result(task_id).task_id == task_id
    store._statuses[str(task_id)].expires_at = datetime.now(UTC) - timedelta(seconds=1)
    assert store.get_result(task_id) is None
    assert store.get(task_id) is None


def test_pending_tasks_do_not_expire_and_copies_cannot_mutate_store(store):
    task_id = uuid4()
    task = store.create(task_id)
    task.status = "failed"
    assert store.get(task_id).status == "queued"
    store.start(task_id)
    store.prune()
    assert store.get(task_id).status == "running"
    assert store.get(task_id).expires_at is None


def test_failure_and_duplicate_completion(store):
    task_id = uuid4()
    store.create(task_id)
    final = store.finish(task_id, error=TaskError(code="failed", message="failed"))
    assert store.get_result(task_id) is None
    assert store.finish(task_id, error=TaskError(code="again", message="again")) == final
    with pytest.raises(ValueError):
        store.finish(task_id)


def test_fail_pending_preserves_completed_results(store):
    ids = [uuid4() for _ in range(3)]
    for task_id in ids:
        store.create(task_id)
    store.start(ids[1])
    store.finish(
        ids[2],
        result=EmbeddingResult(
            task_id=ids[2], model="BAAI/bge-m3", embedding_types=["dense"], data=[]
        ),
    )
    store.fail_pending("worker_exited", "worker exited")
    assert [store.get(task_id).status for task_id in ids] == ["failed", "failed", "succeeded"]
    assert store.get_result(ids[2]) is not None


def test_prune_reclaims_expired_results(store):
    task_id = uuid4()
    store.create(task_id)
    store.finish(
        task_id,
        result=EmbeddingResult(
            task_id=task_id, model="BAAI/bge-m3", embedding_types=["dense"], data=[]
        ),
    )
    store._statuses[str(task_id)].expires_at = datetime.now(UTC) - timedelta(seconds=1)
    store.prune()
    assert not store._statuses
    assert not store._results
