from uuid import uuid4

from common.schemas import EmbeddingResult, TaskError


def test_queued_and_running_tasks_have_no_ttl(store, redis):
    task_id = uuid4()
    store.create(task_id)
    assert redis.ttl(store.status_key(task_id)) == -1
    store.start(task_id)
    assert redis.ttl(store.status_key(task_id)) == -1


def test_completed_result_and_status_expire_together(store, redis, settings):
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
    assert 86399 <= redis.ttl(store.status_key(task_id)) <= 86400
    assert redis.ttl(store.result_key(task_id)) == redis.ttl(store.status_key(task_id))
    redis.delete(store.status_key(task_id), store.result_key(task_id))
    assert store.get(task_id) is None
    assert store.get_result(task_id) is None


def test_failed_task_expires_and_does_not_have_result(store, redis):
    task_id = uuid4()
    store.create(task_id)
    final = store.finish(task_id, error=TaskError(code="failed", message="failed"))
    assert final.status == "failed"
    assert redis.ttl(store.status_key(task_id)) > 0
    assert store.get_result(task_id) is None
