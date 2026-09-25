from kokoro_ds.worker import WorkItem, shard_work

from .conftest import make_record


def items(n: int) -> list[WorkItem]:
    return [WorkItem("train", make_record(f"id_{i:04d}")) for i in range(n)]


def test_shard_work_distributes_all_items_exactly_once():
    work = items(23)
    shards = shard_work(work, n_workers=4)
    flat_ids = sorted(w.record.id for shard in shards for w in shard)
    assert flat_ids == sorted(w.record.id for w in work)


def test_shard_work_is_balanced_within_one():
    work = items(23)
    shards = shard_work(work, n_workers=4)
    sizes = [len(s) for s in shards]
    assert max(sizes) - min(sizes) <= 1
    assert sum(sizes) == 23


def test_shard_work_handles_fewer_items_than_workers():
    work = items(2)
    shards = shard_work(work, n_workers=8)
    non_empty = [s for s in shards if s]
    assert sum(len(s) for s in non_empty) == 2
    assert all(len(s) == 1 for s in non_empty)


def test_shard_work_single_worker_keeps_original_order():
    work = items(10)
    shards = shard_work(work, n_workers=1)
    assert len(shards) == 1
    assert [w.record.id for w in shards[0]] == [w.record.id for w in work]


def test_shard_work_zero_items():
    assert shard_work([], n_workers=4) == [[], [], [], []]
