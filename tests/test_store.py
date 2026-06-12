import time

import pytest

from remembrane import Memory, MemoryStore, ScoringConfig


@pytest.fixture()
def store():
    s = MemoryStore(":memory:")
    yield s
    s.close()


def test_store_and_get(store):
    mem = store.store("the user lives in Bhubaneswar", importance=0.8)
    fetched = store.get(mem.id)
    assert fetched is not None
    assert fetched.content == "the user lives in Bhubaneswar"
    assert fetched.importance == 0.8


def test_recall_ranks_relevant_first(store):
    store.store("the user prefers dark mode in the editor")
    store.store("the deployment runs on AWS us-east-1")
    store.store("the user's cat is named Pixel")
    results = store.recall("which theme does the user like?", k=2)
    assert results
    assert "dark mode" in results[0].memory.content


def test_recall_respects_namespace(store):
    store.store("alpha fact", namespace="a")
    store.store("beta fact", namespace="b")
    results = store.recall("fact", namespace="a", k=10)
    assert all(r.memory.namespace == "a" for r in results)
    results_all = store.recall("fact", namespace=None, k=10)
    assert {r.memory.namespace for r in results_all} == {"a", "b"}


def test_recall_touch_updates_access(store):
    mem = store.store("touch me")
    store.recall("touch me", k=1)
    fetched = store.get(mem.id)
    assert fetched.access_count == 1
    assert fetched.last_accessed_at is not None


def test_recall_no_touch(store):
    mem = store.store("do not touch")
    store.recall("do not touch", k=1, touch=False)
    assert store.get(mem.id).access_count == 0


def test_forget_by_id(store):
    mem = store.store("ephemeral")
    assert store.forget(mem.id) == 1
    assert store.get(mem.id) is None


def test_forget_namespace(store):
    store.store("x", namespace="tmp")
    store.store("y", namespace="tmp")
    store.store("z", namespace="keep")
    assert store.forget(namespace="tmp") == 2
    assert store.count() == 1


def test_forget_older_than(store):
    old = store.store("ancient wisdom")
    store._conn.execute(
        "UPDATE memories SET created_at=? WHERE id=?", (time.time() - 10_000, old.id)
    )
    store._conn.commit()
    store.store("fresh news")
    assert store.forget(older_than_seconds=5_000) == 1
    assert store.count() == 1


def test_forget_requires_criteria(store):
    with pytest.raises(ValueError):
        store.forget()


def test_reinforce(store):
    mem = store.store("important fact", importance=0.5)
    updated = store.reinforce(mem.id)
    assert updated.importance == pytest.approx(0.6)
    assert updated.access_count == 1


def test_consolidate_merges_duplicates(store):
    store.store("the user prefers dark mode", metadata={"src": "first"})
    store.store("the user prefers dark mode", metadata={"src": "second"})
    store.store("completely unrelated topic about quantum physics")
    removed = store.consolidate()
    assert removed == 1
    assert store.count() == 2


def test_store_many(store):
    mems = store.store_many(["one", "two", "three"], namespace="batch")
    assert len(mems) == 3
    assert store.count("batch") == 3


def test_export(store):
    store.store("exportable", metadata={"k": "v"})
    data = store.export()
    assert len(data) == 1
    assert data[0]["content"] == "exportable"
    assert data[0]["metadata"] == {"k": "v"}


def test_persistence(tmp_path):
    db = tmp_path / "mem.db"
    with MemoryStore(db) as s:
        s.store("persisted across connections")
    with MemoryStore(db) as s2:
        results = s2.recall("persisted")
        assert results
        assert results[0].memory.content == "persisted across connections"


def test_invalid_memory():
    with pytest.raises(ValueError):
        Memory(content="   ")
    with pytest.raises(ValueError):
        Memory(content="ok", importance=1.5)


def test_scoring_config_validation():
    with pytest.raises(ValueError):
        ScoringConfig(weight_similarity=0, weight_recency=0, weight_importance=0)
    with pytest.raises(ValueError):
        ScoringConfig(half_life_seconds=-1)
