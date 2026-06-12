import time

import pytest

from remembrane import MemoryStore


@pytest.fixture()
def store():
    s = MemoryStore(":memory:")
    yield s
    s.close()


def test_log_records_operations(store):
    mem = store.store("first fact")
    store.reinforce(mem.id)
    store.forget(mem.id)
    ops = [e.op for e in store.log()]
    assert ops == ["forget", "reinforce", "store"]  # newest first


def test_snapshot_and_as_of(store):
    store.store("known before snapshot")
    store.snapshot("before")
    time.sleep(0.01)
    store.store("learned after snapshot")
    past = store.as_of("before")
    assert len(past) == 1
    assert past[0]["content"] == "known before snapshot"


def test_diff_added_and_removed(store):
    early = store.store("will be forgotten")
    store.snapshot("start")
    time.sleep(0.01)
    store.store("newly learned")
    store.forget(early.id)
    d = store.diff("start")
    assert [x["content"] for x in d["added"]] == ["newly learned"]
    assert [x["content"] for x in d["removed"]] == ["will be forgotten"]


def test_diff_importance_change(store):
    mem = store.store("growing in importance", importance=0.5)
    store.snapshot("a")
    time.sleep(0.01)
    store.reinforce(mem.id)
    d = store.diff("a")
    assert len(d["changed"]) == 1
    assert d["changed"][0]["importance_before"] == 0.5
    assert d["changed"][0]["importance_after"] == pytest.approx(0.6)


def test_diff_between_two_snapshots(store):
    store.snapshot("empty")
    time.sleep(0.01)
    store.store("middle fact")
    store.snapshot("middle")
    time.sleep(0.01)
    store.store("late fact")
    d = store.diff("empty", "middle")
    assert [x["content"] for x in d["added"]] == ["middle fact"]


def test_diff_is_directional(store):
    store.snapshot("a")
    time.sleep(0.01)
    store.store("x")
    store.snapshot("b")
    forward = store.diff("a", "b")
    backward = store.diff("b", "a")
    assert [m["content"] for m in forward["added"]] == ["x"]
    assert forward["removed"] == []
    assert [m["content"] for m in backward["removed"]] == ["x"]
    assert backward["added"] == []


def test_unknown_snapshot_raises(store):
    with pytest.raises(KeyError):
        store.as_of("never-made")


def test_consolidate_reflected_in_history(store):
    store.store("the user prefers dark mode")
    store.store("the user prefers dark mode")
    store.snapshot("before-cleanup")
    time.sleep(0.01)
    store.consolidate()
    d = store.diff("before-cleanup")
    assert len(d["removed"]) == 1


def test_history_survives_reopen(tmp_path):
    db = tmp_path / "h.db"
    with MemoryStore(db) as s:
        s.store("persistent history")
        s.snapshot("v1")
    with MemoryStore(db) as s2:
        assert [e.op for e in s2.log()] == ["store"]
        assert len(s2.as_of("v1")) == 1
