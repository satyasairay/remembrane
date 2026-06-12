import pytest

from remembrane import MemoryStore
from remembrane.scoring import usefulness_factor


@pytest.fixture()
def store():
    s = MemoryStore(":memory:")
    yield s
    s.close()


def test_mark_useful_accumulates(store):
    mem = store.store("helpful fact")
    store.mark_useful(mem.id)
    store.mark_useful(mem.id)
    assert store.get(mem.id).usefulness == 2.0


def test_mark_useless_decreases(store):
    mem = store.store("noise")
    store.mark_useless(mem.id, weight=2.0)
    assert store.get(mem.id).usefulness == -2.0


def test_feedback_changes_ranking(store):
    a = store.store("the API endpoint is /v2/users")
    b = store.store("the API endpoint is /v2/users  ")  # near-identical content
    for _ in range(5):
        store.mark_useful(a.id)
        store.mark_useless(b.id)
    results = store.recall("API endpoint", k=2, touch=False)
    assert results[0].memory.id == a.id


def test_neutral_factor_is_half():
    assert usefulness_factor(0.0) == 0.5
    assert usefulness_factor(10.0) > 0.99
    assert usefulness_factor(-10.0) < 0.01


def test_feedback_journaled(store):
    mem = store.store("tracked")
    store.mark_useful(mem.id)
    entry = next(e for e in store.log() if e.op == "feedback")
    assert entry.payload["useful"] is True


def test_feedback_in_explain(store):
    mem = store.store("explained")
    store.mark_useful(mem.id, weight=3.0)
    r = store.recall("explained", k=1, touch=False)[0]
    assert r.explain()["components"]["usefulness"] == 3.0


def test_invalid_weight(store):
    mem = store.store("x")
    with pytest.raises(ValueError):
        store.feedback(mem.id, True, weight=0)


def test_migration_from_old_db(tmp_path):
    import sqlite3
    db = tmp_path / "old.db"
    conn = sqlite3.connect(db)
    conn.executescript("""
        CREATE TABLE memories (
            id TEXT PRIMARY KEY, namespace TEXT NOT NULL, content TEXT NOT NULL,
            importance REAL NOT NULL, metadata TEXT, created_at REAL NOT NULL,
            last_accessed_at REAL, access_count INTEGER NOT NULL DEFAULT 0,
            embedding BLOB NOT NULL);
    """)
    conn.commit()
    conn.close()
    s = MemoryStore(db)  # should ALTER TABLE without error
    s.store("works after migration")
    assert s.recall("works", touch=False)
    s.close()
