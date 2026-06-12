import pytest

from remembrane import MemoryStore


@pytest.fixture()
def store():
    s = MemoryStore(":memory:")
    yield s
    s.close()


def test_detects_changed_fact(store):
    store.store("the user lives in London")
    store.store("the user moved to Tokyo, no longer in London")
    found = store.conflicts()
    assert found
    assert found[0].confidence == "likely"
    assert any("change_markers" in sig for sig in found[0].signals)


def test_detects_numeric_mismatch(store):
    store.store("the deadline is on day 12 of the month")
    store.store("the deadline is on day 26 of the month")
    found = store.conflicts()
    assert found
    assert found[0].confidence == "likely"
    assert any("numeric_mismatch" in sig for sig in found[0].signals)


def test_unrelated_memories_not_flagged(store):
    store.store("the user prefers dark mode")
    store.store("quarterly revenue grew twelve percent")
    assert store.conflicts() == []


def test_near_duplicates_not_flagged_as_conflicts(store):
    store.store("the user prefers dark mode")
    store.store("the user prefers dark mode")
    # identical pairs are consolidate() targets, not conflicts
    assert store.conflicts() == []


def test_query_scoped_conflicts(store):
    store.store("the user lives in London")
    store.store("the user moved to Tokyo, not London anymore")
    store.store("the build pipeline uses python 3.12")
    found = store.conflicts("where does the user live?")
    assert found
    contents = {found[0].a.content, found[0].b.content}
    assert all("user" in c for c in contents)


def test_describe_orders_old_to_new(store):
    import time
    store.store("the user lives in London")
    time.sleep(0.01)
    store.store("the user moved to Tokyo, no longer London")
    text = store.conflicts()[0].describe()
    assert text.index("older:") < text.index("newer:")
    assert "London" in text and "Tokyo" in text


def test_resolve_keeps_winner_journals_loser(store):
    a = store.store("the user lives in London")
    b = store.store("the user moved to Tokyo, not London")
    dropped = store.resolve(b.id, [a.id], reason="user confirmed Tokyo")
    assert dropped == 1
    assert store.get(a.id) is None
    assert store.get(b.id) is not None
    ops = [e.op for e in store.log()]
    assert "resolve" in ops
    entry = next(e for e in store.log() if e.op == "resolve")
    assert entry.payload["reason"] == "user confirmed Tokyo"


def test_resolve_unknown_keeper_raises(store):
    with pytest.raises(KeyError):
        store.resolve("nope", ["also-nope"])


def test_namespaces_isolated(store):
    store.store("the user lives in London", namespace="a")
    store.store("the user moved to Tokyo, not London", namespace="b")
    assert store.conflicts(namespace="a") == []
