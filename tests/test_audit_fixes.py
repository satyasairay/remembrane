import pytest

from remembrane import Memory, MemoryStore


@pytest.fixture()
def store():
    s = MemoryStore(":memory:")
    yield s
    s.close()


def test_zero_relevance_memories_never_returned(store):
    store.store("alpha beta")
    store.store("gamma delta")
    assert store.recall("", touch=False) == []
    assert store.recall("completely unrelated zzz qqq xxx", touch=False, mode="keyword") == []


def test_relevance_required_despite_importance(store):
    store.store("maximum importance noise", importance=1.0)
    assert store.recall("", touch=False) == []


def test_recall_rejects_non_string_query(store):
    with pytest.raises(ValueError, match="query must be a string"):
        store.recall(None)
    with pytest.raises(ValueError, match="query must be a string"):
        store.recall(42)


def test_store_creates_parent_directories(tmp_path):
    db = tmp_path / "deeply" / "nested" / "dirs" / "mem.db"
    s = MemoryStore(db)
    s.store("created the path for you")
    assert db.exists()
    s.close()


def test_metadata_must_be_dict():
    with pytest.raises(ValueError, match="metadata must be a dict"):
        Memory(content="x", metadata=["not", "a", "dict"])


def test_store_metadata_must_be_dict(store):
    with pytest.raises(ValueError, match="metadata must be a dict"):
        store.store("x", metadata="nope")


def test_cli_concise_error_on_bad_input(tmp_path, capsys):
    from remembrane.cli import _main_wrapper
    rc = _main_wrapper(["--db", str(tmp_path / "x.db"), "store", ""])
    assert rc == 1
    err = capsys.readouterr().err
    assert err.startswith("error:")
    assert "Traceback" not in err


def test_cli_unknown_snapshot_concise(tmp_path, capsys):
    from remembrane.cli import _main_wrapper
    db = str(tmp_path / "x.db")
    _main_wrapper(["--db", db, "store", "something"])
    capsys.readouterr()
    rc = _main_wrapper(["--db", db, "diff", "no-such-snapshot"])
    assert rc == 1
    assert "error:" in capsys.readouterr().err
