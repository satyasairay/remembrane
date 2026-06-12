
from remembrane import MemoryStore


def test_merge_adds_new_memories(tmp_path):
    a = MemoryStore(tmp_path / "a.db")
    b = MemoryStore(tmp_path / "b.db")
    a.store("fact known to A")
    b.store("fact known to B")
    result = a.merge_from(b)
    assert result == {"added": 1, "merged": 0}
    assert a.count() == 2
    a.close()
    b.close()


def test_merge_dedupes_near_duplicates(tmp_path):
    a = MemoryStore(tmp_path / "a.db")
    b = MemoryStore(tmp_path / "b.db")
    a.store("the user prefers dark mode", importance=0.5)
    b.store("the user prefers dark mode", importance=0.9)
    result = a.merge_from(b)
    assert result == {"added": 0, "merged": 1}
    assert a.count() == 1
    assert a.all()[0].importance == 0.9  # max importance wins
    a.close()
    b.close()


def test_merge_from_path(tmp_path):
    b = MemoryStore(tmp_path / "b.db")
    b.store("portable fact")
    b.close()
    a = MemoryStore(tmp_path / "a.db")
    result = a.merge_from(tmp_path / "b.db")
    assert result["added"] == 1
    a.close()


def test_merge_namespace_filter(tmp_path):
    a = MemoryStore(tmp_path / "a.db")
    b = MemoryStore(tmp_path / "b.db")
    b.store("wanted", namespace="keep")
    b.store("unwanted", namespace="skip")
    a.merge_from(b, namespaces=["keep"])
    assert a.count() == 1
    assert a.all()[0].namespace == "keep"
    a.close()
    b.close()


def test_merge_disable_dedupe(tmp_path):
    a = MemoryStore(tmp_path / "a.db")
    b = MemoryStore(tmp_path / "b.db")
    a.store("identical content")
    b.store("identical content")
    result = a.merge_from(b, dedupe_threshold=1.1)
    assert result == {"added": 1, "merged": 0}
    assert a.count() == 2
    a.close()
    b.close()


def test_merged_memories_recallable(tmp_path):
    a = MemoryStore(tmp_path / "a.db")
    b = MemoryStore(tmp_path / "b.db")
    b.store("the wifi password is hunter2")
    a.merge_from(b)
    results = a.recall("what is the wifi password?")
    assert "hunter2" in results[0].memory.content
    a.close()
    b.close()


def test_merge_appears_in_journal(tmp_path):
    a = MemoryStore(tmp_path / "a.db")
    b = MemoryStore(tmp_path / "b.db")
    b.store("journaled merge")
    a.merge_from(b)
    assert any(e.op == "merge" for e in a.log())
    a.close()
    b.close()
