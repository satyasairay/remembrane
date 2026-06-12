"""Regression tests for the round-3 ("mother of all audits") findings."""
import threading

import pytest

from remembrane import MemoryStore


def test_cross_connection_cache_coherence(tmp_path):
    """Round 3 High: second connection's write must not break a warm cache."""
    db = tmp_path / "x.db"
    a = MemoryStore(db)
    b = MemoryStore(db)
    a.store("alpha", memory_id="a")
    assert a.recall("alpha", namespace=None, touch=False)
    b.store("beta", memory_id="b")  # external write while a's cache is warm
    res = a.recall("beta", namespace=None, touch=False)  # KeyError in 0.4.0
    assert res and res[0].memory.id == "b"
    a.close()
    b.close()


def test_concurrent_multi_connection_hammer(tmp_path):
    db = tmp_path / "h.db"
    stores = [MemoryStore(db) for _ in range(3)]
    errors = []

    def worker(store, tag):
        try:
            for i in range(50):
                store.store(f"{tag} fact {i}")
                store.recall(f"{tag} fact", touch=(i % 3 == 0))
        except Exception as exc:  # pragma: no cover
            errors.append(repr(exc))

    threads = [threading.Thread(target=worker, args=(s, f"t{i}"))
               for i, s in enumerate(stores)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    assert stores[0].count() == 150
    for s in stores:
        s.close()


def test_wal_mode_default_and_opt_out(tmp_path):
    a = MemoryStore(tmp_path / "w.db")
    assert a._conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    a.close()
    b = MemoryStore(tmp_path / "d.db", journal_mode="DELETE")
    assert b._conn.execute("PRAGMA journal_mode").fetchone()[0] == "delete"
    b.close()


def test_corrupt_embedding_blob_self_heals(tmp_path):
    """Round 3 Medium: wrong-length blob crashed recall in 0.4.0."""
    m = MemoryStore(tmp_path / "c.db")
    mem = m.store("healable fact")
    m._conn.execute("UPDATE memories SET embedding=X'00' WHERE id=?", (mem.id,))
    m._conn.commit()
    m._invalidate()
    res = m.recall("healable fact", touch=False)
    assert res and res[0].memory.id == mem.id
    m.close()


def test_broken_numpy_does_not_break_import(tmp_path):
    """Round 3 Medium: a present-but-broken numpy must not crash imports."""
    import os
    import subprocess
    import sys
    bad = tmp_path / "numpy.py"
    bad.write_text("raise RuntimeError('broken numpy install')\n")
    import remembrane
    pkg_root = str(__import__("pathlib").Path(remembrane.__file__).parent.parent)
    env = dict(os.environ)  # keep SystemRoot etc. — stripping env breaks Windows Python
    env["PYTHONPATH"] = os.pathsep.join([str(tmp_path), pkg_root])
    script = (
        "import sys\n"
        "try:\n"
        "    import numpy\n"
        "    sys.exit('fake numpy was not picked up')\n"
        "except RuntimeError:\n"
        "    pass\n"
        "import remembrane\n"
        "print(remembrane.__version__)\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True, env=env,
    )
    assert proc.returncode == 0, proc.stderr


def test_pack_exact_with_numpy():
    import itertools
    import random

    import remembrane.conflicts as c
    if c._np is None:
        pytest.skip("numpy not installed")
    rng = random.Random(4004001)
    for _ in range(100):
        n = rng.randint(1, 12)
        scores = [rng.random() for _ in range(n)]
        tokens = [rng.randint(1, 5000) for _ in range(n)]
        budget = rng.randint(1, 8000)
        sel, total = c.knapsack_pack(scores, tokens, budget)
        assert total <= budget
        got = sum(scores[i] for i in sel)
        best = max(
            (sum(scores[i] for i in combo)
             for r in range(n + 1)
             for combo in itertools.combinations(range(n), r)
             if sum(tokens[i] for i in combo) <= budget),
            default=0.0,
        )
        assert abs(got - best) < 1e-9


def test_namespace_must_be_string():
    m = MemoryStore(":memory:")
    with pytest.raises(ValueError, match="namespace"):
        m.store("x", namespace=None)
    m.close()


def test_nan_metadata_rejected():
    m = MemoryStore(":memory:")
    with pytest.raises(ValueError, match="JSON-serializable"):
        m.store("x", metadata={"v": float("nan")})
    import datetime
    with pytest.raises(ValueError, match="JSON-serializable"):
        m.store("x", metadata={"v": datetime.datetime.now()})
    m.close()


def test_weekday_value_mismatch_is_likely():
    """Round 3 Medium: audit's false negative."""
    m = MemoryStore(":memory:")
    m.store("the deadline is Friday")
    m.store("the deadline is now Monday")
    assert m.conflicts(min_confidence="likely")
    m.close()


def test_unrelated_value_words_not_flagged():
    m = MemoryStore(":memory:")
    m.store("the team meets on Monday")
    m.store("the report is due on Friday")
    assert m.conflicts(min_confidence="likely") == []
    m.close()


def test_empty_existing_file_gets_schema(tmp_path):
    """Round 3 Medium: a kill mid-creation leaves an empty file; reopening must repair."""
    db = tmp_path / "empty.db"
    db.touch()
    m = MemoryStore(db)
    m.store("schema repaired")
    assert m.recall("schema repaired", touch=False)
    m.close()


def test_langchain_structured_content_blocks():
    pytest.importorskip("langchain_core")
    from langchain_core.messages import HumanMessage
    from remembrane.adapters import RemembraneChatMessageHistory
    m = MemoryStore(":memory:")
    hist = RemembraneChatMessageHistory(m, "s")
    hist.add_message(HumanMessage(content=[{"type": "text", "text": "block content"}]))
    assert hist.messages[0].content == "block content"
    m.close()
