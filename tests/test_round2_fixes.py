"""Regression tests for the round-2 audit findings."""
import time

import pytest

from remembrane import MemoryStore
from remembrane.conflicts import knapsack_pack


def test_pack_budget_never_exceeded_codex_repro():
    """Round 2 High: two 1501-token memories, budget 3000 -> must not return both."""
    m = MemoryStore(":memory:")
    m.store("a " * 3000)
    m.store("b " * 3000)
    chosen = m.pack("a b", budget_tokens=3000, touch=False, dedupe_threshold=1.1)
    assert sum(r.tokens for r in chosen) <= 3000
    m.close()


def test_knapsack_ceil_weights_property():
    import random
    rng = random.Random(42)
    for _ in range(50):
        n = rng.randint(1, 12)
        scores = [rng.random() for _ in range(n)]
        tokens = [rng.randint(1, 4000) for _ in range(n)]
        budget = rng.randint(1, 6000)
        sel, total = knapsack_pack(scores, tokens, budget)
        assert total <= budget, (scores, tokens, budget, sel)


def test_diff_directionality():
    """Round 2 Medium: diff(b, a) must invert diff(a, b)."""
    m = MemoryStore(":memory:")
    m.snapshot("a")
    time.sleep(0.01)
    m.store("added later")
    m.snapshot("b")
    fwd, back = m.diff("a", "b"), m.diff("b", "a")
    assert [x["content"] for x in fwd["added"]] == ["added later"]
    assert [x["content"] for x in back["removed"]] == ["added later"]
    assert fwd["removed"] == [] and back["added"] == []
    m.close()


def test_corrupt_journal_payload_survives(tmp_path):
    """Round 2 Medium: malformed journal JSON must not crash log()/as_of()."""
    m = MemoryStore(tmp_path / "j.db")
    m.store("good")
    m._conn.execute("UPDATE journal SET payload='{{{not json' WHERE seq=1")
    m._conn.commit()
    entries = m.log()
    assert entries and "_corrupt" in entries[0].payload
    state = m.as_of(time.time())  # corrupt store-op is skipped, no crash
    assert isinstance(state, list)
    m.close()


def test_conflicts_min_confidence_filters_weak_markers():
    """Round 2 Medium: benign change-words must not reach 'likely'."""
    m = MemoryStore(":memory:")
    m.store("the new hire moved the meeting to the main room")
    m.store("the new hire booked the main room")
    assert m.conflicts(min_confidence="likely") == []
    m2 = MemoryStore(":memory:")
    m2.store("version 1 supports export")
    m2.store("version 2 supports import")
    assert m2.conflicts(min_confidence="likely") == []
    m.close()
    m2.close()


def test_conflicts_true_positives_still_likely():
    pairs = [
        ("the user lives in London", "the user moved to Tokyo, no longer in London"),
        ("the deadline is on day 12", "the deadline is on day 26"),
        ("the meeting is at 3pm", "the meeting was moved to 5pm"),
    ]
    for a, b in pairs:
        m = MemoryStore(":memory:")
        m.store(a)
        m.store(b)
        assert m.conflicts(min_confidence="likely"), (a, b)
        m.close()


def test_conflicts_invalid_min_confidence():
    m = MemoryStore(":memory:")
    with pytest.raises(ValueError):
        m.conflicts(min_confidence="certain")
    m.close()


def test_cli_store_from_file(tmp_path, capsys):
    from remembrane.cli import _main_wrapper
    big = tmp_path / "big.txt"
    big.write_text("huge content " * 1000)
    rc = _main_wrapper(["--db", str(tmp_path / "f.db"), "store", "--file", str(big)])
    assert rc == 0
    assert "stored" in capsys.readouterr().out


def test_crewai_adapter_current_protocol_surface():
    """Round 2 High: search(scope_prefix=...), delete, update, list_records."""
    from remembrane.adapters import RemembraneStorage
    m = MemoryStore(":memory:")
    st = RemembraneStorage(m)
    rid = st.save("the deadline is friday", metadata={"task": "x"}, scope_prefix="crew/1")
    hits = st.search("deadline", scope_prefix="crew/1")  # audit's exact failing kwarg
    assert hits and hits[0]["content"] == "the deadline is friday"
    assert st.update(rid, value="the deadline is monday")
    assert st.search("deadline")[0]["content"] == "the deadline is monday"
    assert len(st.list_records()) == 1
    assert st.delete(rid) is True
    assert st.list_records() == []
    m.close()
