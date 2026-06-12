"""Regressions from the round-4 independent verification audit (v0.5.0).

The auditor's exact probe sentences live in their harness; the pairs below
reconstruct the named findings from the report. Unfixed behaviors are
asserted AS DOCUMENTED LIMITATIONS, not skipped, so any silent change in
behavior fails loudly.
"""
import warnings

import pytest

from remembrane import MemoryStore


@pytest.fixture()
def store():
    s = MemoryStore(":memory:")
    yield s
    s.close()


# --- Medium: likely-tier false negatives (value substitution) ---------------

def test_technology_swap_reaches_likely_tier(store):
    """Round-4 FN: Python -> Rust. Same template, one slot swapped."""
    store.store("the backend service is written in Python")
    store.store("the backend service is written in Rust")
    found = store.conflicts(min_confidence="likely")
    assert found, "technology value swap must reach the likely tier"
    assert any("substitution" in sig for sig in found[0].signals)


def test_region_swap_reaches_likely_tier(store):
    """Round-4 FN, the auditor's exact phrasing — not a longer paraphrase."""
    store.store("the deployment is in us-east-1")
    store.store("the deployment is in eu-west-1")
    found = store.conflicts(min_confidence="likely")
    assert found, "region value swap must reach the likely tier"
    assert any("substitution" in sig for sig in found[0].signals)


def test_substitution_requires_template_match(store):
    """Round-2 guard must hold: rephrasings are not slot swaps."""
    store.store("the new hire moved the meeting to the main room")
    store.store("the new hire booked the main room")
    assert store.conflicts(min_confidence="likely") == []


def test_subject_swap_not_upgraded_to_likely(store):
    """'alice drinks coffee' vs 'bob drinks coffee' = two subjects, both true.
    The leading-diff-word guard keeps subject swaps out of the likely tier."""
    store.store("alice drinks coffee daily")
    store.store("bob drinks coffee daily")
    assert store.conflicts(min_confidence="likely") == []


def test_known_limitation_old_new_entity_false_positive(store):
    """Round-4 benign FP, DOCUMENTED as a limitation: 'old laptop 16GB' vs
    'new laptop 32GB' describes two coexisting entities but flags likely.
    Suppressing it would also suppress real old->new contradictions, so we
    keep it and document it. If this stops flagging, the docs must change."""
    store.store("the old laptop had 16GB of RAM")
    store.store("the new laptop has 32GB of RAM")
    found = store.conflicts(min_confidence="likely")
    assert found, "documented FP behavior changed -- update README and this test"


# --- Low: corrupt embedding blobs must signal, not just self-heal -----------

def test_corrupt_blob_self_heals_with_warning(tmp_path):
    db = tmp_path / "m.db"
    s = MemoryStore(str(db))
    m = s.store("the user prefers dark mode")
    s._conn.execute("UPDATE memories SET embedding=? WHERE id=?", (b"\x00", m.id))
    s._conn.commit()
    s._corpus.clear()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        results = s.recall("dark mode")
    assert results and results[0].memory.id == m.id  # still available
    runtime = [w for w in caught if issubclass(w.category, RuntimeWarning)]
    assert runtime, "self-heal must emit a RuntimeWarning"
    assert "re-embedded" in str(runtime[0].message)
    assert m.id in str(runtime[0].message)
    s.close()


def test_healthy_store_emits_no_heal_warning(store):
    store.store("nothing wrong here")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        store.recall("anything")
    assert not [w for w in caught if "re-embedded" in str(w.message)]


def test_language_swap_audit_phrasing(store):
    """Round-4 FN, auditor's exact phrasing."""
    store.store("the service is written in Python")
    store.store("the service is written in Rust")
    assert store.conflicts(min_confidence="likely")
