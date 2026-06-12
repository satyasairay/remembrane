import pytest

from remembrane import MemoryStore
from remembrane.bm25 import bm25_scores, normalized_bm25


@pytest.fixture()
def store():
    s = MemoryStore(":memory:")
    s.store("the deployment pipeline failed on tuesday")
    s.store("user prefers tabs over spaces")
    s.store("API key rotation happens monthly")
    yield s
    s.close()


def test_bm25_exact_term_match():
    docs = ["alpha beta gamma", "delta epsilon", "alpha alpha zeta"]
    scores = bm25_scores("alpha", docs)
    assert scores[0] > 0 and scores[2] > 0
    assert scores[1] == 0


def test_bm25_normalized_bounds():
    docs = ["one two three", "two three four"]
    scores = normalized_bm25("two", docs)
    assert max(scores) == 1.0
    assert all(0 <= s <= 1 for s in scores)


def test_bm25_empty():
    assert bm25_scores("query", []) == []
    assert bm25_scores("", ["doc"]) == [0.0]


def test_hybrid_is_default_and_populates_components(store):
    r = store.recall("deployment pipeline", k=1)[0]
    assert "deployment" in r.memory.content
    assert r.vector_score > 0
    assert r.keyword_score > 0


def test_modes_differ(store):
    kw = store.recall("tabs spaces", k=3, mode="keyword", touch=False)
    vec = store.recall("tabs spaces", k=3, mode="vector", touch=False)
    assert kw[0].vector_score == 0.0
    assert vec[0].keyword_score == 0.0


def test_invalid_mode(store):
    with pytest.raises(ValueError):
        store.recall("x", mode="quantum")


def test_explain_structure(store):
    r = store.recall("API key rotation", k=1)[0]
    info = r.explain()
    assert set(info["components"]) == {
        "vector_similarity", "keyword_bm25", "combined_similarity", "recency", "importance",
    }
    assert info["memory_id"] == r.memory.id
    assert "score" in r.explain_text()


def test_deterministic_with_fixed_now(store):
    a = store.recall("deployment", k=3, touch=False, now=1_700_000_000.0)
    b = store.recall("deployment", k=3, touch=False, now=1_700_000_000.0)
    assert [(r.memory.id, r.score) for r in a] == [(r.memory.id, r.score) for r in b]
