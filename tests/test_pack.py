import pytest

from remembrane import MemoryStore
from remembrane.conflicts import estimate_tokens, knapsack_pack


@pytest.fixture()
def store():
    s = MemoryStore(":memory:")
    s.store("user prefers dark mode in all editors and terminals")
    s.store("user prefers dark mode!")  # near-duplicate (same tokens)
    s.store("the deployment target is AWS us-east-1 region")
    s.store("user timezone is IST, working hours 10am to 7pm")
    yield s
    s.close()


def test_knapsack_respects_budget():
    sel, total = knapsack_pack([1.0, 1.0, 1.0], [50, 50, 50], budget=100)
    assert len(sel) == 2
    assert total <= 100


def test_knapsack_prefers_value():
    sel, _ = knapsack_pack([0.9, 0.1, 0.85], [60, 60, 60], budget=130)
    assert set(sel) == {0, 2}


def test_knapsack_empty():
    assert knapsack_pack([], [], 100) == ([], 0)
    assert knapsack_pack([1.0], [10], 0) == ([], 0)


def test_pack_within_budget(store):
    chosen = store.pack("user preferences", budget_tokens=30, touch=False)
    assert chosen
    assert sum(r.tokens for r in chosen) <= 30


def test_pack_dedupes_near_duplicates():
    s = MemoryStore(":memory:")
    s.store("user prefers dark mode")
    s.store("user prefers dark mode!")   # same tokens -> near-identical embedding
    s.store("deployment target is AWS")
    chosen = s.pack("dark mode preferences", budget_tokens=10_000, touch=False)
    contents = [r.memory.content for r in chosen]
    assert sum("dark mode" in c for c in contents) == 1
    s.close()


def test_pack_populates_tokens(store):
    chosen = store.pack("anything at all", budget_tokens=500, touch=False)
    for r in chosen:
        assert r.tokens == estimate_tokens(r.memory.content)


def test_pack_custom_estimator(store):
    chosen = store.pack(
        "user preferences", budget_tokens=2, touch=False,
        token_estimator=lambda text: 1,
    )
    assert len(chosen) <= 2


def test_pack_touch_updates_access(store):
    chosen = store.pack("deployment", budget_tokens=200, touch=True)
    assert chosen
    assert store.get(chosen[0].memory.id).access_count == 1


def test_pack_empty_store():
    s = MemoryStore(":memory:")
    assert s.pack("anything", budget_tokens=100) == []
    s.close()


def test_pack_deterministic(store):
    a = store.pack("user preferences", budget_tokens=60, touch=False, now=1_700_000_000.0)
    b = store.pack("user preferences", budget_tokens=60, touch=False, now=1_700_000_000.0)
    assert [r.memory.id for r in a] == [r.memory.id for r in b]
