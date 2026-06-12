"""Test helpers: assert what your agent remembers, deterministically, in CI.

remembrane's default embedder and ranking are fully deterministic, and
``recall(now=...)`` freezes time — so memory behavior is unit-testable in a way
cloud memory APIs can never be.

Example::

    from remembrane import MemoryStore
    from remembrane.testing import assert_recalls, assert_recalls_first

    def test_agent_remembers_preferences():
        mem = MemoryStore(":memory:")
        mem.store("user prefers dark mode", importance=0.8)
        mem.store("user is allergic to peanuts", importance=1.0)
        assert_recalls(mem, "what theme does the user like?", "dark mode")
        assert_recalls_first(mem, "any food allergies?", "peanuts")
"""
from __future__ import annotations

from typing import List, Optional

from .store import MemoryStore


def recalled_contents(
    store: MemoryStore,
    query: str,
    *,
    k: int = 5,
    namespace: Optional[str] = "default",
    mode: str = "hybrid",
    now: Optional[float] = None,
) -> List[str]:
    """Contents recalled for a query, without touching access stats."""
    return [
        r.memory.content
        for r in store.recall(query, k=k, namespace=namespace, touch=False, mode=mode, now=now)
    ]


def _fail(message: str, contents: List[str]) -> None:
    listing = "\n".join(f"  {i + 1}. {c}" for i, c in enumerate(contents)) or "  (nothing)"
    raise AssertionError(f"{message}\nRecalled instead:\n{listing}")


def assert_recalls(store: MemoryStore, query: str, expected: str, **kw) -> None:
    """Assert that some recalled memory contains `expected` (case-insensitive)."""
    contents = recalled_contents(store, query, **kw)
    if not any(expected.lower() in c.lower() for c in contents):
        _fail(f"Expected a recalled memory containing {expected!r} for query {query!r}.", contents)


def assert_recalls_first(store: MemoryStore, query: str, expected: str, **kw) -> None:
    """Assert that the TOP recalled memory contains `expected` (case-insensitive)."""
    contents = recalled_contents(store, query, **kw)
    if not contents:
        _fail(f"Expected top recall containing {expected!r} for query {query!r}.", contents)
    if expected.lower() not in contents[0].lower():
        _fail(
            f"Expected TOP recall to contain {expected!r} for query {query!r}, "
            f"but top was {contents[0]!r}.",
            contents,
        )


def assert_not_recalls(store: MemoryStore, query: str, unexpected: str, **kw) -> None:
    """Assert that NO recalled memory contains `unexpected` (case-insensitive)."""
    contents = recalled_contents(store, query, **kw)
    if any(unexpected.lower() in c.lower() for c in contents):
        _fail(f"Expected no recalled memory containing {unexpected!r} for query {query!r}.", contents)
