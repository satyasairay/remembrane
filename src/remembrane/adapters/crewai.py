"""CrewAI-compatible storage adapter.

Duck-typed against crewai's external-memory Storage interface
(save / search / reset), so it works without importing crewai itself.

Example::

    from remembrane import MemoryStore
    from remembrane.adapters import RemembraneStorage

    storage = RemembraneStorage(MemoryStore("crew.db"))
    # pass wherever CrewAI accepts a custom storage backend,
    # or call .save() / .search() directly.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..store import MemoryStore


class RemembraneStorage:
    """Storage backend backed by a remembrane MemoryStore."""

    def __init__(self, store: MemoryStore, namespace: str = "crew"):
        self.store = store
        self.namespace = namespace

    def save(self, value: Any, metadata: Optional[Dict[str, Any]] = None) -> None:
        self.store.store(str(value), namespace=self.namespace, metadata=metadata or {})

    def search(self, query: str, limit: int = 5, score_threshold: float = 0.0) -> List[Dict[str, Any]]:
        results = self.store.recall(
            query, k=limit, namespace=self.namespace, min_score=score_threshold
        )
        return [
            {
                "id": r.memory.id,
                "context": r.memory.content,
                "metadata": r.memory.metadata,
                "score": r.score,
            }
            for r in results
        ]

    def reset(self) -> None:
        self.store.forget(namespace=self.namespace)
