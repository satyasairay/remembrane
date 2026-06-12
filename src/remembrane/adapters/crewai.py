"""CrewAI-compatible storage adapter.

A remembrane-backed storage helper for CrewAI-style pipelines: save / search /
delete / update / list_records / get_record / count / reset, duck-typed, with
kwargs tolerance (scope_prefix etc.). crewai itself is never imported.

HONEST LIMITATION: this is NOT a registered subclass of crewai's
``StorageBackend`` ABC, and ``search()`` returns dicts rather than
``(MemoryRecord, score)`` tuples — so wiring it directly into
``crewai.Memory(...)`` does not work as of crewai 1.14. Native StorageBackend
integration is tracked at:
https://github.com/satyasairay/remembrane/issues
Use it directly, or behind your own thin StorageBackend shim.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..store import MemoryStore


class RemembraneStorage:
    """Storage backend backed by a remembrane MemoryStore."""

    def __init__(self, store: MemoryStore, namespace: str = "crew"):
        self.store = store
        self.namespace = namespace

    def save(self, value: Any, metadata: Optional[Dict[str, Any]] = None, **kwargs: Any) -> str:
        meta = dict(metadata) if isinstance(metadata, dict) else {}
        for key, val in kwargs.items():  # keep scope hints etc. as metadata
            if isinstance(val, (str, int, float, bool)):
                meta.setdefault(key, val)
        mem = self.store.store(str(value), namespace=self.namespace, metadata=meta)
        return mem.id

    def search(
        self,
        query: str,
        limit: int = 5,
        score_threshold: float = 0.0,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        results = self.store.recall(
            str(query), k=int(limit), namespace=self.namespace, min_score=score_threshold
        )
        return [
            {
                "id": r.memory.id,
                "context": r.memory.content,
                "content": r.memory.content,
                "metadata": r.memory.metadata,
                "score": r.score,
            }
            for r in results
        ]

    def delete(self, record_id: Optional[str] = None, **kwargs: Any) -> bool:
        if record_id is None:
            return False
        return self.store.forget(record_id) > 0

    def update(
        self,
        record_id: str,
        value: Any = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> bool:
        existing = self.store.get(record_id)
        if existing is None:
            return False
        content = str(value) if value is not None else existing.content
        meta = {**existing.metadata, **(metadata or {})}
        self.store.forget(record_id)
        self.store.store(
            content, namespace=self.namespace, importance=existing.importance,
            metadata=meta, memory_id=record_id,
        )
        return True

    def list_records(self, limit: int = 100, **kwargs: Any) -> List[Dict[str, Any]]:
        return [
            {"id": m.id, "content": m.content, "metadata": m.metadata}
            for m in self.store.all(self.namespace)[: int(limit)]
        ]

    def get_record(self, record_id: str, **kwargs: Any) -> Optional[Dict[str, Any]]:
        m = self.store.get(record_id)
        if m is None:
            return None
        return {"id": m.id, "content": m.content, "metadata": m.metadata}

    def count(self, **kwargs: Any) -> int:
        return self.store.count(self.namespace)

    def reset(self) -> None:
        self.store.forget(namespace=self.namespace)
