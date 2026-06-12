"""MemoryStore: SQLite-backed persistent memory with semantic recall."""
from __future__ import annotations

import json
import sqlite3
import struct
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

from .embedders import Embedder, HashEmbedder, cosine_similarity
from .models import Memory, RecallResult
from .scoring import ScoringConfig, composite_score

_SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    id TEXT PRIMARY KEY,
    namespace TEXT NOT NULL,
    content TEXT NOT NULL,
    importance REAL NOT NULL,
    metadata TEXT,
    created_at REAL NOT NULL,
    last_accessed_at REAL,
    access_count INTEGER NOT NULL DEFAULT 0,
    embedding BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_memories_namespace ON memories(namespace);
CREATE INDEX IF NOT EXISTS idx_memories_created ON memories(created_at);
"""


def _pack(vec: Sequence[float]) -> bytes:
    return struct.pack(f"<{len(vec)}f", *vec)


def _unpack(blob: bytes) -> List[float]:
    n = len(blob) // 4
    return list(struct.unpack(f"<{n}f", blob))


class MemoryStore:
    """Local-first persistent memory for AI agents.

    Example::

        from remembrane import MemoryStore

        mem = MemoryStore("agent.db")
        mem.store("User prefers dark mode", importance=0.8)
        results = mem.recall("what theme does the user like?")
        print(results[0].memory.content)

    Args:
        path: SQLite file path, or ":memory:" for an ephemeral store.
        embedder: Any object with ``embed(texts) -> List[List[float]]`` and a
            ``dimension`` attribute. Defaults to the dependency-free HashEmbedder.
        scoring: ScoringConfig controlling similarity/recency/importance weights.
    """

    def __init__(
        self,
        path: Union[str, Path] = ":memory:",
        embedder: Optional[Embedder] = None,
        scoring: Optional[ScoringConfig] = None,
    ):
        self.path = str(path)
        self.embedder = embedder or HashEmbedder()
        self.scoring = scoring or ScoringConfig()
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    # ------------------------------------------------------------------ write

    def store(
        self,
        content: str,
        *,
        namespace: str = "default",
        importance: float = 0.5,
        metadata: Optional[Dict[str, Any]] = None,
        memory_id: Optional[str] = None,
    ) -> Memory:
        """Persist a memory and index its embedding. Returns the Memory."""
        mem = Memory(
            content=content,
            namespace=namespace,
            importance=importance,
            metadata=metadata or {},
        )
        if memory_id:
            mem.id = memory_id
        vec = self.embedder.embed([content])[0]
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO memories VALUES (?,?,?,?,?,?,?,?,?)",
                (*mem.to_row(), _pack(vec)),
            )
            self._conn.commit()
        return mem

    def store_many(self, contents: Sequence[str], **kwargs) -> List[Memory]:
        """Batch store. Embeds all contents in one embedder call."""
        vecs = self.embedder.embed(list(contents))
        out = []
        with self._lock:
            for content, vec in zip(contents, vecs):
                mem = Memory(
                    content=content,
                    namespace=kwargs.get("namespace", "default"),
                    importance=kwargs.get("importance", 0.5),
                    metadata=dict(kwargs.get("metadata") or {}),
                )
                self._conn.execute(
                    "INSERT OR REPLACE INTO memories VALUES (?,?,?,?,?,?,?,?,?)",
                    (*mem.to_row(), _pack(vec)),
                )
                out.append(mem)
            self._conn.commit()
        return out

    # ------------------------------------------------------------------- read

    def recall(
        self,
        query: str,
        *,
        k: int = 5,
        namespace: Optional[str] = "default",
        min_score: float = 0.0,
        touch: bool = True,
    ) -> List[RecallResult]:
        """Return the top-k memories ranked by similarity x recency x importance.

        Args:
            query: Natural-language query.
            k: Max results.
            namespace: Restrict to one namespace; pass None to search all.
            min_score: Drop results scoring below this.
            touch: If True (default), bump access stats on returned memories,
                which strengthens them against decay (spaced-repetition style).
        """
        qvec = self.embedder.embed([query])[0]
        now = time.time()
        results: List[RecallResult] = []
        with self._lock:
            if namespace is None:
                rows = self._conn.execute(
                    "SELECT id, namespace, content, importance, metadata, created_at,"
                    " last_accessed_at, access_count, embedding FROM memories"
                ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT id, namespace, content, importance, metadata, created_at,"
                    " last_accessed_at, access_count, embedding FROM memories WHERE namespace=?",
                    (namespace,),
                ).fetchall()
        for row in rows:
            mem = Memory.from_row(row[:8])
            sim = cosine_similarity(qvec, _unpack(row[8]))
            score, rec = composite_score(sim, mem, self.scoring, now=now)
            if score >= min_score:
                results.append(RecallResult(memory=mem, similarity=sim, recency=rec, score=score))
        results.sort(key=lambda r: r.score, reverse=True)
        top = results[:k]
        if touch and top:
            with self._lock:
                self._conn.executemany(
                    "UPDATE memories SET last_accessed_at=?, access_count=access_count+1 WHERE id=?",
                    [(now, r.memory.id) for r in top],
                )
                self._conn.commit()
        return top

    def get(self, memory_id: str) -> Optional[Memory]:
        with self._lock:
            row = self._conn.execute(
                "SELECT id, namespace, content, importance, metadata, created_at,"
                " last_accessed_at, access_count FROM memories WHERE id=?",
                (memory_id,),
            ).fetchone()
        return Memory.from_row(row) if row else None

    def all(self, namespace: Optional[str] = None) -> List[Memory]:
        with self._lock:
            if namespace is None:
                rows = self._conn.execute(
                    "SELECT id, namespace, content, importance, metadata, created_at,"
                    " last_accessed_at, access_count FROM memories ORDER BY created_at"
                ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT id, namespace, content, importance, metadata, created_at,"
                    " last_accessed_at, access_count FROM memories WHERE namespace=? ORDER BY created_at",
                    (namespace,),
                ).fetchall()
        return [Memory.from_row(r) for r in rows]

    def count(self, namespace: Optional[str] = None) -> int:
        with self._lock:
            if namespace is None:
                return self._conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
            return self._conn.execute(
                "SELECT COUNT(*) FROM memories WHERE namespace=?", (namespace,)
            ).fetchone()[0]

    def namespaces(self) -> List[str]:
        with self._lock:
            rows = self._conn.execute("SELECT DISTINCT namespace FROM memories").fetchall()
        return sorted(r[0] for r in rows)

    # ----------------------------------------------------------------- modify

    def reinforce(self, memory_id: str, importance_boost: float = 0.1) -> Optional[Memory]:
        """Strengthen a memory: bump importance (capped at 1.0) and reset decay."""
        with self._lock:
            self._conn.execute(
                "UPDATE memories SET importance=MIN(1.0, importance+?),"
                " last_accessed_at=?, access_count=access_count+1 WHERE id=?",
                (importance_boost, time.time(), memory_id),
            )
            self._conn.commit()
        return self.get(memory_id)

    def forget(
        self,
        memory_id: Optional[str] = None,
        *,
        namespace: Optional[str] = None,
        older_than_seconds: Optional[float] = None,
    ) -> int:
        """Delete memories. Returns count deleted.

        - forget(id) deletes one memory.
        - forget(namespace=...) deletes a namespace.
        - forget(older_than_seconds=...) deletes stale memories (optionally within a namespace).
        """
        clauses, params = [], []
        if memory_id:
            clauses.append("id=?")
            params.append(memory_id)
        if namespace:
            clauses.append("namespace=?")
            params.append(namespace)
        if older_than_seconds is not None:
            clauses.append("COALESCE(last_accessed_at, created_at) < ?")
            params.append(time.time() - older_than_seconds)
        if not clauses:
            raise ValueError("forget() requires memory_id, namespace, or older_than_seconds")
        with self._lock:
            cur = self._conn.execute(
                f"DELETE FROM memories WHERE {' AND '.join(clauses)}", params
            )
            self._conn.commit()
        return cur.rowcount

    def consolidate(self, namespace: str = "default", similarity_threshold: float = 0.92) -> int:
        """Merge near-duplicate memories within a namespace.

        For each pair above the threshold, the older memory survives; it absorbs
        the newer one's access count, max importance, and metadata. Returns the
        number of memories removed.
        """
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, namespace, content, importance, metadata, created_at,"
                " last_accessed_at, access_count, embedding FROM memories"
                " WHERE namespace=? ORDER BY created_at",
                (namespace,),
            ).fetchall()
        mems = [(Memory.from_row(r[:8]), _unpack(r[8])) for r in rows]
        removed_ids = set()
        merges = []  # (survivor, absorbed)
        for i in range(len(mems)):
            if mems[i][0].id in removed_ids:
                continue
            for j in range(i + 1, len(mems)):
                if mems[j][0].id in removed_ids:
                    continue
                if cosine_similarity(mems[i][1], mems[j][1]) >= similarity_threshold:
                    merges.append((mems[i][0], mems[j][0]))
                    removed_ids.add(mems[j][0].id)
        with self._lock:
            for survivor, absorbed in merges:
                merged_meta = {**absorbed.metadata, **survivor.metadata}
                self._conn.execute(
                    "UPDATE memories SET access_count=access_count+?,"
                    " importance=MAX(importance, ?), metadata=? WHERE id=?",
                    (absorbed.access_count, absorbed.importance, json.dumps(merged_meta), survivor.id),
                )
                self._conn.execute("DELETE FROM memories WHERE id=?", (absorbed.id,))
            self._conn.commit()
        return len(removed_ids)

    # ------------------------------------------------------------------ misc

    def export(self, namespace: Optional[str] = None) -> List[Dict[str, Any]]:
        """Dump memories as plain dicts (for JSON export / migration)."""
        return [
            {
                "id": m.id,
                "namespace": m.namespace,
                "content": m.content,
                "importance": m.importance,
                "metadata": m.metadata,
                "created_at": m.created_at,
                "last_accessed_at": m.last_accessed_at,
                "access_count": m.access_count,
            }
            for m in self.all(namespace)
        ]

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __enter__(self) -> "MemoryStore":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
