"""MemoryStore: SQLite-backed persistent memory with exact hybrid recall and history."""
from __future__ import annotations

import json
import math
import sqlite3
import struct
import threading
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

from .bm25 import normalized_bm25_tokens, term_frequencies, tokenize
from .conflicts import Conflict, detect_conflicts, estimate_tokens, knapsack_pack
from .embedders import Embedder, HashEmbedder, cosine_similarity
from .models import JournalEntry, Memory, RecallResult
from .scoring import ScoringConfig, usefulness_factor

try:  # optional acceleration — pip install remembrane[fast]
    import numpy as _np
except Exception:  # pragma: no cover — a broken numpy must never break remembrane
    _np = None

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
    usefulness REAL NOT NULL DEFAULT 0,
    embedding BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_memories_namespace ON memories(namespace);
CREATE INDEX IF NOT EXISTS idx_memories_created ON memories(created_at);
CREATE TABLE IF NOT EXISTS journal (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    op TEXT NOT NULL,
    memory_id TEXT NOT NULL,
    namespace TEXT NOT NULL,
    payload TEXT
);
CREATE TABLE IF NOT EXISTS snapshots (
    label TEXT PRIMARY KEY,
    ts REAL NOT NULL
);
"""

_MEM_COLS = (
    "id, namespace, content, importance, metadata, created_at, last_accessed_at,"
    " access_count, usefulness"
)

_INSERT_SQL = (
    "INSERT OR REPLACE INTO memories"
    " (id, namespace, content, importance, metadata, created_at, last_accessed_at,"
    " access_count, usefulness, embedding) VALUES (?,?,?,?,?,?,?,?,?,?)"
)


def _pack(vec: Sequence[float]) -> bytes:
    return struct.pack(f"<{len(vec)}f", *vec)


def _safe_payload(raw) -> Dict[str, Any]:
    """Journal payloads must never crash history reconstruction."""
    if not raw:
        return {}
    try:
        data = json.loads(raw)
        return data if isinstance(data, dict) else {"_corrupt": str(data)[:200]}
    except (json.JSONDecodeError, TypeError):
        return {"_corrupt": str(raw)[:200]}


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
        print(results[0].explain_text())

    Args:
        path: SQLite file path, or ":memory:" for an ephemeral store.
        embedder: Any object with ``embed(texts) -> List[List[float]]`` and a
            ``dimension`` attribute. Defaults to the dependency-free HashEmbedder.
        scoring: ScoringConfig controlling ranking weights and decay half-life.
        journal_mode: SQLite journal mode for file-backed dbs ("WAL" default;
            pass "DELETE" for the pre-0.5 single-file behavior, e.g. on
            network filesystems).
    """

    def __init__(
        self,
        path: Union[str, Path] = ":memory:",
        embedder: Optional[Embedder] = None,
        scoring: Optional[ScoringConfig] = None,
        journal_mode: str = "WAL",
    ):
        self.path = str(path)
        if self.path != ":memory:":
            parent = Path(self.path).expanduser().resolve().parent
            parent.mkdir(parents=True, exist_ok=True)
        self.embedder = embedder or HashEmbedder()
        self.scoring = scoring or ScoringConfig()
        self._lock = threading.RLock()
        self._corpus: Dict[Any, Dict[str, Any]] = {}
        # isolation_level="IMMEDIATE": write transactions take the write lock up
        # front, so busy_timeout can wait politely instead of deadlocking on a
        # deferred-to-write lock upgrade under concurrency.
        self._conn = sqlite3.connect(
            self.path, check_same_thread=False, isolation_level="IMMEDIATE"
        )
        if self.path != ":memory:":
            # WAL allows concurrent readers during writes; busy_timeout waits out
            # short lock contention instead of raising. Note: WAL keeps transient
            # -wal/-shm sidecar files next to the db, and is unsafe on network
            # filesystems (NFS/SMB) — see README scope notes.
            self._conn.execute(f"PRAGMA journal_mode={journal_mode}")
            self._conn.execute("PRAGMA busy_timeout=30000")
        self._data_version = self._conn.execute("PRAGMA data_version").fetchone()[0]
        self._conn.executescript(_SCHEMA)
        cols = {r[1] for r in self._conn.execute("PRAGMA table_info(memories)").fetchall()}
        if "usefulness" not in cols:  # migrate dbs created before 0.3.0
            self._conn.execute("ALTER TABLE memories ADD COLUMN usefulness REAL NOT NULL DEFAULT 0")
        self._conn.commit()

    # -------------------------------------------------------------- write retry

    def _retry_locked(self, fn, attempts: int = 6):
        """Run a write closure, retrying on transient SQLITE_BUSY.

        WAL + busy_timeout handle most contention; this catches the residual
        snapshot-upgrade races SQLite reports immediately instead of waiting.
        """
        for attempt in range(attempts):
            try:
                return fn()
            except sqlite3.OperationalError as exc:
                if "locked" not in str(exc).lower() or attempt == attempts - 1:
                    raise
                try:
                    self._conn.rollback()
                except sqlite3.OperationalError:
                    pass
                time.sleep(0.02 * (2 ** attempt))

    # ----------------------------------------------------------- corpus cache

    def _invalidate(self) -> None:
        self._corpus.clear()

    def _check_external_changes(self) -> None:
        """Invalidate caches when ANOTHER connection (thread/process) wrote the db.

        SQLite's data_version pragma changes only when a different connection
        commits — our own writes are handled by _invalidate() directly.
        """
        version = self._conn.execute("PRAGMA data_version").fetchone()[0]
        if version != self._data_version:
            self._data_version = version
            self._corpus.clear()

    def _get_corpus(self, namespace: Optional[str]) -> Dict[str, Any]:
        """Cached embeddings + tokenizations per namespace (numpy matrix when available).

        Invalidated on every mutation; recall reads stay O(1) SQL + vector math.
        """
        self._check_external_changes()
        key = namespace
        cached = self._corpus.get(key)
        if cached is not None:
            return cached
        with self._lock:
            if namespace is None:
                rows = self._conn.execute("SELECT id, content, embedding FROM memories").fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT id, content, embedding FROM memories WHERE namespace=?", (namespace,)
                ).fetchall()
        expected_len = 4 * self.embedder.dimension
        healed = []
        fixed_rows = []
        for r in rows:
            if r[2] is None or len(r[2]) != expected_len:
                vec = self.embedder.embed([r[1]])[0]  # self-heal: re-embed from content
                blob = _pack(vec)
                healed.append((blob, r[0]))
                fixed_rows.append((r[0], r[1], blob))
            else:
                fixed_rows.append(r)
        if healed:
            with self._lock:
                self._conn.executemany("UPDATE memories SET embedding=? WHERE id=?", healed)
                self._conn.commit()
            healed_ids = [h[1] for h in healed]
            shown = ", ".join(healed_ids[:5]) + ("..." if len(healed_ids) > 5 else "")
            warnings.warn(
                f"remembrane: {len(healed_ids)} memory embedding blob(s) were missing or "
                f"corrupt and have been re-embedded from content (ids: {shown}). "
                "If this recurs, check for external writers or storage corruption.",
                RuntimeWarning,
                stacklevel=3,
            )
        rows = fixed_rows
        ids = [r[0] for r in rows]
        tokens = [tokenize(r[1]) for r in rows]
        corpus: Dict[str, Any] = {"ids": ids, "idx": {mid: i for i, mid in enumerate(ids)},
                                  "tokens": tokens,
                                  "tfs": [term_frequencies(t) for t in tokens],
                                  "doc_lens": [len(t) for t in tokens],
                                  "matrix": None, "norms": None, "vecs": None}
        if rows:
            if _np is not None:
                matrix = _np.vstack([_np.frombuffer(r[2], dtype=_np.float32) for r in rows])
                matrix = _np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
                norms = _np.linalg.norm(matrix, axis=1)
                norms[norms == 0] = 1.0
                corpus["matrix"] = matrix
                corpus["norms"] = norms
            else:
                corpus["vecs"] = [_unpack(r[2]) for r in rows]
        self._corpus[key] = corpus
        return corpus

    def _similarities(self, qvec: List[float], corpus: Dict[str, Any]) -> List[float]:
        """Exact cosine similarity of the query against every cached vector."""
        if not corpus["ids"]:
            return []
        if _np is not None and corpus["matrix"] is not None:
            q = _np.asarray(qvec, dtype=_np.float32)
            qn = float(_np.linalg.norm(q)) or 1.0
            sims = (corpus["matrix"] @ q) / (corpus["norms"] * qn)
            return sims.tolist()
        return [cosine_similarity(qvec, v) for v in corpus["vecs"]]

    # ---------------------------------------------------------------- journal

    def _journal(self, op: str, memory_id: str, namespace: str, payload: Dict[str, Any]) -> None:
        self._conn.execute(
            "INSERT INTO journal (ts, op, memory_id, namespace, payload) VALUES (?,?,?,?,?)",
            (time.time(), op, memory_id, namespace, json.dumps(payload)),
        )

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

        def _write():
            with self._lock:
                self._conn.execute(_INSERT_SQL, (*mem.to_row(), _pack(vec)))
                self._invalidate()
                self._journal(
                    "store",
                    mem.id,
                    namespace,
                    {"content": content, "importance": importance, "metadata": mem.metadata,
                     "created_at": mem.created_at},
                )
                self._conn.commit()

        self._retry_locked(_write)
        return mem

    def store_many(self, contents: Sequence[str], **kwargs) -> List[Memory]:
        """Batch store. Embeds all contents in one embedder call."""
        vecs = self.embedder.embed(list(contents))
        out = []

        def _write():
            out.clear()
            with self._lock:
                self._write_many(contents, vecs, kwargs, out)

        self._retry_locked(_write)
        return out

    def _write_many(self, contents, vecs, kwargs, out):
            for content, vec in zip(contents, vecs):
                mem = Memory(
                    content=content,
                    namespace=kwargs.get("namespace", "default"),
                    importance=kwargs.get("importance", 0.5),
                    metadata=dict(kwargs.get("metadata") or {}),
                )
                self._conn.execute(_INSERT_SQL, (*mem.to_row(), _pack(vec)))
                self._invalidate()
                self._journal(
                    "store",
                    mem.id,
                    mem.namespace,
                    {"content": content, "importance": mem.importance,
                     "metadata": mem.metadata, "created_at": mem.created_at},
                )
                out.append(mem)
            self._conn.commit()

    # ------------------------------------------------------------------- read

    def recall(
        self,
        query: str,
        *,
        k: int = 5,
        namespace: Optional[str] = "default",
        min_score: float = 0.0,
        min_similarity: float = 0.0,
        touch: bool = True,
        mode: str = "hybrid",
        now: Optional[float] = None,
    ) -> List[RecallResult]:
        """Return the top-k memories, ranked by similarity, recency, importance, and earned usefulness.

        Search is *exact*, not approximate: every memory in scope is scored.

        Args:
            query: Natural-language query.
            k: Max results.
            namespace: Restrict to one namespace; pass None to search all.
            min_score: Drop results scoring below this.
            min_similarity: Results must exceed this text relevance to qualify
                at all (default 0.0 — zero-relevance memories are never
                returned, regardless of recency or importance).
            touch: If True (default), bump access stats on returned memories,
                which strengthens them against decay (spaced-repetition style).
            mode: "hybrid" (vector + BM25 keyword, default), "vector", or "keyword".
            now: Override the current timestamp — makes recall fully
                deterministic for tests and replay.
        """
        if not isinstance(query, str):
            raise ValueError("query must be a string")
        if mode not in ("hybrid", "vector", "keyword"):
            raise ValueError("mode must be 'hybrid', 'vector', or 'keyword'")
        now = now if now is not None else time.time()
        with self._lock:
            if namespace is None:
                rows = self._conn.execute(f"SELECT {_MEM_COLS} FROM memories").fetchall()
            else:
                rows = self._conn.execute(
                    f"SELECT {_MEM_COLS} FROM memories WHERE namespace=?", (namespace,)
                ).fetchall()
        if not rows:
            return []

        corpus = self._get_corpus(namespace)
        if any(r[0] not in corpus["idx"] for r in rows):
            # another writer changed the db between the row fetch and cache read
            self._invalidate()
            corpus = self._get_corpus(namespace)
            rows = [r for r in rows if r[0] in corpus["idx"]]
            if not rows:
                return []
        order = [corpus["idx"][r[0]] for r in rows]

        if mode in ("hybrid", "vector"):
            qvec = self.embedder.embed([query])[0]
            sims_all = self._similarities(qvec, corpus)
            vec_scores = [sims_all[i] for i in order]
        else:
            vec_scores = [0.0] * len(rows)

        if mode in ("hybrid", "keyword"):
            kw_all = normalized_bm25_tokens(
                tokenize(query), corpus["tokens"],
                tfs=corpus["tfs"], doc_lens=corpus["doc_lens"],
            )
            kw_scores = [kw_all[i] for i in order]
        else:
            kw_scores = [0.0] * len(rows)

        # score raw rows; build Memory objects only for the winners
        cfg = self.scoring
        kw_weight = cfg.keyword_weight
        ln2 = math.log(2)
        scored = []  # (score, sim, rec, vs, ks, row)
        for row, vs, ks in zip(rows, vec_scores, kw_scores):
            if mode == "hybrid":
                sim = (1 - kw_weight) * vs + kw_weight * ks
            elif mode == "vector":
                sim = vs
            else:
                sim = ks
            if sim <= min_similarity:
                continue  # recency/importance rank relevant memories; they never substitute for relevance
            anchor = row[6] if row[6] is not None else row[5]
            rec = math.exp(-ln2 * max(0.0, now - anchor) / cfg.half_life_seconds)
            score = (
                cfg.weight_similarity * sim
                + cfg.weight_recency * rec
                + cfg.weight_importance * row[3]
                + cfg.weight_usefulness * usefulness_factor(row[8])
            )
            if score >= min_score:
                scored.append((score, sim, rec, vs, ks, row))
        scored.sort(key=lambda t: t[0], reverse=True)
        top = [
            RecallResult(
                memory=Memory.from_row(row), similarity=sim, recency=rec,
                score=score, vector_score=vs, keyword_score=ks,
            )
            for score, sim, rec, vs, ks, row in scored[:k]
        ]
        if touch and top:
            def _touch():
                with self._lock:
                    self._conn.executemany(
                        "UPDATE memories SET last_accessed_at=?, access_count=access_count+1 WHERE id=?",
                        [(now, r.memory.id) for r in top],
                    )
                    self._conn.commit()
            self._retry_locked(_touch)
        return top

    def get(self, memory_id: str) -> Optional[Memory]:
        with self._lock:
            row = self._conn.execute(
                f"SELECT {_MEM_COLS} FROM memories WHERE id=?", (memory_id,)
            ).fetchone()
        return Memory.from_row(row) if row else None

    def all(self, namespace: Optional[str] = None) -> List[Memory]:
        with self._lock:
            if namespace is None:
                rows = self._conn.execute(
                    f"SELECT {_MEM_COLS} FROM memories ORDER BY created_at"
                ).fetchall()
            else:
                rows = self._conn.execute(
                    f"SELECT {_MEM_COLS} FROM memories WHERE namespace=? ORDER BY created_at",
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
            mem = self.get(memory_id)
            if mem:
                self._journal("reinforce", memory_id, mem.namespace,
                              {"importance": mem.importance})
            self._conn.commit()
        return mem

    def forget(
        self,
        memory_id: Optional[str] = None,
        *,
        namespace: Optional[str] = None,
        older_than_seconds: Optional[float] = None,
    ) -> int:
        """Delete memories. Returns count deleted."""
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
        where = " AND ".join(clauses)
        with self._lock:
            doomed = self._conn.execute(
                f"SELECT id, namespace FROM memories WHERE {where}", params
            ).fetchall()
            cur = self._conn.execute(f"DELETE FROM memories WHERE {where}", params)
            self._invalidate()
            for mid, ns in doomed:
                self._journal("forget", mid, ns, {})
            self._conn.commit()
        return cur.rowcount

    def consolidate(self, namespace: str = "default", similarity_threshold: float = 0.92) -> int:
        """Merge near-duplicate memories within a namespace. Returns memories removed."""
        with self._lock:
            rows = self._conn.execute(
                f"SELECT {_MEM_COLS}, embedding FROM memories WHERE namespace=? ORDER BY created_at",
                (namespace,),
            ).fetchall()
        mems = [(Memory.from_row(r[:9]), _unpack(r[9])) for r in rows]
        removed_ids = set()
        merges = []
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
                new_importance = max(survivor.importance, absorbed.importance)
                self._conn.execute(
                    "UPDATE memories SET access_count=access_count+?,"
                    " importance=?, metadata=? WHERE id=?",
                    (absorbed.access_count, new_importance, json.dumps(merged_meta), survivor.id),
                )
                self._conn.execute("DELETE FROM memories WHERE id=?", (absorbed.id,))
                self._journal("consolidate", survivor.id, namespace,
                              {"absorbed": absorbed.id, "importance": new_importance,
                               "metadata": merged_meta})
                self._journal("forget", absorbed.id, namespace, {"reason": "consolidated"})
            self._invalidate()
            self._conn.commit()
        return len(removed_ids)


    # --------------------------------------------------------------- feedback

    def feedback(self, memory_id: str, useful: bool, weight: float = 1.0) -> Optional[Memory]:
        """Record a task outcome for a recalled memory.

        Positive feedback raises the memory's earned usefulness (sigmoid-squashed
        into ranking); negative lowers it. This is how salience is *learned from
        outcomes* instead of guessed at write time.
        """
        if weight <= 0:
            raise ValueError("weight must be positive")
        delta = weight if useful else -weight
        with self._lock:
            self._conn.execute(
                "UPDATE memories SET usefulness = usefulness + ? WHERE id=?",
                (delta, memory_id),
            )
            mem = self.get(memory_id)
            if mem:
                self._journal("feedback", memory_id, mem.namespace,
                              {"useful": useful, "weight": weight,
                               "usefulness": mem.usefulness})
            self._conn.commit()
        return mem

    def mark_useful(self, memory_id: str, weight: float = 1.0) -> Optional[Memory]:
        """This memory helped the agent complete its task."""
        return self.feedback(memory_id, True, weight)

    def mark_useless(self, memory_id: str, weight: float = 1.0) -> Optional[Memory]:
        """This memory was recalled but contributed nothing."""
        return self.feedback(memory_id, False, weight)

    # -------------------------------------------------------------- conflicts

    def conflicts(
        self,
        query: Optional[str] = None,
        *,
        namespace: Optional[str] = "default",
        k: int = 50,
        similarity_floor: float = 0.35,
        similarity_ceiling: float = 0.92,
        min_confidence: str = "possible",
    ) -> List[Conflict]:
        """Surface memories in tension, instead of silently picking a winner.

        With a query, only memories relevant to it are examined (the agent's
        "wait — do I hold conflicting beliefs about this?" check before
        answering). Without one, the whole namespace is swept.

        Detection is deterministic and free; adjudication is the agent's job —
        resolve() the loser, or ask the user.
        """
        if min_confidence not in ("possible", "likely"):
            raise ValueError("min_confidence must be 'possible' or 'likely'")
        if query is not None:
            results = self.recall(query, k=k, namespace=namespace, touch=False)
            mems = [r.memory for r in results]
        else:
            mems = self.all(namespace)
        if len(mems) < 2:
            return []
        vecs = self._embeddings_for([m.id for m in mems])
        found = detect_conflicts(
            mems, vecs,
            similarity_floor=similarity_floor,
            similarity_ceiling=similarity_ceiling,
        )
        if min_confidence == "likely":
            found = [c for c in found if c.confidence == "likely"]
        return found

    def resolve(self, keep_id: str, drop_ids: Sequence[str], reason: str = "") -> int:
        """Settle a conflict: keep one memory, forget the others. Fully journaled.

        The dropped memories remain reconstructable via as_of()/log() — resolution
        is an auditable decision, not an erasure of history.
        """
        kept = self.get(keep_id)
        if kept is None:
            raise KeyError(f"no memory {keep_id!r}")
        dropped = 0
        with self._lock:
            for did in drop_ids:
                mem = self.get(did)
                if mem is None:
                    continue
                self._conn.execute("DELETE FROM memories WHERE id=?", (did,))
                self._journal("resolve", keep_id, kept.namespace,
                              {"dropped": did, "dropped_content": mem.content,
                               "reason": reason})
                self._journal("forget", did, mem.namespace, {"reason": f"resolved: {reason}"})
                dropped += 1
            self._invalidate()
            self._conn.commit()
        return dropped

    def _embeddings_for(self, ids: Sequence[str]) -> List[List[float]]:
        with self._lock:
            rows = {
                r[0]: r[1]
                for r in self._conn.execute(
                    f"SELECT id, embedding FROM memories WHERE id IN ({','.join('?' * len(ids))})",
                    list(ids),
                ).fetchall()
            }
        return [_unpack(rows[i]) for i in ids]

    # ---------------------------------------------------------------- packing

    def pack(
        self,
        query: str,
        *,
        budget_tokens: int = 800,
        namespace: Optional[str] = "default",
        mode: str = "hybrid",
        candidates: int = 256,
        dedupe_threshold: float = 0.92,
        now: Optional[float] = None,
        touch: bool = True,
        token_estimator=None,
    ) -> List[RecallResult]:
        """The best possible use of a token budget, not an arbitrary top-k.

        Scores candidates exactly (like recall), drops near-duplicates so the
        budget isn't spent saying the same thing twice, then solves the packing
        problem. ``budget_tokens`` is a hard cap in every configuration; the
        selection is exactly optimal when numpy is installed and near-optimal
        (coarsened DP + greedy refill) in the pure-python fallback.
        Deterministic, no LLM.

        Args:
            token_estimator: callable(text) -> int. Defaults to a ~4 chars/token
                heuristic; pass your tokenizer's count for exact budgets.
        """
        est = token_estimator or estimate_tokens
        results = self.recall(
            query, k=candidates, namespace=namespace, mode=mode, touch=False, now=now
        )
        if not results:
            return []
        # near-duplicate suppression, keeping the higher-scored of each pair
        kept: List[int] = []
        if _np is not None:
            corpus = self._get_corpus(namespace)
            idxs = [corpus["idx"][r.memory.id] for r in results]
            sub = corpus["matrix"][idxs]
            subn = corpus["norms"][idxs]
            simmat = (sub @ sub.T) / _np.outer(subn, subn)
            for i in range(len(results)):
                if all(simmat[i, j] < dedupe_threshold for j in kept):
                    kept.append(i)
        else:
            vecs = self._embeddings_for([r.memory.id for r in results])
            for i in range(len(results)):
                if all(
                    cosine_similarity(vecs[i], vecs[j]) < dedupe_threshold for j in kept
                ):
                    kept.append(i)
        candidates_kept = [results[i] for i in kept]
        tokens = [est(r.memory.content) for r in candidates_kept]
        selected_idx, _total = knapsack_pack(
            [r.score for r in candidates_kept], tokens, budget_tokens
        )
        chosen = []
        for i in selected_idx:
            r = candidates_kept[i]
            r.tokens = tokens[i]
            chosen.append(r)
        chosen.sort(key=lambda r: r.score, reverse=True)
        if touch and chosen:
            bump_now = now if now is not None else time.time()
            with self._lock:
                self._conn.executemany(
                    "UPDATE memories SET last_accessed_at=?, access_count=access_count+1 WHERE id=?",
                    [(bump_now, r.memory.id) for r in chosen],
                )
                self._conn.commit()
        return chosen

    # ------------------------------------------------------------ time travel

    def snapshot(self, label: str) -> float:
        """Record a named point in time. Use with diff()/as_of()."""
        ts = time.time()
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO snapshots (label, ts) VALUES (?,?)", (label, ts)
            )
            self._conn.commit()
        return ts

    def snapshots(self) -> Dict[str, float]:
        with self._lock:
            rows = self._conn.execute("SELECT label, ts FROM snapshots ORDER BY ts").fetchall()
        return {r[0]: r[1] for r in rows}

    def log(self, namespace: Optional[str] = None, limit: int = 100) -> List[JournalEntry]:
        """The memory's history, newest first. Every store/forget/reinforce/merge."""
        with self._lock:
            if namespace is None:
                rows = self._conn.execute(
                    "SELECT seq, ts, op, memory_id, namespace, payload FROM journal"
                    " ORDER BY seq DESC LIMIT ?", (limit,),
                ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT seq, ts, op, memory_id, namespace, payload FROM journal"
                    " WHERE namespace=? ORDER BY seq DESC LIMIT ?", (namespace, limit),
                ).fetchall()
        return [
            JournalEntry(seq=r[0], ts=r[1], op=r[2], memory_id=r[3], namespace=r[4],
                         payload=_safe_payload(r[5]))
            for r in rows
        ]

    def _resolve_ts(self, point: Union[str, float]) -> float:
        if isinstance(point, (int, float)):
            return float(point)
        snaps = self.snapshots()
        if point not in snaps:
            raise KeyError(f"unknown snapshot {point!r}; known: {sorted(snaps)}")
        return snaps[point]

    def _state_at(self, ts: float) -> Dict[str, Dict[str, Any]]:
        """Replay the journal up to ts. Returns {memory_id: {content, importance, namespace}}."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT op, memory_id, namespace, payload FROM journal WHERE ts<=? ORDER BY seq",
                (ts,),
            ).fetchall()
        state: Dict[str, Dict[str, Any]] = {}
        for op, mid, ns, payload_json in rows:
            payload = _safe_payload(payload_json)
            if "_corrupt" in payload and op in ("store", "merge"):
                continue  # cannot reconstruct content from a corrupt entry; skip it
            if op in ("store", "merge"):
                state[mid] = {
                    "content": payload.get("content", state.get(mid, {}).get("content", "")),
                    "importance": payload.get("importance", 0.5),
                    "namespace": ns,
                }
            elif op == "forget":
                state.pop(mid, None)
            elif op in ("reinforce", "consolidate") and mid in state:
                state[mid]["importance"] = payload.get("importance", state[mid]["importance"])
        return state

    def as_of(self, point: Union[str, float]) -> List[Dict[str, Any]]:
        """What did this agent remember at a snapshot label or unix timestamp?

        Returns a list of dicts (id, content, importance, namespace) reconstructed
        from the journal — the past is read-only.
        """
        ts = self._resolve_ts(point)
        state = self._state_at(ts)
        return [
            {"id": mid, **fields}
            for mid, fields in sorted(state.items(), key=lambda kv: kv[0])
        ]

    def diff(self, a: Union[str, float], b: Union[str, float, None] = None) -> Dict[str, list]:
        """What changed between two points in time?

        Directional: diff(a, b) reads as "what happened going from a to b" —
        diff(b, a) reports the same changes inverted (added <-> removed).

        Args:
            a: Snapshot label or unix timestamp (the "before").
            b: Snapshot label or timestamp (the "after"). Defaults to now.

        Returns:
            {"added": [...], "removed": [...], "changed": [...]} where each item
            is a dict with id/content (and old/new importance for changed).
        """
        ts_a = self._resolve_ts(a)
        ts_b = self._resolve_ts(b) if b is not None else time.time()
        before = self._state_at(ts_a)
        after = self._state_at(ts_b)
        added = [
            {"id": mid, "content": f["content"], "namespace": f["namespace"]}
            for mid, f in after.items() if mid not in before
        ]
        removed = [
            {"id": mid, "content": f["content"], "namespace": f["namespace"]}
            for mid, f in before.items() if mid not in after
        ]
        changed = [
            {
                "id": mid,
                "content": after[mid]["content"],
                "importance_before": before[mid]["importance"],
                "importance_after": after[mid]["importance"],
            }
            for mid in before.keys() & after.keys()
            if before[mid]["importance"] != after[mid]["importance"]
        ]
        return {"added": added, "removed": removed, "changed": changed}

    # ------------------------------------------------------------------ merge

    def merge_from(
        self,
        source: Union[str, Path, "MemoryStore"],
        *,
        namespaces: Optional[Sequence[str]] = None,
        dedupe_threshold: float = 0.95,
    ) -> Dict[str, int]:
        """Merge another memory database into this one.

        Near-duplicates (similarity >= dedupe_threshold within the same
        namespace) are absorbed: max importance, merged metadata, summed access
        counts. Everything else is copied with original timestamps preserved.

        Args:
            source: Path to another remembrane db, or an open MemoryStore.
            namespaces: Only merge these namespaces (default: all).
            dedupe_threshold: Set to 1.1 to disable deduplication entirely.

        Returns:
            {"added": n, "merged": m}
        """
        other = source if isinstance(source, MemoryStore) else MemoryStore(source)
        close_other = not isinstance(source, MemoryStore)
        try:
            incoming = [
                m for m in other.all()
                if namespaces is None or m.namespace in namespaces
            ]
            added = merged = 0
            for mem in incoming:
                vec = self.embedder.embed([mem.content])[0]
                duplicate_id = None
                if dedupe_threshold <= 1.0:
                    with self._lock:
                        rows = self._conn.execute(
                            "SELECT id, embedding FROM memories WHERE namespace=?",
                            (mem.namespace,),
                        ).fetchall()
                    for mid, blob in rows:
                        if cosine_similarity(vec, _unpack(blob)) >= dedupe_threshold:
                            duplicate_id = mid
                            break
                with self._lock:
                    if duplicate_id:
                        existing = self.get(duplicate_id)
                        merged_meta = {**mem.metadata, **existing.metadata}
                        new_importance = max(existing.importance, mem.importance)
                        self._conn.execute(
                            "UPDATE memories SET access_count=access_count+?,"
                            " importance=?, metadata=? WHERE id=?",
                            (mem.access_count, new_importance,
                             json.dumps(merged_meta), duplicate_id),
                        )
                        self._journal("consolidate", duplicate_id, mem.namespace,
                                      {"absorbed_external": mem.id,
                                       "importance": new_importance,
                                       "metadata": merged_meta})
                        merged += 1
                    else:
                        self._conn.execute(_INSERT_SQL, (*mem.to_row(), _pack(vec)))
                        self._journal("merge", mem.id, mem.namespace,
                                      {"content": mem.content,
                                       "importance": mem.importance,
                                       "metadata": mem.metadata,
                                       "source": str(getattr(other, "path", "store"))})
                        added += 1
                    self._conn.commit()
            self._invalidate()
            return {"added": added, "merged": merged}
        finally:
            if close_other:
                other.close()

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
                "usefulness": m.usefulness,
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
