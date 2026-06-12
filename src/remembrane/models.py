"""Data models for remembrane."""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class Memory:
    """A single unit of agent memory.

    Attributes:
        content: The text content of the memory.
        id: Unique identifier (uuid4 hex by default).
        namespace: Logical grouping (e.g. per-agent, per-user, per-project).
        importance: 0.0-1.0 weight set by the caller; influences recall ranking.
        metadata: Arbitrary JSON-serializable dict.
        created_at: Unix timestamp of creation.
        last_accessed_at: Unix timestamp of the most recent recall hit.
        access_count: Number of times this memory was returned by recall().
        usefulness: Signed outcome-feedback accumulator. Positive: this memory
            helped the agent complete tasks; negative: recalled but useless.
    """

    content: str
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    namespace: str = "default"
    importance: float = 0.5
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    last_accessed_at: Optional[float] = None
    access_count: int = 0
    usefulness: float = 0.0

    def __post_init__(self) -> None:
        if not self.content or not self.content.strip():
            raise ValueError("Memory content must be a non-empty string")
        if not 0.0 <= self.importance <= 1.0:
            raise ValueError("importance must be between 0.0 and 1.0")

    def to_row(self) -> tuple:
        return (
            self.id,
            self.namespace,
            self.content,
            self.importance,
            json.dumps(self.metadata),
            self.created_at,
            self.last_accessed_at,
            self.access_count,
            self.usefulness,
        )

    @classmethod
    def from_row(cls, row: tuple) -> "Memory":
        return cls(
            id=row[0],
            namespace=row[1],
            content=row[2],
            importance=row[3],
            metadata=json.loads(row[4]) if row[4] else {},
            created_at=row[5],
            last_accessed_at=row[6],
            access_count=row[7],
            usefulness=row[8] if len(row) > 8 else 0.0,
        )


@dataclass
class RecallResult:
    """A memory returned by recall, with every ranking signal exposed."""

    memory: Memory
    similarity: float       # combined text relevance used for ranking
    recency: float          # exp-decay factor in [0, 1]
    score: float            # final composite score
    vector_score: float = 0.0   # exact cosine similarity component
    keyword_score: float = 0.0  # exact BM25 component (normalized)
    tokens: int = 0             # estimated token cost (populated by pack())

    def explain(self) -> Dict[str, Any]:
        """Full breakdown of why this memory was recalled. Glass box, not black box."""
        return {
            "content": self.memory.content,
            "score": round(self.score, 4),
            "components": {
                "vector_similarity": round(self.vector_score, 4),
                "keyword_bm25": round(self.keyword_score, 4),
                "combined_similarity": round(self.similarity, 4),
                "recency": round(self.recency, 4),
                "importance": self.memory.importance,
                "usefulness": round(self.memory.usefulness, 4),
            },
            "memory_id": self.memory.id,
            "namespace": self.memory.namespace,
            "access_count": self.memory.access_count,
        }

    def explain_text(self) -> str:
        """Human-readable one-paragraph explanation."""
        return (
            f"score {self.score:.3f} = similarity {self.similarity:.3f} "
            f"(vector {self.vector_score:.3f}, keyword {self.keyword_score:.3f}) "
            f"+ recency {self.recency:.3f} + importance {self.memory.importance:.2f} "
            f"| recalled {self.memory.access_count}x"
        )

    def __repr__(self) -> str:  # pragma: no cover
        return f"RecallResult(score={self.score:.3f}, content={self.memory.content[:60]!r})"


@dataclass
class JournalEntry:
    """One event in the memory journal (append-only history)."""

    seq: int
    ts: float
    op: str                  # store | forget | reinforce | merge | consolidate
    memory_id: str
    namespace: str
    payload: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:  # pragma: no cover
        return f"JournalEntry(#{self.seq} {self.op} {self.memory_id[:8]})"
