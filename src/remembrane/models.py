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
    """

    content: str
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    namespace: str = "default"
    importance: float = 0.5
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    last_accessed_at: Optional[float] = None
    access_count: int = 0

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
        )


@dataclass
class RecallResult:
    """A memory returned by recall, with its ranking signals."""

    memory: Memory
    similarity: float
    recency: float
    score: float

    def __repr__(self) -> str:  # pragma: no cover
        return f"RecallResult(score={self.score:.3f}, content={self.memory.content[:60]!r})"
