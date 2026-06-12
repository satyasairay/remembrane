"""Conflict-aware memory: surface tensions between memories instead of hiding them.

Every other memory system silently resolves contradictions with an LLM and
returns a single confident answer — which is exactly how agents end up
confidently wrong. remembrane takes the opposite stance: detection is
deterministic and free, and *adjudication belongs to the agent* (which can
reason about it, or ask the user).

Detection is heuristic and honest about being heuristic: it surfaces
*candidates* — pairs of memories about the same subject whose contents
diverge — with the signals that triggered them. It will sometimes flag benign
pairs; it will never silently merge two truths into one falsehood.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Sequence, Tuple

from .embedders import cosine_similarity
from .models import Memory

_WORD_RE = re.compile(r"[a-z0-9]+")
_NUM_RE = re.compile(r"\d+(?:\.\d+)?")

# Words that often mark a change of state — a strong hint that one memory
# updates or contradicts another.
CHANGE_MARKERS = frozenset(
    "not no never now new moved changed changing instead longer anymore was were "
    "used formerly previously old stopped quit switched cancelled canceled".split()
)

_STOPWORDS = frozenset(
    "the a an is are was were be been to of in on at for and or it its this that "
    "with by from as has have had user prefers prefer likes like".split()
)


@dataclass
class Conflict:
    """Two memories that appear to disagree about the same subject."""

    a: Memory
    b: Memory
    similarity: float
    confidence: str            # "likely" | "possible"
    signals: List[str] = field(default_factory=list)

    def describe(self) -> str:
        """Agent-ready description of the tension, with provenance."""
        newer, older = (self.a, self.b) if self.a.created_at >= self.b.created_at else (self.b, self.a)
        return (
            f"Conflicting memories ({self.confidence}, {', '.join(self.signals)}):\n"
            f"  older: {older.content!r} (recalled {older.access_count}x)\n"
            f"  newer: {newer.content!r} (recalled {newer.access_count}x)"
        )

    def __repr__(self) -> str:  # pragma: no cover
        return f"Conflict({self.confidence}: {self.a.content[:30]!r} vs {self.b.content[:30]!r})"


def _content_words(text: str) -> set:
    return {w for w in _WORD_RE.findall(text.lower()) if w not in _STOPWORDS}


def detect_conflicts(
    memories: Sequence[Memory],
    vectors: Sequence[Sequence[float]],
    *,
    similarity_floor: float = 0.35,
    similarity_ceiling: float = 0.92,
    min_anchor_overlap: float = 0.15,
) -> List[Conflict]:
    """Find pairs of memories in tension.

    A pair qualifies when it is topically similar (cosine between floor and
    ceiling — above the ceiling it's a duplicate for consolidate(), not a
    conflict) and shares anchor words, yet the contents diverge. Confidence is
    "likely" when a change marker or a numeric mismatch is present.
    """
    conflicts: List[Conflict] = []
    words = [_content_words(m.content) for m in memories]
    numbers = [set(_NUM_RE.findall(m.content)) for m in memories]
    for i in range(len(memories)):
        for j in range(i + 1, len(memories)):
            if memories[i].namespace != memories[j].namespace:
                continue
            sim = cosine_similarity(vectors[i], vectors[j])
            if not (similarity_floor <= sim < similarity_ceiling):
                continue
            union = words[i] | words[j]
            if not union:
                continue
            overlap = len(words[i] & words[j]) / len(union)
            if overlap < min_anchor_overlap:
                continue
            signals = [f"anchor_overlap={overlap:.2f}"]
            confidence = "possible"
            all_words = _WORD_RE.findall(memories[i].content.lower()) + _WORD_RE.findall(
                memories[j].content.lower()
            )
            markers = CHANGE_MARKERS.intersection(all_words)
            if markers:
                confidence = "likely"
                signals.append(f"change_markers={sorted(markers)}")
            if numbers[i] and numbers[j] and numbers[i] != numbers[j]:
                confidence = "likely"
                signals.append(f"numeric_mismatch={sorted(numbers[i])}vs{sorted(numbers[j])}")
            # weak-overlap pairs only qualify when a strong signal is present
            if confidence == "possible" and overlap < 2 * min_anchor_overlap:
                continue
            conflicts.append(
                Conflict(a=memories[i], b=memories[j], similarity=sim,
                         confidence=confidence, signals=signals)
            )
    # strongest tensions first: likely before possible, then by similarity
    conflicts.sort(key=lambda c: (c.confidence != "likely", -c.similarity))
    return conflicts


def estimate_tokens(text: str) -> int:
    """Cheap token estimate (~4 chars/token). Swap in a real tokenizer via pack()."""
    return len(text) // 4 + 1


def knapsack_pack(
    scores: Sequence[float],
    tokens: Sequence[int],
    budget: int,
) -> Tuple[List[int], int]:
    """Exact 0/1 knapsack: pick indices maximizing total score within token budget.

    Token granularity is coarsened so the DP stays fast (~1024 buckets); at
    that resolution the solution is exact. Returns (selected_indices, total_tokens).
    """
    n = len(scores)
    if n == 0 or budget <= 0:
        return [], 0
    gran = max(1, budget // 1024)
    b = budget // gran
    w = [max(1, t // gran) for t in tokens]
    # dp[cap] = (best_score, chosen_bitmask_as_set) — store parents for reconstruction
    dp = [0.0] + [0.0] * b
    choice = [[False] * (b + 1) for _ in range(n)]
    for i in range(n):
        wi, si = w[i], scores[i]
        if si <= 0:
            continue
        for cap in range(b, wi - 1, -1):
            cand = dp[cap - wi] + si
            if cand > dp[cap]:
                dp[cap] = cand
                choice[i][cap] = True
    # reconstruct
    selected: List[int] = []
    cap = max(range(b + 1), key=lambda c: dp[c])
    for i in range(n - 1, -1, -1):
        if choice[i][cap]:
            selected.append(i)
            cap -= w[i]
    selected.reverse()
    return selected, sum(tokens[i] for i in selected)
