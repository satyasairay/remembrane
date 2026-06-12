"""Recall ranking: similarity x recency x importance x earned usefulness.

score = w_sim * similarity + w_rec * recency + w_imp * importance + w_use * usefulness_factor

recency = exp(-ln(2) * age / half_life)  -> halves every `half_life` seconds.
Reinforced memories (recalled or explicitly reinforced) age from their last
access instead of creation, mimicking spaced repetition.

usefulness_factor = sigmoid(usefulness): outcome feedback accumulated via
``MemoryStore.mark_useful()/mark_useless()``. A memory with no feedback sits at
the neutral 0.5; salience is *earned from task outcomes*, not guessed at write
time by an LLM.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Optional, Tuple

from .models import Memory

_LN2 = math.log(2)


def usefulness_factor(usefulness: float) -> float:
    """Squash the signed feedback accumulator into [0, 1]; 0 feedback -> 0.5."""
    return 1.0 / (1.0 + math.exp(-usefulness))


@dataclass
class ScoringConfig:
    weight_similarity: float = 0.65
    weight_recency: float = 0.15
    weight_importance: float = 0.10
    weight_usefulness: float = 0.10
    half_life_seconds: float = 7 * 24 * 3600.0  # one week
    keyword_weight: float = 0.35  # BM25 share of combined similarity in hybrid mode

    def __post_init__(self) -> None:
        total = (
            self.weight_similarity
            + self.weight_recency
            + self.weight_importance
            + self.weight_usefulness
        )
        if total <= 0:
            raise ValueError("scoring weights must sum to a positive number")
        # normalize so score stays in [0, 1]
        self.weight_similarity /= total
        self.weight_recency /= total
        self.weight_importance /= total
        self.weight_usefulness /= total
        if self.half_life_seconds <= 0:
            raise ValueError("half_life_seconds must be positive")
        if not 0.0 <= self.keyword_weight <= 1.0:
            raise ValueError("keyword_weight must be between 0.0 and 1.0")


def recency_factor(memory: Memory, now: Optional[float] = None, half_life: float = 7 * 24 * 3600.0) -> float:
    now = now if now is not None else time.time()
    anchor = memory.last_accessed_at or memory.created_at
    age = max(0.0, now - anchor)
    return math.exp(-_LN2 * age / half_life)


def composite_score(
    similarity: float,
    memory: Memory,
    config: ScoringConfig,
    now: Optional[float] = None,
) -> Tuple[float, float]:
    """Return (score, recency)."""
    rec = recency_factor(memory, now=now, half_life=config.half_life_seconds)
    score = (
        config.weight_similarity * similarity
        + config.weight_recency * rec
        + config.weight_importance * memory.importance
        + config.weight_usefulness * usefulness_factor(memory.usefulness)
    )
    return score, rec
