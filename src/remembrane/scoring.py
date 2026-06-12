"""Recall ranking: similarity x recency x importance.

score = w_sim * similarity + w_rec * recency + w_imp * importance

recency = exp(-ln(2) * age / half_life)  -> halves every `half_life` seconds.
Reinforced memories (recalled or explicitly reinforced) age from their last
access instead of creation, mimicking spaced repetition.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass

from .models import Memory

_LN2 = math.log(2)


@dataclass
class ScoringConfig:
    weight_similarity: float = 0.7
    weight_recency: float = 0.15
    weight_importance: float = 0.15
    half_life_seconds: float = 7 * 24 * 3600.0  # one week

    def __post_init__(self) -> None:
        total = self.weight_similarity + self.weight_recency + self.weight_importance
        if total <= 0:
            raise ValueError("scoring weights must sum to a positive number")
        # normalize so score stays in [0, 1]
        self.weight_similarity /= total
        self.weight_recency /= total
        self.weight_importance /= total
        if self.half_life_seconds <= 0:
            raise ValueError("half_life_seconds must be positive")


def recency_factor(memory: Memory, now: float | None = None, half_life: float = 7 * 24 * 3600.0) -> float:
    now = now if now is not None else time.time()
    anchor = memory.last_accessed_at or memory.created_at
    age = max(0.0, now - anchor)
    return math.exp(-_LN2 * age / half_life)


def composite_score(
    similarity: float,
    memory: Memory,
    config: ScoringConfig,
    now: float | None = None,
) -> tuple:
    """Return (score, recency)."""
    rec = recency_factor(memory, now=now, half_life=config.half_life_seconds)
    score = (
        config.weight_similarity * similarity
        + config.weight_recency * rec
        + config.weight_importance * memory.importance
    )
    return score, rec
