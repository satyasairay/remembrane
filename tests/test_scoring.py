import math
import time

from remembrane import Memory, ScoringConfig
from remembrane.scoring import composite_score, recency_factor


def test_recency_decays_with_half_life():
    now = time.time()
    mem = Memory(content="x", created_at=now - 3600)
    assert recency_factor(mem, now=now, half_life=3600) == pytest_approx(0.5)


def pytest_approx(v, tol=1e-6):
    class _A:
        def __eq__(self, other):
            return math.isclose(other, v, rel_tol=tol)

    return _A()


def test_recency_uses_last_access():
    now = time.time()
    mem = Memory(content="x", created_at=now - 100_000, last_accessed_at=now)
    assert recency_factor(mem, now=now) > 0.999


def test_composite_score_weights_normalized():
    cfg = ScoringConfig(weight_similarity=7, weight_recency=1.5, weight_importance=1.5,
                        weight_usefulness=0)
    now = time.time()
    mem = Memory(content="x", importance=1.0, created_at=now)
    score, rec = composite_score(1.0, mem, cfg, now=now)
    assert math.isclose(score, 1.0, rel_tol=1e-9)
    assert math.isclose(rec, 1.0, rel_tol=1e-9)


def test_higher_importance_scores_higher():
    cfg = ScoringConfig()
    now = time.time()
    low = Memory(content="x", importance=0.1, created_at=now)
    high = Memory(content="x", importance=0.9, created_at=now)
    s_low, _ = composite_score(0.5, low, cfg, now=now)
    s_high, _ = composite_score(0.5, high, cfg, now=now)
    assert s_high > s_low
