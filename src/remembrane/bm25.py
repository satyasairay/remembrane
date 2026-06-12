"""Pure-python BM25 (Okapi) for exact keyword scoring.

remembrane already scans every memory at recall time for exact vector
similarity, so BM25 comes nearly free: one extra pass over the same rows.
At agent-memory scale (thousands of rows) this stays sub-millisecond.
"""
from __future__ import annotations

import math
import re
from typing import Dict, List, Sequence

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


def bm25_scores(
    query: str,
    documents: Sequence[str],
    k1: float = 1.5,
    b: float = 0.75,
) -> List[float]:
    """BM25 score of `query` against each document. Returns raw scores (>= 0)."""
    q_terms = tokenize(query)
    docs = [tokenize(d) for d in documents]
    n = len(docs)
    if n == 0 or not q_terms:
        return [0.0] * n
    avgdl = sum(len(d) for d in docs) / n or 1.0

    df: Dict[str, int] = {}
    for term in set(q_terms):
        df[term] = sum(1 for d in docs if term in d)

    scores = []
    for d in docs:
        tf: Dict[str, int] = {}
        for t in d:
            tf[t] = tf.get(t, 0) + 1
        s = 0.0
        for term in q_terms:
            f = tf.get(term, 0)
            if f == 0:
                continue
            idf = math.log(1 + (n - df[term] + 0.5) / (df[term] + 0.5))
            s += idf * f * (k1 + 1) / (f + k1 * (1 - b + b * len(d) / avgdl))
        scores.append(s)
    return scores


def normalized_bm25(query: str, documents: Sequence[str], **kw) -> List[float]:
    """BM25 scores scaled to [0, 1] by the max score in this result set."""
    raw = bm25_scores(query, documents, **kw)
    peak = max(raw) if raw else 0.0
    if peak == 0:
        return raw
    return [s / peak for s in raw]
