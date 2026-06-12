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
    return bm25_scores_tokens(tokenize(query), [tokenize(d) for d in documents], k1=k1, b=b)


def term_frequencies(tokens: List[str]) -> Dict[str, int]:
    """Per-document term-frequency map (cacheable by callers)."""
    tf: Dict[str, int] = {}
    for t in tokens:
        tf[t] = tf.get(t, 0) + 1
    return tf


def bm25_scores_tokens(
    q_terms: List[str],
    docs: Sequence[List[str]],
    k1: float = 1.5,
    b: float = 0.75,
    tfs: Sequence[Dict[str, int]] = None,
    doc_lens: Sequence[int] = None,
) -> List[float]:
    """BM25 over pre-tokenized documents. Pass cached `tfs`/`doc_lens` to skip
    per-query recounting (remembrane's store does this automatically)."""
    n = len(docs) if docs is not None else len(tfs)
    if n == 0 or not q_terms:
        return [0.0] * n
    if tfs is None:
        tfs = [term_frequencies(d) for d in docs]
    if doc_lens is None:
        doc_lens = [len(d) for d in docs]
    avgdl = sum(doc_lens) / n or 1.0

    uniq = set(q_terms)
    df = {term: sum(1 for tf in tfs if term in tf) for term in uniq}
    idf = {
        term: math.log(1 + (n - df[term] + 0.5) / (df[term] + 0.5))
        for term in uniq
    }

    scores = []
    k1p1 = k1 + 1
    for tf, dl in zip(tfs, doc_lens):
        s = 0.0
        denom_norm = k1 * (1 - b + b * dl / avgdl)
        for term in q_terms:
            f = tf.get(term, 0)
            if f:
                s += idf[term] * f * k1p1 / (f + denom_norm)
        scores.append(s)
    return scores


def normalized_bm25(query: str, documents: Sequence[str], **kw) -> List[float]:
    """BM25 scores scaled to [0, 1] by the max score in this result set."""
    return _normalize_scores(bm25_scores(query, documents, **kw))


def normalized_bm25_tokens(q_terms: List[str], docs: Sequence[List[str]] = None, **kw) -> List[float]:
    """Normalized BM25 over pre-tokenized documents (or cached tfs via kwargs)."""
    return _normalize_scores(bm25_scores_tokens(q_terms, docs, **kw))


def _normalize_scores(raw: List[float]) -> List[float]:
    peak = max(raw) if raw else 0.0
    if peak == 0:
        return raw
    return [s / peak for s in raw]
