"""Pluggable embedding backends.

The default :class:`HashEmbedder` is dependency-free and deterministic: it hashes
character n-grams into a fixed-size vector. It captures lexical similarity well
enough for small/medium memory stores and tests, and it never phones home.

For true semantic recall, plug in :class:`SentenceTransformerEmbedder` or
:class:`OpenAIEmbedder` (both optional extras).
"""
from __future__ import annotations

import hashlib
import math
import re
from typing import List, Protocol, Sequence


class Embedder(Protocol):
    """Anything with embed() and a dimension."""

    dimension: int

    def embed(self, texts: Sequence[str]) -> List[List[float]]: ...


_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _normalize(vec: List[float]) -> List[float]:
    norm = math.sqrt(sum(v * v for v in vec))
    if norm == 0:
        return vec
    return [v / norm for v in vec]


class HashEmbedder:
    """Deterministic, dependency-free embedder using hashed word + character n-grams.

    Not a neural embedding — similarity is lexical, not semantic. Good default for
    local-first use, testing, and stores under ~50k memories. Swap in a real
    embedder via ``MemoryStore(embedder=...)`` when you need semantics.
    """

    def __init__(self, dimension: int = 512, char_ngrams: tuple = (3, 4), word_weight: float = 2.0):
        self.dimension = dimension
        self.char_ngrams = char_ngrams
        self.word_weight = word_weight

    def _features(self, text: str) -> List[tuple]:
        text = text.lower()
        tokens = _TOKEN_RE.findall(text)
        feats: List[tuple] = [(f"w:{t}", self.word_weight) for t in tokens]
        joined = " ".join(tokens)
        for n in self.char_ngrams:
            for i in range(len(joined) - n + 1):
                feats.append((f"c{n}:{joined[i : i + n]}", 1.0))
        return feats

    def _embed_one(self, text: str) -> List[float]:
        vec = [0.0] * self.dimension
        for feat, weight in self._features(text):
            digest = hashlib.md5(feat.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:4], "little") % self.dimension
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vec[idx] += sign * weight
        return _normalize(vec)

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        return [self._embed_one(t) for t in texts]


class SentenceTransformerEmbedder:
    """Local neural embeddings via sentence-transformers (``pip install remembrane[sentence-transformers]``)."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "sentence-transformers is not installed. "
                "Run: pip install remembrane[sentence-transformers]"
            ) from exc
        self._model = SentenceTransformer(model_name)
        self.dimension = self._model.get_sentence_embedding_dimension()

    def embed(self, texts: Sequence[str]) -> List[List[float]]:  # pragma: no cover
        return [list(map(float, v)) for v in self._model.encode(list(texts), normalize_embeddings=True)]


class OpenAIEmbedder:
    """OpenAI API embeddings (``pip install remembrane[openai]``)."""

    _DIMS = {"text-embedding-3-small": 1536, "text-embedding-3-large": 3072}

    def __init__(self, model: str = "text-embedding-3-small", api_key: str | None = None):
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "openai is not installed. Run: pip install remembrane[openai]"
            ) from exc
        self._client = OpenAI(api_key=api_key) if api_key else OpenAI()
        self.model = model
        self.dimension = self._DIMS.get(model, 1536)

    def embed(self, texts: Sequence[str]) -> List[List[float]]:  # pragma: no cover
        resp = self._client.embeddings.create(model=self.model, input=list(texts))
        return [d.embedding for d in resp.data]


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity between two vectors (assumes comparable dimensions)."""
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)
