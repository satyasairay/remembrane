import math

from remembrane import HashEmbedder, cosine_similarity


def test_deterministic():
    e = HashEmbedder()
    a = e.embed(["hello world"])[0]
    b = e.embed(["hello world"])[0]
    assert a == b


def test_normalized():
    e = HashEmbedder()
    v = e.embed(["some text to embed"])[0]
    assert math.isclose(math.sqrt(sum(x * x for x in v)), 1.0, rel_tol=1e-6)


def test_similar_texts_closer_than_dissimilar():
    e = HashEmbedder()
    base, similar, different = e.embed(
        [
            "the user prefers dark mode",
            "user preference: dark mode theme",
            "quarterly revenue grew by twelve percent",
        ]
    )
    assert cosine_similarity(base, similar) > cosine_similarity(base, different)


def test_empty_text():
    e = HashEmbedder()
    v = e.embed([""])[0]
    assert len(v) == e.dimension


def test_cosine_zero_vectors():
    assert cosine_similarity([0, 0], [0, 0]) == 0.0
