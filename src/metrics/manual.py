"""Manual (explicit-loop) similarity and distance helpers for teaching the math."""

from __future__ import annotations

import math
from typing import Sequence


def l2_normalize(vector: Sequence[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vector))
    if norm == 0.0:
        raise ValueError("Cannot normalize a zero vector")
    return [x / norm for x in vector]


def l2_distance(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b):
        raise ValueError("Vectors must share the same dimension")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def dot_product(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b):
        raise ValueError("Vectors must share the same dimension")
    return sum(x * y for x, y in zip(a, b))


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    na = l2_normalize(a)
    nb = l2_normalize(b)
    return dot_product(na, nb)


def cosine_distance(a: Sequence[float], b: Sequence[float]) -> float:
    return 1.0 - cosine_similarity(a, b)


def rank_by_cosine(documents_emb: Sequence[Sequence[float]], query_emb: Sequence[float]) -> list[tuple[int, float]]:
    """Return (index, cosine_similarity) pairs sorted descending."""
    scored = [
        (idx, cosine_similarity(doc, query_emb))
        for idx, doc in enumerate(documents_emb)
    ]
    return sorted(scored, key=lambda item: item[1], reverse=True)


def argmax_cosine(documents_emb: Sequence[Sequence[float]], query_emb: Sequence[float]) -> int:
    return rank_by_cosine(documents_emb, query_emb)[0][0]
