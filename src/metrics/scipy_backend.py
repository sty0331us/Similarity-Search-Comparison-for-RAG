"""SciPy distance helpers for similarity search comparisons."""

from __future__ import annotations

import numpy as np
from scipy.spatial import distance


def l2_distance(docs: np.ndarray, query: np.ndarray) -> np.ndarray:
    d = np.asarray(docs, dtype=float)
    q = np.asarray(query, dtype=float).reshape(1, -1)
    return distance.cdist(d, q, metric="euclidean").ravel()


def cosine_distance(docs: np.ndarray, query: np.ndarray) -> np.ndarray:
    d = np.asarray(docs, dtype=float)
    q = np.asarray(query, dtype=float).reshape(1, -1)
    return distance.cdist(d, q, metric="cosine").ravel()


def cosine_similarity(docs: np.ndarray, query: np.ndarray) -> np.ndarray:
    return 1.0 - cosine_distance(docs, query)


def dot_product_similarity(docs: np.ndarray, query: np.ndarray) -> np.ndarray:
    """SciPy is distance-oriented; fall back to NumPy matmul for dot product."""
    d = np.asarray(docs, dtype=float)
    q = np.asarray(query, dtype=float).reshape(-1, 1)
    return (d @ q).ravel()


def argmax_cosine(docs: np.ndarray, query: np.ndarray) -> int:
    # Cosine *distance* is smaller when more similar.
    return int(np.argmin(cosine_distance(docs, query)))


def pairwise_cosine_matrix(vectors: np.ndarray) -> np.ndarray:
    return distance.cdist(vectors, vectors, metric="cosine")
