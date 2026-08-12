"""NumPy backends for L2, dot product, and cosine similarity search."""

from __future__ import annotations

import numpy as np


def l2_normalize(matrix: np.ndarray, axis: int = -1) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=axis, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return matrix / norms


def l2_distance(docs: np.ndarray, query: np.ndarray) -> np.ndarray:
    """Pairwise Euclidean distance between each doc row and the query vector."""
    q = np.asarray(query, dtype=float).reshape(1, -1)
    d = np.asarray(docs, dtype=float)
    return np.linalg.norm(d - q, axis=1)


def dot_product_similarity(docs: np.ndarray, query: np.ndarray) -> np.ndarray:
    d = np.asarray(docs, dtype=float)
    q = np.asarray(query, dtype=float).reshape(-1, 1)
    return (d @ q).ravel()


def cosine_similarity(docs: np.ndarray, query: np.ndarray) -> np.ndarray:
    d = l2_normalize(np.asarray(docs, dtype=float), axis=1)
    q = l2_normalize(np.asarray(query, dtype=float).reshape(1, -1), axis=1)
    return (d @ q.T).ravel()


def cosine_distance(docs: np.ndarray, query: np.ndarray) -> np.ndarray:
    return 1.0 - cosine_similarity(docs, query)


def argmax_cosine(docs: np.ndarray, query: np.ndarray) -> int:
    return int(np.argmax(cosine_similarity(docs, query)))


def topk_cosine(docs: np.ndarray, query: np.ndarray, k: int = 1) -> tuple[np.ndarray, np.ndarray]:
    scores = cosine_similarity(docs, query)
    k = min(k, scores.shape[0])
    idx = np.argpartition(-scores, kth=k - 1)[:k]
    order = np.argsort(-scores[idx])
    ranked = idx[order]
    return ranked, scores[ranked]
