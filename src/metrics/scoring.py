"""Score a corpus against a query with each metric (NumPy reference backend)."""

from __future__ import annotations

import numpy as np

from src.metrics.backends import numpy_backend
from src.metrics.types import METRIC_SPECS, Metric, RankingSense, best_index


def score(
    metric: Metric,
    docs: np.ndarray,
    query: np.ndarray,
) -> np.ndarray:
    """Return per-document scores for ``metric`` (NumPy implementation)."""
    docs = np.asarray(docs, dtype=float)
    query = np.asarray(query, dtype=float).reshape(-1)

    if metric is Metric.COSINE:
        return numpy_backend.cosine_similarity(docs, query)
    if metric is Metric.INNER_PRODUCT:
        return numpy_backend.dot_product_similarity(docs, query)
    if metric is Metric.EUCLIDEAN:
        return numpy_backend.l2_distance(docs, query)
    raise ValueError(f"Unknown metric: {metric}")


def rank_indices(metric: Metric, scores: np.ndarray) -> np.ndarray:
    """Return document indices sorted best → worst for ``metric``."""
    ranking = METRIC_SPECS[metric].ranking
    scores = np.asarray(scores)
    if ranking is RankingSense.HIGHER_IS_BETTER:
        return np.argsort(-scores)
    return np.argsort(scores)


def select_best(metric: Metric, scores: np.ndarray) -> int:
    """Return the single best document index for ``metric``."""
    return best_index(scores, METRIC_SPECS[metric].ranking)


def score_all_metrics(
    docs: np.ndarray,
    query: np.ndarray,
) -> dict[Metric, np.ndarray]:
    """Score every compared metric on the same corpus/query."""
    return {metric: score(metric, docs, query) for metric in Metric}


def l2_normalize(vectors: np.ndarray, axis: int = -1) -> np.ndarray:
    """Row-wise (or axis-wise) L2 normalization — makes cosine ≡ inner product."""
    return numpy_backend.l2_normalize(np.asarray(vectors, dtype=float), axis=axis)
