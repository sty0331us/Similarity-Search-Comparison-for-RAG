"""Metric definitions: Cosine Similarity, Inner Product, and Euclidean Distance."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Metric(str, Enum):
    """The three similarity / distance metrics compared in this project."""

    COSINE = "cosine"
    INNER_PRODUCT = "inner_product"
    EUCLIDEAN = "euclidean"


class RankingSense(str, Enum):
    """Whether larger or smaller scores mean 'more similar'."""

    HIGHER_IS_BETTER = "higher_is_better"  # similarities
    LOWER_IS_BETTER = "lower_is_better"  # distances


@dataclass(frozen=True)
class MetricSpec:
    """Human-readable metadata for a metric (formulas, RAG use cases, caveats)."""

    metric: Metric
    display_name: str
    formula: str
    ranking: RankingSense
    sensitive_to_magnitude: bool
    summary: str
    typical_rag_use: str
    trade_off: str


METRIC_SPECS: dict[Metric, MetricSpec] = {
    Metric.COSINE: MetricSpec(
        metric=Metric.COSINE,
        display_name="Cosine Similarity",
        formula=r"(a · b) / (||a|| ||b||)",
        ranking=RankingSense.HIGHER_IS_BETTER,
        sensitive_to_magnitude=False,
        summary="Measures the angle between vectors (direction only).",
        typical_rag_use=(
            "Default for semantic text embeddings when you care about meaning "
            "alignment, not vector length."
        ),
        trade_off=(
            "Ignores magnitude. After L2-normalization, cosine ≡ inner product, "
            "so retrieval often collapses to a single matrix multiply."
        ),
    ),
    Metric.INNER_PRODUCT: MetricSpec(
        metric=Metric.INNER_PRODUCT,
        display_name="Inner Product (Dot Product)",
        formula=r"a · b",
        ranking=RankingSense.HIGHER_IS_BETTER,
        sensitive_to_magnitude=True,
        summary="Measures alignment weighted by magnitude.",
        typical_rag_use=(
            "Common when embeddings are already unit-normalized (then ≡ cosine), "
            "or when magnitude is intentional (e.g. confidence / length features)."
        ),
        trade_off=(
            "Longer vectors score higher even if the direction is only roughly aligned. "
            "Prefer cosine when length should not affect ranking."
        ),
    ),
    Metric.EUCLIDEAN: MetricSpec(
        metric=Metric.EUCLIDEAN,
        display_name="Euclidean Distance (L2)",
        formula=r"||a − b||₂",
        ranking=RankingSense.LOWER_IS_BETTER,
        sensitive_to_magnitude=True,
        summary="Straight-line distance between points in embedding space.",
        typical_rag_use=(
            "Nearest-neighbor style retrieval, or when you think in 'distance to "
            "the query' rather than 'similarity score'."
        ),
        trade_off=(
            "Sensitive to magnitude and absolute position. For unit-normalized "
            "vectors, ranking by Euclidean distance is monotonically related to "
            "cosine similarity (smaller L2 ↔ larger cosine)."
        ),
    ),
}


def best_index(scores, ranking: RankingSense) -> int:
    """Return the index of the best score given the metric's ranking sense."""
    import numpy as np

    arr = np.asarray(scores)
    if ranking is RankingSense.HIGHER_IS_BETTER:
        return int(np.argmax(arr))
    return int(np.argmin(arr))
