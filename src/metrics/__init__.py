"""Distance & similarity metrics for RAG retrieval.

Primary focus: Cosine Similarity, Inner Product (Dot Product), and Euclidean Distance.
Execution backends (manual / NumPy / SciPy / PyTorch) live under ``backends`` and are
secondary — they compute the same metrics with different implementations.
"""

from src.metrics.scoring import (
    l2_normalize,
    rank_indices,
    score,
    score_all_metrics,
    select_best,
)
from src.metrics.types import METRIC_SPECS, Metric, MetricSpec, RankingSense

__all__ = [
    "METRIC_SPECS",
    "Metric",
    "MetricSpec",
    "RankingSense",
    "l2_normalize",
    "rank_indices",
    "score",
    "score_all_metrics",
    "select_best",
]
