"""Compare Cosine / Inner Product / Euclidean — with optional backend parity checks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.metrics import METRIC_SPECS, Metric, l2_normalize, score_all_metrics, select_best
from src.metrics.backends import manual, numpy_backend, scipy_backend, torch_backend


@dataclass(frozen=True)
class MetricResult:
    """Scores and ranking for one metric on a fixed corpus/query."""

    metric: Metric
    display_name: str
    scores: np.ndarray
    best_index: int
    ranking_order: np.ndarray  # indices best → worst


@dataclass(frozen=True)
class MetricComparison:
    """Side-by-side metric results, plus agreement analysis."""

    results: list[MetricResult]
    all_agree_on_best: bool
    winners_by_metric: dict[str, int]


@dataclass(frozen=True)
class BackendParityResult:
    """Secondary check: do execution backends agree on a given metric?"""

    metric: Metric
    backend_best: dict[str, int]
    backends_agree: bool


def compare_metrics(docs: np.ndarray, query: np.ndarray) -> MetricComparison:
    """Primary comparison: Cosine vs Inner Product vs Euclidean on the same inputs."""
    docs = np.asarray(docs, dtype=float)
    query = np.asarray(query, dtype=float).reshape(-1)

    scored = score_all_metrics(docs, query)
    results: list[MetricResult] = []
    winners: dict[str, int] = {}

    for metric, scores in scored.items():
        spec = METRIC_SPECS[metric]
        best = select_best(metric, scores)
        if spec.ranking.value == "higher_is_better":
            order = np.argsort(-scores)
        else:
            order = np.argsort(scores)
        results.append(
            MetricResult(
                metric=metric,
                display_name=spec.display_name,
                scores=scores,
                best_index=best,
                ranking_order=order,
            )
        )
        winners[metric.value] = best

    all_agree = len(set(winners.values())) == 1
    return MetricComparison(
        results=results,
        all_agree_on_best=all_agree,
        winners_by_metric=winners,
    )


def compare_normalized_vs_raw(
    docs: np.ndarray,
    query: np.ndarray,
) -> tuple[MetricComparison, MetricComparison]:
    """Show how magnitude affects metrics: raw vectors vs L2-normalized vectors.

    After unit-normalization, Cosine and Inner Product produce identical rankings
    (and identical scores). Euclidean ranking becomes monotonically related to cosine.
    """
    docs = np.asarray(docs, dtype=float)
    query = np.asarray(query, dtype=float).reshape(-1)
    raw = compare_metrics(docs, query)
    normalized = compare_metrics(l2_normalize(docs, axis=1), l2_normalize(query))
    return raw, normalized


def backend_parity(docs: np.ndarray, query: np.ndarray, metric: Metric) -> BackendParityResult:
    """Secondary: verify Manual / NumPy / SciPy / PyTorch agree for one metric."""
    docs = np.asarray(docs, dtype=float)
    query = np.asarray(query, dtype=float).reshape(-1)

    if metric is Metric.COSINE:
        manual_scores = np.array([manual.cosine_similarity(doc, query) for doc in docs])
        scores = {
            "manual": manual_scores,
            "numpy": numpy_backend.cosine_similarity(docs, query),
            "scipy": scipy_backend.cosine_similarity(docs, query),
            "torch": torch_backend.cosine_similarity(docs, query).detach().cpu().numpy(),
        }
        best = {name: int(np.argmax(s)) for name, s in scores.items()}
    elif metric is Metric.INNER_PRODUCT:
        manual_scores = np.array([manual.dot_product(doc, query) for doc in docs])
        scores = {
            "manual": manual_scores,
            "numpy": numpy_backend.dot_product_similarity(docs, query),
            "scipy": scipy_backend.dot_product_similarity(docs, query),
            "torch": torch_backend.dot_product_similarity(docs, query).detach().cpu().numpy(),
        }
        best = {name: int(np.argmax(s)) for name, s in scores.items()}
    elif metric is Metric.EUCLIDEAN:
        manual_scores = np.array([manual.l2_distance(doc, query) for doc in docs])
        scores = {
            "manual": manual_scores,
            "numpy": numpy_backend.l2_distance(docs, query),
            "scipy": scipy_backend.l2_distance(docs, query),
            "torch": torch_backend.l2_distance(docs, query).detach().cpu().numpy(),
        }
        best = {name: int(np.argmin(s)) for name, s in scores.items()}
    else:
        raise ValueError(f"Unknown metric: {metric}")

    return BackendParityResult(
        metric=metric,
        backend_best=best,
        backends_agree=len(set(best.values())) == 1,
    )


# Backwards-compatible aliases used by older tests / scripts
@dataclass(frozen=True)
class BackendResult:
    name: str
    best_index: int
    cosine_scores: np.ndarray
    l2_distances: np.ndarray
    dot_scores: np.ndarray


def compare_backends(docs: np.ndarray, query: np.ndarray) -> list[BackendResult]:
    """Secondary helper: score all three metrics with every execution backend."""
    docs = np.asarray(docs, dtype=float)
    query = np.asarray(query, dtype=float).reshape(-1)

    manual_cos = np.array([manual.cosine_similarity(doc, query) for doc in docs], dtype=float)
    manual_l2 = np.array([manual.l2_distance(doc, query) for doc in docs], dtype=float)
    manual_dot = np.array([manual.dot_product(doc, query) for doc in docs], dtype=float)

    return [
        BackendResult(
            name="manual",
            best_index=int(manual_cos.argmax()),
            cosine_scores=manual_cos,
            l2_distances=manual_l2,
            dot_scores=manual_dot,
        ),
        BackendResult(
            name="numpy",
            best_index=int(np.argmax(numpy_backend.cosine_similarity(docs, query))),
            cosine_scores=numpy_backend.cosine_similarity(docs, query),
            l2_distances=numpy_backend.l2_distance(docs, query),
            dot_scores=numpy_backend.dot_product_similarity(docs, query),
        ),
        BackendResult(
            name="scipy",
            best_index=int(np.argmax(scipy_backend.cosine_similarity(docs, query))),
            cosine_scores=scipy_backend.cosine_similarity(docs, query),
            l2_distances=scipy_backend.l2_distance(docs, query),
            dot_scores=scipy_backend.dot_product_similarity(docs, query),
        ),
        BackendResult(
            name="torch",
            best_index=int(
                torch_backend.cosine_similarity(docs, query).detach().cpu().numpy().argmax()
            ),
            cosine_scores=torch_backend.cosine_similarity(docs, query).detach().cpu().numpy(),
            l2_distances=torch_backend.l2_distance(docs, query).detach().cpu().numpy(),
            dot_scores=torch_backend.dot_product_similarity(docs, query)
            .detach()
            .cpu()
            .numpy(),
        ),
    ]


def backends_agree_on_argmax(results: list[BackendResult]) -> bool:
    return len({result.best_index for result in results}) == 1


def metrics_agree_on_best(comparison: MetricComparison) -> bool:
    return comparison.all_agree_on_best
