"""Side-by-side comparison of Manual / NumPy / SciPy / PyTorch similarity search."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.metrics.backends import manual, numpy_backend, scipy_backend, torch_backend


@dataclass(frozen=True)
class BackendResult:
    name: str
    best_index: int
    cosine_scores: np.ndarray
    l2_distances: np.ndarray
    dot_scores: np.ndarray


def compare_backends(docs: np.ndarray, query: np.ndarray) -> list[BackendResult]:
    """Run the same corpus/query through every backend and collect scores."""
    docs = np.asarray(docs, dtype=float)
    query = np.asarray(query, dtype=float).reshape(-1)

    manual_scores = np.array(
        [manual.cosine_similarity(doc, query) for doc in docs], dtype=float
    )
    manual_l2 = np.array([manual.l2_distance(doc, query) for doc in docs], dtype=float)
    manual_dot = np.array([manual.dot_product(doc, query) for doc in docs], dtype=float)

    results = [
        BackendResult(
            name="manual",
            best_index=int(manual_scores.argmax()),
            cosine_scores=manual_scores,
            l2_distances=manual_l2,
            dot_scores=manual_dot,
        ),
        BackendResult(
            name="numpy",
            best_index=numpy_backend.argmax_cosine(docs, query),
            cosine_scores=numpy_backend.cosine_similarity(docs, query),
            l2_distances=numpy_backend.l2_distance(docs, query),
            dot_scores=numpy_backend.dot_product_similarity(docs, query),
        ),
        BackendResult(
            name="scipy",
            best_index=scipy_backend.argmax_cosine(docs, query),
            cosine_scores=scipy_backend.cosine_similarity(docs, query),
            l2_distances=scipy_backend.l2_distance(docs, query),
            dot_scores=scipy_backend.dot_product_similarity(docs, query),
        ),
        BackendResult(
            name="torch",
            best_index=torch_backend.argmax_cosine(docs, query),
            cosine_scores=torch_backend.cosine_similarity(docs, query).detach().cpu().numpy(),
            l2_distances=torch_backend.l2_distance(docs, query).detach().cpu().numpy(),
            dot_scores=torch_backend.dot_product_similarity(docs, query).detach().cpu().numpy(),
        ),
    ]
    return results


def backends_agree_on_argmax(results: list[BackendResult]) -> bool:
    winners = {result.best_index for result in results}
    return len(winners) == 1
