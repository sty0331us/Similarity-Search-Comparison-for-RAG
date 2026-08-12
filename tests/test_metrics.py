"""Metric behavior tests plus backend parity checks."""

from __future__ import annotations

import numpy as np
import pytest

from src.compare import backends_agree_on_argmax, compare_backends, compare_metrics
from src.data.samples import magnitude_disagreement_vectors
from src.metrics import Metric
from src.metrics.backends import manual, numpy_backend, scipy_backend, torch_backend


@pytest.fixture
def corpus_and_query() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(42)
    docs = rng.normal(size=(5, 16))
    query = rng.normal(size=(16,))
    return docs, query


def test_manual_cosine_matches_numpy(corpus_and_query):
    docs, query = corpus_and_query
    manual_scores = np.array([manual.cosine_similarity(doc, query) for doc in docs])
    numpy_scores = numpy_backend.cosine_similarity(docs, query)
    np.testing.assert_allclose(manual_scores, numpy_scores, rtol=1e-5, atol=1e-6)


def test_scipy_cosine_matches_numpy(corpus_and_query):
    docs, query = corpus_and_query
    np.testing.assert_allclose(
        scipy_backend.cosine_similarity(docs, query),
        numpy_backend.cosine_similarity(docs, query),
        rtol=1e-5,
        atol=1e-6,
    )


def test_torch_cosine_matches_numpy(corpus_and_query):
    docs, query = corpus_and_query
    torch_scores = torch_backend.cosine_similarity(docs, query).detach().cpu().numpy()
    np.testing.assert_allclose(
        torch_scores,
        numpy_backend.cosine_similarity(docs, query),
        rtol=1e-5,
        atol=1e-6,
    )


def test_l2_parity(corpus_and_query):
    docs, query = corpus_and_query
    manual_l2 = np.array([manual.l2_distance(doc, query) for doc in docs])
    np.testing.assert_allclose(manual_l2, numpy_backend.l2_distance(docs, query), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(
        scipy_backend.l2_distance(docs, query),
        numpy_backend.l2_distance(docs, query),
        rtol=1e-5,
        atol=1e-6,
    )
    torch_l2 = torch_backend.l2_distance(docs, query).detach().cpu().numpy()
    np.testing.assert_allclose(torch_l2, numpy_backend.l2_distance(docs, query), rtol=1e-5, atol=1e-6)


def test_compare_backends_agree(corpus_and_query):
    docs, query = corpus_and_query
    results = compare_backends(docs, query)
    assert backends_agree_on_argmax(results)
    for result in results:
        assert result.best_index == int(numpy_backend.argmax_cosine(docs, query))


def test_magnitude_demo_metrics_disagree():
    docs, query, _ = magnitude_disagreement_vectors()
    comparison = compare_metrics(docs, query)
    winners = {r.metric: r.best_index for r in comparison.results}

    assert winners[Metric.COSINE] == 0
    assert winners[Metric.INNER_PRODUCT] == 1
    assert winners[Metric.EUCLIDEAN] == 2
    assert not comparison.all_agree_on_best
