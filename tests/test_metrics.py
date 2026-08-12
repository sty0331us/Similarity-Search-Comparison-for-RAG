"""Tests focused on metric behavior (Cosine / Inner Product / Euclidean)."""

from __future__ import annotations

import numpy as np
import pytest

from src.compare import (
    backend_parity,
    backends_agree_on_argmax,
    compare_backends,
    compare_metrics,
    compare_normalized_vs_raw,
)
from src.data.samples import magnitude_disagreement_vectors
from src.metrics import METRIC_SPECS, Metric, l2_normalize, score, select_best
from src.metrics.backends import manual, numpy_backend, scipy_backend, torch_backend


@pytest.fixture
def corpus_and_query() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(42)
    docs = rng.normal(size=(5, 16))
    query = rng.normal(size=(16,))
    return docs, query


def test_magnitude_demo_metrics_disagree():
    docs, query, _ = magnitude_disagreement_vectors()
    comparison = compare_metrics(docs, query)
    winners = {r.metric: r.best_index for r in comparison.results}

    assert winners[Metric.COSINE] == 0
    assert winners[Metric.INNER_PRODUCT] == 1
    assert winners[Metric.EUCLIDEAN] == 2
    assert not comparison.all_agree_on_best


def test_normalized_cosine_equals_inner_product():
    docs, query, _ = magnitude_disagreement_vectors()
    raw, normalized = compare_normalized_vs_raw(docs, query)
    assert not raw.all_agree_on_best

    cos = next(r for r in normalized.results if r.metric is Metric.COSINE)
    ip = next(r for r in normalized.results if r.metric is Metric.INNER_PRODUCT)
    np.testing.assert_allclose(cos.scores, ip.scores, rtol=1e-5, atol=1e-6)
    assert cos.best_index == ip.best_index


def test_unit_vectors_euclidean_agrees_with_cosine(corpus_and_query):
    docs, query = corpus_and_query
    docs_n = l2_normalize(docs, axis=1)
    query_n = l2_normalize(query)
    cos_scores = score(Metric.COSINE, docs_n, query_n)
    l2_scores = score(Metric.EUCLIDEAN, docs_n, query_n)
    # For unit vectors: ||a-b||^2 = 2 - 2 cos ⇒ rankings match.
    assert select_best(Metric.COSINE, cos_scores) == select_best(Metric.EUCLIDEAN, l2_scores)
    assert np.argsort(-cos_scores).tolist() == np.argsort(l2_scores).tolist()


def test_metric_specs_cover_all_metrics():
    assert set(METRIC_SPECS) == set(Metric)
    assert METRIC_SPECS[Metric.COSINE].sensitive_to_magnitude is False
    assert METRIC_SPECS[Metric.INNER_PRODUCT].sensitive_to_magnitude is True
    assert METRIC_SPECS[Metric.EUCLIDEAN].sensitive_to_magnitude is True


def test_manual_cosine_matches_numpy(corpus_and_query):
    docs, query = corpus_and_query
    manual_scores = np.array([manual.cosine_similarity(doc, query) for doc in docs])
    numpy_scores = numpy_backend.cosine_similarity(docs, query)
    np.testing.assert_allclose(manual_scores, numpy_scores, rtol=1e-5, atol=1e-6)


def test_backend_parity_all_metrics(corpus_and_query):
    docs, query = corpus_and_query
    for metric in Metric:
        parity = backend_parity(docs, query, metric)
        assert parity.backends_agree, parity.backend_best


def test_scipy_and_torch_match_numpy(corpus_and_query):
    docs, query = corpus_and_query
    np.testing.assert_allclose(
        scipy_backend.cosine_similarity(docs, query),
        numpy_backend.cosine_similarity(docs, query),
        rtol=1e-5,
        atol=1e-6,
    )
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
