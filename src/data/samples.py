"""Shared sample documents, query, and vectors that expose metric disagreements."""

from __future__ import annotations

import numpy as np

DOCUMENTS: list[str] = [
    "Bugs introduced by the intern had to be squashed by the lead developer.",
    "Bugs found by the quality assurance engineer were difficult to debug.",
    "Bugs are common throughout the warm summer months, according to the entomologist.",
    "Bugs, in particular spiders, are extensively studied by arachnologists.",
]

QUERY: str = (
    "Who is responsible for a coding project and fixing others' mistakes?"
)

# Expected top hit for the QUERY under typical semantic embeddings (cosine).
EXPECTED_TOP_DOCUMENT: str = DOCUMENTS[0]


def magnitude_disagreement_vectors() -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Hand-crafted 2D vectors where the three metrics each pick a different winner.

    Query at (1, 0). Geometry is chosen so ranking disagrees:

    - Doc A: short, tightly aligned → **Cosine** wins
    - Doc B: long, roughly aligned → **Inner Product** wins
    - Doc C: nearest point in space → **Euclidean** wins

    Labels describe the geometric role, not the RAG text corpus.
    """
    query = np.array([1.0, 0.0], dtype=float)
    docs = np.array(
        [
            [0.8, 0.05],  # A: best angle, modest magnitude & distance
            [4.0, 1.0],   # B: largest dot product
            [1.05, 0.1],  # C: smallest Euclidean distance to the query
        ],
        dtype=float,
    )
    labels = [
        "A: short & tightly aligned (favors cosine)",
        "B: long & roughly aligned (favors inner product)",
        "C: nearby in space (favors Euclidean)",
    ]
    return docs, query, labels

def rag_demo_vectors() -> tuple[np.ndarray, np.ndarray]:
    """Deterministic stand-in embeddings for the sample RAG documents/query.

    Biases the expected coding-bug document toward the query so the narrative
    holds offline (no model download). Useful for cosine-style retrieval demos.
    """
    dim = 32
    matrix = np.vstack([_embed_one(text, dim=dim, seed=10 + i) for i, text in enumerate(DOCUMENTS)])
    query = _embed_one(QUERY, dim=dim, seed=99)
    expected_idx = DOCUMENTS.index(EXPECTED_TOP_DOCUMENT)
    matrix[expected_idx] = query + np.random.default_rng(0).normal(scale=0.01, size=query.shape)
    return matrix, query


def _embed_one(text: str, dim: int = 32, seed: int = 0) -> np.ndarray:
    """Stable hashed bag-of-bytes embedding (no BLAS matmul — keeps demos quiet)."""
    rng = np.random.default_rng(seed)
    vec = rng.normal(size=dim)
    for idx, byte in enumerate(text.encode("utf-8")):
        vec[idx % dim] += (byte / 255.0) * 0.05
    return vec.astype(float)
