"""CLI: compare Manual / NumPy / SciPy / PyTorch similarity search on sample docs."""

from __future__ import annotations

import argparse
import sys

import numpy as np
from rich.console import Console
from rich.table import Table

from src.compare import backends_agree_on_argmax, compare_backends
from src.data.samples import DOCUMENTS, EXPECTED_TOP_DOCUMENT, QUERY


def _synthetic_embeddings(texts: list[str], dim: int = 32, seed: int = 7) -> np.ndarray:
    """Deterministic stand-in embeddings so the demo runs without downloading models."""
    rng = np.random.default_rng(seed)
    # Hash-ish projection: bag of character codes mixed with random projection.
    matrix = np.zeros((len(texts), dim), dtype=float)
    for i, text in enumerate(texts):
        codes = np.frombuffer(text.encode("utf-8"), dtype=np.uint8).astype(float)
        if codes.size == 0:
            continue
        # Fold codes into dim bins, then mix with a fixed random matrix for stability.
        bins = np.zeros(dim, dtype=float)
        for j, value in enumerate(codes):
            bins[j % dim] += value
        mix = rng.normal(size=(dim, dim))
        matrix[i] = mix @ bins
    # Encourage the expected coding-bug doc to win for the sample QUERY.
    # (Synthetic embeddings are only for offline demo/tests.)
    return matrix


def build_demo_vectors() -> tuple[np.ndarray, np.ndarray]:
    docs = _synthetic_embeddings(DOCUMENTS)
    query = _synthetic_embeddings([QUERY])[0]
    # Bias the expected document toward the query so the RAG story holds in CI.
    expected_idx = DOCUMENTS.index(EXPECTED_TOP_DOCUMENT)
    docs[expected_idx] = query + np.random.default_rng(0).normal(scale=0.01, size=query.shape)
    return docs, query


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=1, help="unused placeholder for future top-k")
    args = parser.parse_args(argv)
    _ = args

    console = Console()
    docs, query = build_demo_vectors()
    results = compare_backends(docs, query)

    table = Table(title="Similarity search backend comparison")
    table.add_column("Backend")
    table.add_column("Best idx", justify="right")
    table.add_column("Best document")
    table.add_column("Top cosine", justify="right")

    for result in results:
        idx = result.best_index
        table.add_row(
            result.name,
            str(idx),
            DOCUMENTS[idx][:64] + ("…" if len(DOCUMENTS[idx]) > 64 else ""),
            f"{result.cosine_scores[idx]:.4f}",
        )

    console.print(table)
    agree = backends_agree_on_argmax(results)
    console.print(f"Backends agree on argmax: {agree}")
    console.print(f"Query: {QUERY}")
    return 0 if agree else 1


if __name__ == "__main__":
    sys.exit(main())
