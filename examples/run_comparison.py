"""CLI: compare Cosine Similarity vs Inner Product vs Euclidean Distance."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow `python examples/run_comparison.py` without an editable install.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.compare import (
    backend_parity,
    compare_metrics,
    compare_normalized_vs_raw,
)
from src.data.samples import (
    DOCUMENTS,
    QUERY,
    magnitude_disagreement_vectors,
    rag_demo_vectors,
)
from src.metrics import METRIC_SPECS, Metric

def _score_table(
    title: str,
    comparison,
    labels: list[str],
) -> Table:
    table = Table(title=title, show_lines=False)
    table.add_column("Metric")
    table.add_column("Best idx", justify="right")
    table.add_column("Best item")
    table.add_column("Score @ best", justify="right")
    table.add_column("Ranking (best→worst)")

    for result in comparison.results:
        spec = METRIC_SPECS[result.metric]
        idx = result.best_index
        label = labels[idx]
        short = label if len(label) <= 56 else label[:53] + "…"
        score = result.scores[idx]
        order = " > ".join(str(i) for i in result.ranking_order.tolist())
        table.add_row(
            spec.display_name,
            str(idx),
            short,
            f"{score:.4f}",
            order,
        )
    return table


def _print_metric_primer(console: Console) -> None:
    table = Table(title="Metric primer", show_header=True)
    table.add_column("Metric")
    table.add_column("Formula")
    table.add_column("Rank by")
    table.add_column("Magnitude?")
    for metric in Metric:
        spec = METRIC_SPECS[metric]
        table.add_row(
            spec.display_name,
            spec.formula,
            "↑ higher" if "higher" in spec.ranking.value else "↓ lower",
            "yes" if spec.sensitive_to_magnitude else "no (direction only)",
        )
    console.print(table)


def run_disagreement_demo(console: Console) -> bool:
    docs, query, labels = magnitude_disagreement_vectors()
    raw, normalized = compare_normalized_vs_raw(docs, query)

    console.print(
        Panel.fit(
            "[bold]Core demo[/bold]: same query & corpus, three metrics — "
            "winners differ when vector magnitudes differ.",
            border_style="cyan",
        )
    )
    console.print(_score_table("Raw vectors (metrics can disagree)", raw, labels))
    console.print(
        f"All metrics agree on best (raw)? [bold]{raw.all_agree_on_best}[/bold]  "
        f"winners={raw.winners_by_metric}"
    )

    console.print(
        _score_table(
            "After L2-normalization (cosine ≡ inner product)",
            normalized,
            labels,
        )
    )
    cos = next(r for r in normalized.results if r.metric is Metric.COSINE)
    ip = next(r for r in normalized.results if r.metric is Metric.INNER_PRODUCT)
    cos_eq_ip = np.allclose(cos.scores, ip.scores, rtol=1e-5, atol=1e-6)
    console.print(
        f"Cosine scores ≡ Inner Product scores after normalize? [bold]{cos_eq_ip}[/bold]  "
        f"all agree on best? [bold]{normalized.all_agree_on_best}[/bold]"
    )
    return (not raw.all_agree_on_best) and cos_eq_ip


def run_rag_demo(console: Console) -> None:
    docs, query = rag_demo_vectors()
    comparison = compare_metrics(docs, query)
    labels = DOCUMENTS
    console.print(
        Panel.fit(
            f"[bold]RAG sample[/bold]\nQuery: {QUERY}",
            border_style="green",
        )
    )
    console.print(_score_table("Metric rankings on sample RAG embeddings", comparison, labels))
    console.print(
        f"All metrics agree on best doc? [bold]{comparison.all_agree_on_best}[/bold]"
    )


def run_backend_parity(console: Console, docs: np.ndarray, query: np.ndarray) -> bool:
    table = Table(title="Backend parity (secondary — same metric, different libraries)")
    table.add_column("Metric")
    table.add_column("manual", justify="right")
    table.add_column("numpy", justify="right")
    table.add_column("scipy", justify="right")
    table.add_column("torch", justify="right")
    table.add_column("Agree?")

    all_ok = True
    for metric in Metric:
        parity = backend_parity(docs, query, metric)
        best = parity.backend_best
        table.add_row(
            METRIC_SPECS[metric].display_name,
            str(best["manual"]),
            str(best["numpy"]),
            str(best["scipy"]),
            str(best["torch"]),
            "yes" if parity.backends_agree else "NO",
        )
        all_ok = all_ok and parity.backends_agree
    console.print(table)
    return all_ok


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compare Cosine Similarity, Inner Product, and Euclidean Distance "
            "for vector retrieval (RAG)."
        )
    )
    parser.add_argument(
        "--with-backends",
        action="store_true",
        help="Also show Manual/NumPy/SciPy/PyTorch parity (secondary)",
    )
    parser.add_argument(
        "--skip-rag",
        action="store_true",
        help="Skip the sample RAG document demo",
    )
    args = parser.parse_args(argv)

    console = Console()
    _print_metric_primer(console)
    console.print()

    ok = run_disagreement_demo(console)
    console.print()

    if not args.skip_rag:
        run_rag_demo(console)
        console.print()

    if args.with_backends:
        docs, query, _ = magnitude_disagreement_vectors()
        backends_ok = run_backend_parity(console, docs, query)
        ok = ok and backends_ok

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
