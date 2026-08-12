"""Similarity metric backends: manual, NumPy, SciPy, and PyTorch."""

from src.metrics.backends import manual, numpy_backend, scipy_backend, torch_backend

__all__ = ["manual", "numpy_backend", "scipy_backend", "torch_backend"]
