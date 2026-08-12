"""Execution backends for computing metrics (secondary to metric choice).

Prefer the metric-level API in ``src.metrics``. Use these modules when you need
a specific implementation style (manual loops, NumPy, SciPy, or PyTorch).
"""

from . import manual, numpy_backend, scipy_backend, torch_backend

__all__ = ["manual", "numpy_backend", "scipy_backend", "torch_backend"]
