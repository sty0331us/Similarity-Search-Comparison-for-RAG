"""PyTorch tensor backends for similarity search (CPU or GPU)."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _as_float_tensor(data, device: torch.device | None = None) -> torch.Tensor:
    if isinstance(data, torch.Tensor):
        tensor = data.float()
    else:
        tensor = torch.as_tensor(data, dtype=torch.float32)
    if device is not None:
        tensor = tensor.to(device)
    return tensor


def l2_normalize(matrix, dim: int = -1, device: torch.device | None = None) -> torch.Tensor:
    return F.normalize(_as_float_tensor(matrix, device=device), p=2, dim=dim)


def l2_distance(docs, query, device: torch.device | None = None) -> torch.Tensor:
    d = _as_float_tensor(docs, device=device)
    q = _as_float_tensor(query, device=device).reshape(1, -1)
    return torch.cdist(d, q, p=2).ravel()


def dot_product_similarity(docs, query, device: torch.device | None = None) -> torch.Tensor:
    d = _as_float_tensor(docs, device=device)
    q = _as_float_tensor(query, device=device).reshape(-1, 1)
    return (d @ q).ravel()


def cosine_similarity(docs, query, device: torch.device | None = None) -> torch.Tensor:
    d = l2_normalize(docs, dim=1, device=device)
    q = l2_normalize(query, dim=-1, device=device).reshape(1, -1)
    return (d @ q.T).ravel()


def cosine_distance(docs, query, device: torch.device | None = None) -> torch.Tensor:
    return 1.0 - cosine_similarity(docs, query, device=device)


def argmax_cosine(docs, query, device: torch.device | None = None) -> int:
    scores = cosine_similarity(docs, query, device=device)
    return int(torch.argmax(scores).item())


def topk_cosine(docs, query, k: int = 1, device: torch.device | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    scores = cosine_similarity(docs, query, device=device)
    k = min(k, scores.numel())
    values, indices = torch.topk(scores, k=k)
    return indices, values
