"""
src/quantization/methods/turboquant.py
----------------------------------------
TurboQuant-style: rotação ortogonal Haar + Lloyd-Max no espaço rotacionado,
com outlier channels preservados em FP16.

Inspirado em:
  "TurboQuant: Efficient KV Cache Compression via Rotation and Codebook Quantization"
  (Xu et al., 2025 — arXiv:2504.19874)
"""

from __future__ import annotations

import numpy as np
import torch

# ── cache de artefatos reutilizáveis ──────────────────────────────────────────
_rotation_cache: dict[tuple[int, int], torch.Tensor] = {}
_codebook_cache: dict[tuple[int, int], torch.Tensor] = {}


def _get_rotation(dim: int, seed: int, device: torch.device) -> torch.Tensor:
    """Retorna (e cacheia) matriz ortogonal de Haar para (dim, seed)."""
    key = (dim, seed)
    if key not in _rotation_cache:
        rng = np.random.default_rng(seed)
        g = rng.standard_normal((dim, dim)).astype(np.float32)
        q, _ = np.linalg.qr(g)
        _rotation_cache[key] = torch.from_numpy(q)
    return _rotation_cache[key].to(device)


def _get_codebook(
    sample: torch.Tensor, n_levels: int, bits: int, group_size: int
) -> torch.Tensor:
    """Retorna (e cacheia) codebook Lloyd-Max para (bits, group_size)."""
    key = (bits, group_size)
    if key in _codebook_cache:
        return _codebook_cache[key]

    vals = sample.float().cpu()
    percentiles = torch.linspace(0, 100, n_levels + 2)[1:-1]
    sorted_vals = vals.sort().values
    idx = (percentiles / 100 * (len(sorted_vals) - 1)).long().clamp(0, len(sorted_vals) - 1)
    centroids = sorted_vals[idx].clone()

    for _ in range(20):
        dists = (vals.unsqueeze(1) - centroids.unsqueeze(0)).abs()
        assigns = dists.argmin(dim=1)
        new_centroids = torch.stack([
            vals[assigns == k].mean() if (assigns == k).any() else centroids[k]
            for k in range(n_levels)
        ])
        if (new_centroids - centroids).abs().max() < 1e-6:
            break
        centroids = new_centroids

    _codebook_cache[key] = centroids
    return centroids


def _split_channels(
    flat: torch.Tensor, outlier_channels: int, head_dim: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Separa canais outlier (alta variância) dos canais normais."""
    channel_var = flat.var(dim=0)
    _, outlier_idx = channel_var.topk(min(outlier_channels, head_dim))
    outlier_idx, _ = outlier_idx.sort()
    normal_idx = torch.tensor(
        [i for i in range(head_dim) if i not in outlier_idx.tolist()], device=device
    )
    return outlier_idx, normal_idx, flat[:, outlier_idx], flat[:, normal_idx]


def _quantize_groups(
    rotated: torch.Tensor, group_size: int, bits: int, rotation_seed: int
) -> tuple[torch.Tensor, dict]:
    """Aplica padding, agrupa e quantiza com codebook cacheado."""
    n_rows, dim_normal = rotated.shape
    pad = (group_size - dim_normal % group_size) % group_size
    if pad > 0:
        rotated = torch.nn.functional.pad(rotated, (0, pad))
    n_groups = rotated.shape[-1] // group_size
    grouped = rotated.reshape(n_rows, n_groups, group_size)

    sample = grouped.reshape(-1, group_size)
    if sample.shape[0] > 4096:
        perm = torch.randperm(sample.shape[0], generator=torch.Generator().manual_seed(rotation_seed))
        sample = sample[perm[:4096]]

    n_levels = 2 ** bits
    centroids = _get_codebook(sample.reshape(-1), n_levels, bits, group_size)
    centroids = centroids.to(rotated.device)

    dists = (grouped.unsqueeze(-1) - centroids).abs()
    q_indices = dists.argmin(dim=-1).to(torch.int16)
    return q_indices, {"n_rows": n_rows, "n_groups": n_groups, "group_size": group_size, "pad": pad}


def quantize_turboquant(
    tensor: torch.Tensor,
    bits: int = 4,
    group_size: int = 64,
    outlier_channels: int = 32,
    rotation_seed: int = 42,
) -> tuple[torch.Tensor, dict]:
    """Quantiza tensor KV com rotação ortogonal + Lloyd-Max."""
    original_shape = tensor.shape
    original_dtype = tensor.dtype
    device = tensor.device
    head_dim = tensor.shape[-1]

    flat = tensor.float().reshape(-1, head_dim)
    outlier_idx, normal_idx, outlier_vals, normal_vals = _split_channels(
        flat, outlier_channels, head_dim, device
    )

    dim_normal = normal_vals.shape[-1]
    R = _get_rotation(dim_normal, rotation_seed, device)
    rotated = normal_vals @ R

    q_indices, group_meta = _quantize_groups(rotated, group_size, bits, rotation_seed)

    meta = {
        **group_meta,
        "centroids": _codebook_cache[(bits, group_size)].to(device),
        "outlier_idx": outlier_idx,
        "outlier_vals": outlier_vals.to(original_dtype),
        "normal_idx": normal_idx,
        "rotation_seed": rotation_seed,
        "dim_normal": dim_normal,
        "original_shape": original_shape,
        "original_dtype": str(original_dtype),
    }
    return q_indices, meta


def dequantize_turboquant(quantized: torch.Tensor, meta: dict) -> torch.Tensor:
    """Reconstrói tensor aplicando rotação inversa e reinserindo outliers."""
    dtype = getattr(torch, meta["original_dtype"].replace("torch.", ""))
    device = quantized.device

    centroids = meta["centroids"].to(device)
    reconstructed = centroids[quantized.long()].reshape(meta["n_rows"], -1).float()
    if meta["pad"] > 0:
        reconstructed = reconstructed[:, : -meta["pad"]]

    R = _get_rotation(meta["dim_normal"], meta["rotation_seed"], device)
    unrotated = reconstructed @ R.T

    head_dim = meta["dim_normal"] + len(meta["outlier_idx"])
    full = torch.zeros(meta["n_rows"], head_dim, device=device, dtype=torch.float32)
    full[:, meta["normal_idx"].to(device)] = unrotated
    full[:, meta["outlier_idx"].to(device)] = meta["outlier_vals"].float().to(device)
    return full.reshape(meta["original_shape"]).to(dtype)
