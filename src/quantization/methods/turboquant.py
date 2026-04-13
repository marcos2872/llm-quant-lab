"""
src/quantization/methods/turboquant.py
----------------------------------------
TurboQuant-style: rotação ortogonal Haar + Lloyd-Max + outlier channels em FP16.

Inspirado em:
  "TurboQuant: Efficient KV Cache Compression via Rotation and Codebook Quantization"
  (Xu et al., 2025 — arXiv:2504.19874)
"""

from __future__ import annotations

import numpy as np
import torch

# ── caches de sessão ──────────────────────────────────────────────────────────
# Chave da rotation_cache: (dim, seed)
# Chave da codebook_cache: (bits, group_size, model_id)
# model_id evita reutilização entre modelos com distribuições diferentes
_rotation_cache: dict[tuple[int, int], torch.Tensor] = {}
_codebook_cache: dict[tuple[int, int, str], torch.Tensor] = {}


def clear_caches() -> None:
    """Limpa os caches de rotação e codebook. Útil em sessões longas ou multirrun."""
    _rotation_cache.clear()
    _codebook_cache.clear()


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
    sample: torch.Tensor,
    n_levels: int,
    bits: int,
    group_size: int,
    model_id: str,
) -> torch.Tensor:
    """Retorna (e cacheia) codebook Lloyd-Max para (bits, group_size, model_id)."""
    key = (bits, group_size, model_id)
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
    flat: torch.Tensor,
    outlier_channels: int,
    head_dim: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Separa canais outlier (alta variância) dos canais normais.

    Garante que normal_idx seja dtype=int64 mesmo quando a lista fica vazia.
    """
    # abs().mean() detecta canais com alta magnitude média (mais correto que var,
    # que falha para outliers constantes com variância zero)
    channel_score = flat.abs().mean(dim=0)
    n_outliers = min(outlier_channels, head_dim)
    _, outlier_idx = channel_score.topk(n_outliers)
    outlier_idx, _ = outlier_idx.sort()
    outlier_set = set(outlier_idx.tolist())
    normal_idx = torch.tensor(
        [i for i in range(head_dim) if i not in outlier_set],
        dtype=torch.long,   # explícito: evita float32 quando a lista fica vazia
        device=device,
    )
    return outlier_idx, normal_idx, flat[:, outlier_idx], flat[:, normal_idx]


def _quantize_groups(
    rotated: torch.Tensor,
    group_size: int,
    bits: int,
    rotation_seed: int,
    model_id: str,
) -> tuple[torch.Tensor, dict]:
    """Aplica padding, agrupa e quantiza com codebook cacheado."""
    n_rows, dim_normal = rotated.shape
    pad = (group_size - dim_normal % group_size) % group_size if dim_normal > 0 else 0
    if pad > 0:
        rotated = torch.nn.functional.pad(rotated, (0, pad))
    n_groups = rotated.shape[-1] // group_size if dim_normal > 0 else 0
    grouped = rotated.reshape(n_rows, n_groups, group_size) if n_groups > 0 else rotated

    sample = grouped.reshape(-1, group_size) if n_groups > 0 else grouped
    if sample.shape[0] > 4096:
        perm = torch.randperm(sample.shape[0], generator=torch.Generator().manual_seed(rotation_seed))
        sample = sample[perm[:4096]]

    n_levels = 2 ** bits
    centroids = _get_codebook(sample.reshape(-1), n_levels, bits, group_size, model_id)
    centroids = centroids.to(rotated.device)

    dists = (grouped.unsqueeze(-1) - centroids).abs()
    q_indices = dists.argmin(dim=-1).to(torch.int16)
    return q_indices, {"n_rows": n_rows, "n_groups": n_groups, "group_size": group_size, "pad": pad}


def _degenerate_meta(
    flat: torch.Tensor,
    outlier_idx: torch.Tensor,
    outlier_vals: torch.Tensor,
    normal_idx: torch.Tensor,
    bits: int,
    group_size: int,
    rotation_seed: int,
    original_shape: tuple,
    original_dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, dict]:
    """Retorna estrutura vazia para o caso em que dim_normal == 0."""
    return torch.zeros(0, dtype=torch.int16, device=device), {
        "n_rows": flat.shape[0], "n_groups": 0, "group_size": group_size, "pad": 0,
        "centroids": torch.zeros(2 ** bits, device=device),
        "outlier_idx": outlier_idx, "outlier_vals": outlier_vals.to(original_dtype),
        "normal_idx": normal_idx, "rotation_seed": rotation_seed, "dim_normal": 0,
        "original_shape": original_shape, "original_dtype": str(original_dtype),
    }


def quantize_turboquant(
    tensor: torch.Tensor,
    bits: int = 4,
    group_size: int = 64,
    outlier_channels: int = 32,
    rotation_seed: int = 42,
    model_id: str = "",
) -> tuple[torch.Tensor, dict]:
    """Quantiza tensor KV com rotação ortogonal + Lloyd-Max."""
    original_shape, original_dtype = tensor.shape, tensor.dtype
    device, head_dim = tensor.device, tensor.shape[-1]

    flat = tensor.float().reshape(-1, head_dim)
    outlier_idx, normal_idx, outlier_vals, normal_vals = _split_channels(
        flat, outlier_channels, head_dim, device
    )
    dim_normal = normal_vals.shape[-1]

    if dim_normal == 0:
        return _degenerate_meta(
            flat, outlier_idx, outlier_vals, normal_idx,
            bits, group_size, rotation_seed, original_shape, original_dtype, device,
        )

    R = _get_rotation(dim_normal, rotation_seed, device)
    q_indices, group_meta = _quantize_groups(
        normal_vals @ R, group_size, bits, rotation_seed, model_id
    )
    meta = {
        **group_meta,
        "centroids": _codebook_cache[(bits, group_size, model_id)].to(device),
        "outlier_idx": outlier_idx, "outlier_vals": outlier_vals.to(original_dtype),
        "normal_idx": normal_idx, "rotation_seed": rotation_seed, "dim_normal": dim_normal,
        "original_shape": original_shape, "original_dtype": str(original_dtype),
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

    if meta["dim_normal"] > 0:
        R = _get_rotation(meta["dim_normal"], meta["rotation_seed"], device)
        unrotated = reconstructed @ R.T
    else:
        unrotated = reconstructed

    head_dim = meta["dim_normal"] + len(meta["outlier_idx"])
    full = torch.zeros(meta["n_rows"], head_dim, device=device, dtype=torch.float32)
    if meta["dim_normal"] > 0:
        full[:, meta["normal_idx"].to(device)] = unrotated
    full[:, meta["outlier_idx"].to(device)] = meta["outlier_vals"].float().to(device)
    return full.reshape(meta["original_shape"]).to(dtype)
