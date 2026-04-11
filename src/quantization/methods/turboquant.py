"""
src/quantization/methods/turboquant.py
----------------------------------------
TurboQuant-style: rotação ortogonal aleatória (Haar via QR) + Lloyd-Max
no espaço rotacionado, com outlier channels preservados em FP16.

Inspirado em:
  "TurboQuant: Efficient KV Cache Compression via Rotation and Codebook Quantization"
  (Xu et al., 2025 — arXiv:2504.19874)

Fluxo de quantização:
  1. Identificar os `outlier_channels` com maior variância → preservar em FP16
  2. Aplicar rotação ortogonal R nos demais canais
  3. Quantizar com codebook Lloyd-Max 1D (bins calculados via k-means 1D em amostra)
  4. Armazenar (índices_codebook, centroids, outliers, metadata)

Fluxo de dequantização:
  1. Reconstruir tensor via centroids
  2. Aplicar rotação inversa R^T
  3. Reinserir outliers nas posições originais
"""

from __future__ import annotations

import numpy as np
import torch

# ── rotação ortogonal ─────────────────────────────────────────────────────────

def _random_orthogonal(dim: int, seed: int = 42) -> torch.Tensor:
    """Gera matriz ortogonal de Haar via QR de matriz gaussiana."""
    rng = np.random.default_rng(seed)
    g = rng.standard_normal((dim, dim)).astype(np.float32)
    q, _ = np.linalg.qr(g)
    return torch.from_numpy(q)


# cache de matrizes de rotação por (dim, seed)
_rotation_cache: dict[tuple[int, int], torch.Tensor] = {}


def _get_rotation(dim: int, seed: int, device: torch.device) -> torch.Tensor:
    key = (dim, seed)
    if key not in _rotation_cache:
        _rotation_cache[key] = _random_orthogonal(dim, seed)
    return _rotation_cache[key].to(device)


# ── Lloyd-Max simplificado (k-means 1D) ───────────────────────────────────────

def _lloyd_max_codebook(data_flat: torch.Tensor, n_bins: int) -> torch.Tensor:
    """Treina codebook 1D com k-means simples em data_flat (amostra)."""
    vals = data_flat.float().cpu()
    # inicializa centroids com percentis uniformes
    percentiles = torch.linspace(0, 100, n_bins + 2)[1:-1]
    sorted_vals = vals.sort().values
    idx = (percentiles / 100 * (len(sorted_vals) - 1)).long().clamp(0, len(sorted_vals) - 1)
    centroids = sorted_vals[idx].clone()

    for _ in range(20):  # iterações k-means
        dists = (vals.unsqueeze(1) - centroids.unsqueeze(0)).abs()
        assigns = dists.argmin(dim=1)
        new_centroids = torch.stack([
            vals[assigns == k].mean() if (assigns == k).any() else centroids[k]
            for k in range(n_bins)
        ])
        if (new_centroids - centroids).abs().max() < 1e-6:
            break
        centroids = new_centroids

    return centroids


# ── API pública ────────────────────────────────────────────────────────────────

def quantize_turboquant(
    tensor: torch.Tensor,
    bits: int = 4,
    group_size: int = 64,
    outlier_channels: int = 32,
    rotation_seed: int = 42,
) -> tuple[torch.Tensor, dict]:
    """
    Quantiza tensor KV com rotação ortogonal + Lloyd-Max.

    tensor: shape (batch, heads, seq_len, head_dim)
    """
    original_shape = tensor.shape
    original_dtype = tensor.dtype
    device = tensor.device
    n_levels = 2 ** bits
    head_dim = tensor.shape[-1]

    # achata para (N, head_dim)
    flat = tensor.float().reshape(-1, head_dim)

    # ── 1. identificar outlier channels ──────────────────────────────────────
    channel_var = flat.var(dim=0)
    _, outlier_idx = channel_var.topk(min(outlier_channels, head_dim))
    outlier_idx, _ = outlier_idx.sort()
    normal_idx = torch.tensor(
        [i for i in range(head_dim) if i not in outlier_idx.tolist()],
        device=device,
    )

    outlier_vals = flat[:, outlier_idx]        # FP16 preservado
    normal_vals = flat[:, normal_idx]          # será rotacionado e quantizado

    # ── 2. rotação ortogonal nos canais normais ───────────────────────────────
    dim_normal = normal_vals.shape[-1]
    R = _get_rotation(dim_normal, rotation_seed, device)
    rotated = normal_vals @ R                  # (N, dim_normal)

    # ── 3. codebook Lloyd-Max por grupo ──────────────────────────────────────
    n_rows = rotated.shape[0]
    pad = (group_size - dim_normal % group_size) % group_size
    if pad > 0:
        rotated = torch.nn.functional.pad(rotated, (0, pad))
    n_groups = rotated.shape[-1] // group_size
    grouped = rotated.reshape(n_rows, n_groups, group_size)

    # treina codebook em amostra (máx 4096 vetores para performance)
    sample = grouped.reshape(-1, group_size)
    if sample.shape[0] > 4096:
        idx_sample = torch.randperm(sample.shape[0], generator=torch.Generator().manual_seed(rotation_seed))[:4096]
        sample = sample[idx_sample]
    centroids = _lloyd_max_codebook(sample.reshape(-1), n_levels).to(device)

    # atribui cada valor ao centroid mais próximo
    dists = (grouped.unsqueeze(-1) - centroids).abs()  # (..., group_size, n_levels)
    q_indices = dists.argmin(dim=-1).to(torch.int16)   # (n_rows, n_groups, group_size)

    meta = {
        "centroids": centroids,
        "outlier_idx": outlier_idx,
        "outlier_vals": outlier_vals.to(original_dtype),
        "normal_idx": normal_idx,
        "rotation_seed": rotation_seed,
        "dim_normal": dim_normal,
        "n_rows": n_rows,
        "n_groups": n_groups,
        "group_size": group_size,
        "pad": pad,
        "original_shape": original_shape,
        "original_dtype": str(original_dtype),
    }
    return q_indices, meta


def dequantize_turboquant(
    quantized: torch.Tensor,
    meta: dict,
) -> torch.Tensor:
    """Reconstrói tensor float aplicando rotação inversa e reinserindo outliers."""
    dtype = getattr(torch, meta["original_dtype"].replace("torch.", ""))
    device = quantized.device

    centroids = meta["centroids"].to(device)
    reconstructed = centroids[quantized.long()]        # (n_rows, n_groups, group_size)
    reconstructed = reconstructed.reshape(meta["n_rows"], -1).float()

    if meta["pad"] > 0:
        reconstructed = reconstructed[:, : -meta["pad"]]

    # ── rotação inversa (R^T) ─────────────────────────────────────────────────
    R = _get_rotation(meta["dim_normal"], meta["rotation_seed"], device)
    unrotated = reconstructed @ R.T                    # (n_rows, dim_normal)

    # ── reinserir outliers ────────────────────────────────────────────────────
    head_dim = meta["dim_normal"] + len(meta["outlier_idx"])
    full = torch.zeros(meta["n_rows"], head_dim, device=device, dtype=torch.float32)
    full[:, meta["normal_idx"].to(device)] = unrotated
    full[:, meta["outlier_idx"].to(device)] = meta["outlier_vals"].float().to(device)

    return full.reshape(meta["original_shape"]).to(dtype)
