"""
src/quantization/methods/turboquant.py
----------------------------------------
TurboQuant-style: rotação ortogonal Haar (head_dim completo) + detecção de
outliers pós-rotação + Lloyd-Max + outlier channels em FP16.

Melhorias em relação à versão original:
  1. Rotação aplicada ao head_dim COMPLETO antes de separar outliers.
     → Detecção de outliers no espaço isotropizado (mais precisa).
     → Reduz outlier_channels necessários sem perda de qualidade.
  2. Codebook Lloyd-Max por camada (chave: bits + group_size + model_id + layer_idx).
     → Captura distribuições distintas entre as camadas de atenção.

Inspirado em:
  "TurboQuant: Efficient KV Cache Compression via Rotation and Codebook Quantization"
  (Xu et al., 2025 — arXiv:2504.19874)
"""

from __future__ import annotations

import numpy as np
import torch

# ── caches de sessão ──────────────────────────────────────────────────────────
# rotation_cache:  (dim, seed)
# codebook_cache:  (bits, group_size, model_id, layer_idx)
_rotation_cache: dict[tuple[int, int], torch.Tensor] = {}
_codebook_cache: dict[tuple[int, int, str, int], torch.Tensor] = {}


def clear_caches() -> None:
    """Limpa os caches de rotação e codebook. Útil em sessões longas ou multirun."""
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
    layer_idx: int = 0,
) -> torch.Tensor:
    """Retorna (e cacheia) codebook Lloyd-Max para (bits, group_size, model_id, layer_idx)."""
    key = (bits, group_size, model_id, layer_idx)
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
    Separa canais outlier (alta magnitude média) dos canais normais.

    Garante que normal_idx seja dtype=int64 mesmo quando a lista fica vazia.
    Deve ser chamada sobre o tensor JÁ ROTACIONADO para detecção precisa.
    """
    channel_score = flat.abs().mean(dim=0)
    n_outliers = min(outlier_channels, head_dim)
    _, outlier_idx = channel_score.topk(n_outliers)
    outlier_idx, _ = outlier_idx.sort()
    outlier_set = set(outlier_idx.tolist())
    normal_idx = torch.tensor(
        [i for i in range(head_dim) if i not in outlier_set],
        dtype=torch.long,
        device=device,
    )
    return outlier_idx, normal_idx, flat[:, outlier_idx], flat[:, normal_idx]


def _quantize_groups(
    values: torch.Tensor,
    group_size: int,
    bits: int,
    rotation_seed: int,
    model_id: str,
    layer_idx: int = 0,
) -> tuple[torch.Tensor, dict]:
    """Aplica padding, agrupa e quantiza com codebook Lloyd-Max por camada."""
    n_rows, dim = values.shape
    pad = (group_size - dim % group_size) % group_size if dim > 0 else 0
    if pad > 0:
        values = torch.nn.functional.pad(values, (0, pad))
    n_groups = values.shape[-1] // group_size if dim > 0 else 0
    grouped = values.reshape(n_rows, n_groups, group_size) if n_groups > 0 else values

    sample = grouped.reshape(-1, group_size) if n_groups > 0 else grouped
    if sample.shape[0] > 4096:
        perm = torch.randperm(
            sample.shape[0],
            generator=torch.Generator().manual_seed(rotation_seed),
        )
        sample = sample[perm[:4096]]

    n_levels = 2 ** bits
    centroids = _get_codebook(
        sample.reshape(-1), n_levels, bits, group_size, model_id, layer_idx
    )
    centroids = centroids.to(values.device)

    dists = (grouped.unsqueeze(-1) - centroids).abs()
    q_indices = dists.argmin(dim=-1).to(torch.int16)
    return q_indices, {"n_rows": n_rows, "n_groups": n_groups, "group_size": group_size, "pad": pad}


def _degenerate_meta(
    n_rows: int,
    outlier_idx: torch.Tensor,
    outlier_vals: torch.Tensor,
    normal_idx: torch.Tensor,
    bits: int,
    group_size: int,
    rotation_seed: int,
    original_shape: tuple,
    original_dtype: torch.dtype,
    device: torch.device,
    head_dim: int,
    layer_idx: int = 0,
) -> tuple[torch.Tensor, dict]:
    """Retorna estrutura vazia para o caso em que dim_normal == 0."""
    return torch.zeros(0, dtype=torch.int16, device=device), {
        "n_rows": n_rows, "n_groups": 0, "group_size": group_size, "pad": 0,
        "centroids": torch.zeros(2 ** bits, device=device),
        "outlier_idx": outlier_idx,
        "outlier_vals": outlier_vals.to(original_dtype),
        "normal_idx": normal_idx,
        "rotation_seed": rotation_seed,
        "head_dim": head_dim,
        "dim_normal": 0,
        "original_shape": original_shape,
        "original_dtype": str(original_dtype),
    }


def quantize_turboquant(
    tensor: torch.Tensor,
    bits: int = 4,
    group_size: int = 64,
    outlier_channels: int = 32,
    rotation_seed: int = 42,
    model_id: str = "",
    layer_idx: int = 0,
) -> tuple[torch.Tensor, dict]:
    """
    Quantiza tensor KV com rotação ortogonal completa + detecção de outliers pós-rotação.

    Pipeline:
      1. Rotaciona o head_dim COMPLETO → isotropiza a variância entre todos os canais.
      2. Detecta outliers NO ESPAÇO ROTACIONADO (distribuição mais uniforme → seleção precisa).
      3. Preserva canais outlier em FP16; quantiza os demais com codebook Lloyd-Max por camada.
    """
    original_shape, original_dtype = tensor.shape, tensor.dtype
    device, head_dim = tensor.device, tensor.shape[-1]

    flat = tensor.float().reshape(-1, head_dim)

    # 1. Rotação do head_dim completo — isotropiza variância antes da separação
    R = _get_rotation(head_dim, rotation_seed, device)
    rotated_full = flat @ R

    # 2. Detecta outliers no espaço rotacionado (mais representativo que o original)
    outlier_idx, normal_idx, outlier_vals, normal_vals = _split_channels(
        rotated_full, outlier_channels, head_dim, device
    )
    dim_normal = normal_vals.shape[-1]

    if dim_normal == 0:
        return _degenerate_meta(
            flat.shape[0], outlier_idx, outlier_vals, normal_idx,
            bits, group_size, rotation_seed, original_shape, original_dtype, device,
            head_dim=head_dim, layer_idx=layer_idx,
        )

    # 3. Quantiza canais normais — já rotacionados, sem segunda rotação
    q_indices, group_meta = _quantize_groups(
        normal_vals, group_size, bits, rotation_seed, model_id, layer_idx
    )
    meta = {
        **group_meta,
        "centroids": _codebook_cache[(bits, group_size, model_id, layer_idx)].to(device),
        "outlier_idx": outlier_idx,
        "outlier_vals": outlier_vals.to(original_dtype),
        "normal_idx": normal_idx,
        "rotation_seed": rotation_seed,
        "head_dim": head_dim,
        "dim_normal": dim_normal,
        "original_shape": original_shape,
        "original_dtype": str(original_dtype),
    }
    return q_indices, meta


def dequantize_turboquant(quantized: torch.Tensor, meta: dict) -> torch.Tensor:
    """
    Reconstrói tensor KV a partir do formato comprimido.

    Reverso de quantize_turboquant:
      1. Reconstrói canais normais a partir do codebook Lloyd-Max.
      2. Monta tensor completo no espaço rotacionado (normais + outliers FP16).
      3. Aplica rotação inversa (Rᵀ = R⁻¹ para matriz ortogonal).
    """
    dtype = getattr(torch, meta["original_dtype"].replace("torch.", ""))
    device = quantized.device

    # 1. Reconstrói canais normais quantizados
    centroids = meta["centroids"].to(device)
    reconstructed = centroids[quantized.long()].reshape(meta["n_rows"], -1).float()
    if meta["pad"] > 0:
        reconstructed = reconstructed[:, : -meta["pad"]]

    # 2. Monta tensor completo no espaço rotacionado
    head_dim = meta["head_dim"]
    full_rotated = torch.zeros(meta["n_rows"], head_dim, device=device, dtype=torch.float32)
    if meta["dim_normal"] > 0:
        full_rotated[:, meta["normal_idx"].to(device)] = reconstructed
    full_rotated[:, meta["outlier_idx"].to(device)] = meta["outlier_vals"].float().to(device)

    # 3. Rotação inversa sobre head_dim completo (Rᵀ = R⁻¹)
    R = _get_rotation(head_dim, meta["rotation_seed"], device)
    full = full_rotated @ R.T

    return full.reshape(meta["original_shape"]).to(dtype)
