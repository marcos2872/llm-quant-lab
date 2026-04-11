"""
src/quantization/methods/kivi.py
----------------------------------
Quantização estilo KIVI: quantização por grupo ao longo da dimensão de canal.

Método (inspirado em "KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache"):
  - Divide os canais em grupos de tamanho group_size
  - Calcula escala e zero-point por grupo
  - Quantiza cada grupo independentemente (assimétrico)

Referência: https://arxiv.org/abs/2402.02750
"""

from __future__ import annotations

import torch


def quantize_kivi(
    tensor: torch.Tensor,
    bits: int = 4,
    group_size: int = 64,
) -> tuple[torch.Tensor, dict]:
    """
    Quantização por grupo ao longo da última dimensão (canal/head_dim).

    tensor: shape [..., seq_len, head_dim]
    Retorna (tensor_quantizado, metadata).
    """
    original_shape = tensor.shape
    original_dtype = tensor.dtype
    n_levels = 2 ** bits

    # agrupar ao longo da última dimensão
    flat = tensor.float().reshape(-1, tensor.shape[-1])
    n_rows, n_cols = flat.shape

    # padding para múltiplo de group_size
    pad = (group_size - n_cols % group_size) % group_size
    if pad > 0:
        flat = torch.nn.functional.pad(flat, (0, pad))
    n_groups = flat.shape[-1] // group_size
    grouped = flat.reshape(n_rows, n_groups, group_size)

    g_min = grouped.min(dim=-1, keepdim=True).values
    g_max = grouped.max(dim=-1, keepdim=True).values
    scale = (g_max - g_min) / (n_levels - 1)
    scale = scale.clamp(min=1e-8)

    q = ((grouped - g_min) / scale).round().clamp(0, n_levels - 1)
    dtype = torch.int8 if bits <= 8 else torch.int16
    q = q.to(dtype)

    meta = {
        "scale": scale,
        "zero": g_min,
        "original_shape": original_shape,
        "original_dtype": str(original_dtype),
        "n_rows": n_rows,
        "n_groups": n_groups,
        "group_size": group_size,
        "pad": pad,
    }
    return q, meta


def dequantize_kivi(
    quantized: torch.Tensor,
    meta: dict,
) -> torch.Tensor:
    """Reconstrói tensor float a partir de tensor quantizado por grupo."""
    dtype = getattr(torch, meta["original_dtype"].replace("torch.", ""))
    q = quantized.float()  # shape: (n_rows, n_groups, group_size)

    reconstructed = q * meta["scale"] + meta["zero"]
    reconstructed = reconstructed.reshape(meta["n_rows"], -1)

    # remover padding se necessário
    if meta["pad"] > 0:
        reconstructed = reconstructed[:, : -meta["pad"]]

    return reconstructed.reshape(meta["original_shape"]).to(dtype)
