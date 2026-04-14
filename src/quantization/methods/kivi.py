"""
src/quantization/methods/kivi.py
----------------------------------
Quantização estilo KIVI: quantização por grupo ao longo da dimensão de canal.

Referência: https://arxiv.org/abs/2402.02750
"""

from __future__ import annotations

import torch


def _pack_indices_flat(indices: torch.Tensor, bits: int) -> torch.Tensor:
    """Empacota índices 1D em uint8: bits=2 → 4/byte, bits=4 → 2/byte."""
    if bits not in (2, 4):
        return indices
    ipb = 8 // bits
    flat = indices.int().reshape(-1)
    pad = (-flat.numel()) % ipb
    if pad:
        flat = torch.nn.functional.pad(flat, (0, pad))
    flat_u = flat.to(torch.uint8)
    packed = torch.zeros(flat_u.numel() // ipb, dtype=torch.uint8, device=indices.device)
    for i in range(ipb):
        packed |= flat_u[i::ipb] << ((ipb - 1 - i) * bits)
    return packed


def _unpack_indices_flat(packed: torch.Tensor, bits: int, n_elements: int) -> torch.Tensor:
    """Inverte _pack_indices_flat. n_elements: tamanho original antes do padding."""
    if bits not in (2, 4):
        return packed
    ipb, mask = 8 // bits, (1 << bits) - 1
    out = torch.zeros(packed.numel() * ipb, dtype=torch.int8, device=packed.device)
    for i in range(ipb):
        out[i::ipb] = ((packed >> ((ipb - 1 - i) * bits)) & mask).to(torch.int8)
    return out[:n_elements]


def _pad_and_group(
    flat: torch.Tensor, group_size: int
) -> tuple[torch.Tensor, int, int, int]:
    """Aplica padding e reshape para (n_rows, n_groups, group_size)."""
    n_rows, n_cols = flat.shape
    pad = (group_size - n_cols % group_size) % group_size
    if pad > 0:
        flat = torch.nn.functional.pad(flat, (0, pad))
    n_groups = flat.shape[-1] // group_size
    return flat.reshape(n_rows, n_groups, group_size), n_rows, n_groups, pad


def quantize_kivi(
    tensor: torch.Tensor,
    bits: int = 4,
    group_size: int = 64,
    layer_idx: int = 0,  # ignorado; mantém interface compatível com turboquant
) -> tuple[torch.Tensor, dict]:
    """
    Quantização por grupo ao longo da última dimensão (canal/head_dim).

    tensor: shape [..., seq_len, head_dim]
    """
    original_shape = tensor.shape
    original_dtype = tensor.dtype
    n_levels = 2 ** bits

    flat = tensor.float().reshape(-1, tensor.shape[-1])
    grouped, n_rows, n_groups, pad = _pad_and_group(flat, group_size)

    g_min = grouped.min(dim=-1, keepdim=True).values
    g_max = grouped.max(dim=-1, keepdim=True).values
    scale = (g_max - g_min) / (n_levels - 1)
    scale = scale.clamp(min=1e-8)

    q = ((grouped - g_min) / scale).round().clamp(0, n_levels - 1)
    # int8 suporta apenas -128..127; para 8 bits os índices chegam a 255 → overflow
    dtype = torch.int8 if bits < 8 else torch.int16

    meta = {
        "scale": scale,
        "zero": g_min,
        "original_shape": original_shape,
        "original_dtype": str(original_dtype),
        "n_rows": n_rows,
        "n_groups": n_groups,
        "group_size": group_size,
        "pad": pad,
        "bits": bits,
        "n_elements": q.numel(),
    }
    packed = _pack_indices_flat(q.to(dtype), bits) if bits in (2, 4) else q.to(dtype)
    return packed, meta


def dequantize_kivi(quantized: torch.Tensor, meta: dict) -> torch.Tensor:
    """Reconstrói tensor float a partir de tensor quantizado por grupo."""
    dtype = getattr(torch, meta["original_dtype"].replace("torch.", ""))
    bits = meta.get("bits", 8)
    n_elements = meta.get("n_elements")
    if bits in (2, 4) and n_elements is not None:
        quantized = _unpack_indices_flat(quantized, bits, n_elements)
    grouped = quantized.reshape(meta["n_rows"], meta["n_groups"], meta["group_size"])
    reconstructed = grouped.float() * meta["scale"] + meta["zero"]
    reconstructed = reconstructed.reshape(meta["n_rows"], -1)
    if meta["pad"] > 0:
        reconstructed = reconstructed[:, : -meta["pad"]]
    return reconstructed.reshape(meta["original_shape"]).to(dtype)
