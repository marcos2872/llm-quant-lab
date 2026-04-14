"""
src/quantization/methods/uniform.py
-------------------------------------
Quantização uniforme (min-max) por tensor de KV cache.

Método:
  - Encontra min e max do tensor
  - Escala para [0, 2^bits - 1] como inteiro
  - Dequantiza revertendo a escala

Retorna tensores quantizados como int8/int16/int32 dependendo de bits.
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


def quantize_uniform(
    tensor: torch.Tensor,
    bits: int = 4,
    layer_idx: int = 0,  # ignorado; mantém interface compatível com turboquant
) -> tuple[torch.Tensor, dict]:
    """
    Quantiza tensor com escala uniforme global min-max.

    Retorna (tensor_quantizado, metadata) onde metadata contém
    scale e zero_point para reconstrução.
    """
    n_levels = 2 ** bits
    t_min = tensor.min().item()
    t_max = tensor.max().item()
    scale = (t_max - t_min) / (n_levels - 1) if t_max != t_min else 1.0

    quantized = ((tensor.float() - t_min) / scale).round().clamp(0, n_levels - 1)
    # int8 suporta apenas -128..127; para 8 bits os índices chegam a 255 → overflow
    dtype = torch.int8 if bits < 8 else torch.int16
    quantized = quantized.to(dtype)
    n_elements = quantized.numel()
    packed = _pack_indices_flat(quantized, bits) if bits in (2, 4) else quantized

    meta = {
        "scale": scale, "zero_point": t_min, "shape": tensor.shape,
        "dtype": str(tensor.dtype), "bits": bits, "n_elements": n_elements,
    }
    return packed, meta


def dequantize_uniform(
    quantized: torch.Tensor,
    meta: dict,
) -> torch.Tensor:
    """Reconstrói tensor float a partir de tensor quantizado e metadata."""
    bits = meta.get("bits", 8)
    n_elements = meta.get("n_elements")
    if bits in (2, 4) and n_elements is not None:
        quantized = _unpack_indices_flat(quantized, bits, n_elements)
    dtype = getattr(torch, meta["dtype"].replace("torch.", ""))
    return (quantized.reshape(meta["shape"]).float() * meta["scale"] + meta["zero_point"]).to(dtype)
