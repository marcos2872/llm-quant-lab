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

    meta = {"scale": scale, "zero_point": t_min, "shape": tensor.shape, "dtype": str(tensor.dtype)}
    return quantized, meta


def dequantize_uniform(
    quantized: torch.Tensor,
    meta: dict,
) -> torch.Tensor:
    """Reconstrói tensor float a partir de tensor quantizado e metadata."""
    dtype = getattr(torch, meta["dtype"].replace("torch.", ""))
    return (quantized.float() * meta["scale"] + meta["zero_point"]).to(dtype)
