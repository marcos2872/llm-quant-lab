"""
src/quantization/methods/uniform.py
-------------------------------------
Quantização uniforme (min-max) por cabeça de atenção (per-head) de KV cache.

Método:
  - Separa outlier_channels de maior magnitude e os preserva em FP16
  - Para os canais normais, calcula min/max por head (batch×head)
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


def _detect_outlier_channels(
    tensor: torch.Tensor, n_outliers: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Detecta os n_outliers canais de maior magnitude média.

    Retorna (outlier_idx, normal_idx) — índices sobre a dimensão head_dim.
    """
    head_dim = tensor.shape[-1]
    n_out = min(n_outliers, head_dim)
    scores = tensor.reshape(-1, head_dim).abs().mean(dim=0)
    _, out_idx = scores.topk(n_out)
    out_idx, _ = out_idx.sort()
    mask = torch.ones(head_dim, dtype=torch.bool, device=tensor.device)
    mask[out_idx] = False
    norm_idx = mask.nonzero(as_tuple=True)[0]
    return out_idx, norm_idx


def quantize_uniform(
    tensor: torch.Tensor,
    bits: int = 4,
    outlier_channels: int = 0,
    layer_idx: int = 0,  # ignorado; mantém interface compatível com turboquant
) -> tuple[torch.Tensor, dict]:
    """
    Quantiza tensor com escala uniforme min-max por head de atenção.

    Quando outlier_channels > 0, separa os canais de maior magnitude em FP16
    e quantiza os demais — evita que outliers contaminem a escala dos canais
    normais (causa do colapso em contextos longos).

    Retorna (tensor_quantizado, metadata) onde metadata contém tudo o
    necessário para reconstrução, incluindo os canais outlier em FP16.
    """
    assert tensor.ndim == 4, f"Esperado ndim=4, recebido shape={tensor.shape}"
    b, h, s, d = tensor.shape
    n_levels = 2 ** bits

    outlier_idx: torch.Tensor | None = None
    normal_idx: torch.Tensor | None = None
    outlier_fp16: torch.Tensor | None = None

    if outlier_channels > 0:
        outlier_idx, normal_idx = _detect_outlier_channels(tensor, outlier_channels)
        outlier_fp16 = tensor[:, :, :, outlier_idx].half()
        work = tensor[:, :, :, normal_idx]
    else:
        work = tensor

    # min/max por head: (b, h) → broadcast (b, h, 1, 1)
    t_min = work.flatten(-2).min(-1).values.view(b, h, 1, 1)
    t_max = work.flatten(-2).max(-1).values.view(b, h, 1, 1)
    scale = ((t_max - t_min) / (n_levels - 1)).clamp(min=1e-8)

    quantized = ((work.float() - t_min) / scale).round().clamp(0, n_levels - 1)
    dtype = torch.int8 if bits < 8 else torch.int16
    quantized = quantized.to(dtype)
    n_elements = quantized.numel()
    packed = _pack_indices_flat(quantized, bits) if bits in (2, 4) else quantized

    meta: dict = {
        "scale": scale, "zero_point": t_min, "shape": work.shape,
        "original_shape": tensor.shape, "dtype": str(tensor.dtype),
        "bits": bits, "n_elements": n_elements,
        "outlier_idx": outlier_idx, "normal_idx": normal_idx,
        "outlier_fp16": outlier_fp16,
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

    work = (quantized.reshape(meta["shape"]).float() * meta["scale"] + meta["zero_point"]).to(dtype)

    outlier_idx = meta.get("outlier_idx")
    if outlier_idx is None:
        return work

    # reconstrói tensor completo mesclando normais quantizados + outliers FP16
    orig_shape = meta["original_shape"]
    result = torch.empty(orig_shape, dtype=dtype, device=quantized.device)
    normal_idx = meta["normal_idx"]
    result[:, :, :, normal_idx.to(quantized.device)] = work
    result[:, :, :, outlier_idx.to(quantized.device)] = (
        meta["outlier_fp16"].to(dtype).to(quantized.device)
    )
    return result
