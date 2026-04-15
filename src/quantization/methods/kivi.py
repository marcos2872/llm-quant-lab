"""
src/quantization/methods/kivi.py
----------------------------------
Quantização estilo KIVI: quantização por canal (per-channel) ao longo de
head_dim para KV cache.

Estratégia de outliers:
  Quando outlier_channels > 0, os canais de maior magnitude são preservados
  em FP16 e excluídos da escala de quantização dos canais normais. Isso evita
  que outliers (presentes em ~25% dos canais em modelos como Qwen2.5)
  contaminem a precisão dos canais normais — principal causa do colapso em
  contextos longos.

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


def _quantize_perchannel(
    tensor: torch.Tensor, bits: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """
    Quantização per-channel: cada dimensão head_dim tem sua própria escala.

    tensor: shape (b, h, seq, d_normal)
    Retorna (packed, t_min, scale, n_elements).
    """
    n_levels = 2 ** bits
    # min/max por canal (dim -1) sobre a sequência: shape (b, h, 1, d_normal)
    t_min = tensor.min(dim=-2, keepdim=True).values
    t_max = tensor.max(dim=-2, keepdim=True).values
    scale = ((t_max - t_min) / (n_levels - 1)).clamp(min=1e-8)
    q = ((tensor.float() - t_min) / scale).round().clamp(0, n_levels - 1)
    dtype = torch.int8 if bits < 8 else torch.int16
    q = q.to(dtype)
    n_elements = q.numel()
    packed = _pack_indices_flat(q, bits) if bits in (2, 4) else q
    return packed, t_min, scale, n_elements


def _quantize_pertoken(
    tensor: torch.Tensor, bits: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """
    Quantização per-token: cada posição de sequência tem sua própria escala.

    Usada para o value cache conforme paper KIVI (§3): min/max calculado
    ao longo de head_dim (dim=-1), produzindo shape (b, h, seq, 1).
    Contrasta com _quantize_perchannel que usa dim=-2 para keys.
    """
    n_levels = 2 ** bits
    t_min = tensor.min(dim=-1, keepdim=True).values
    t_max = tensor.max(dim=-1, keepdim=True).values
    scale = ((t_max - t_min) / (n_levels - 1)).clamp(min=1e-8)
    q = ((tensor.float() - t_min) / scale).round().clamp(0, n_levels - 1)
    dtype = torch.int8 if bits < 8 else torch.int16
    q = q.to(dtype)
    n_elements = q.numel()
    packed = _pack_indices_flat(q, bits) if bits in (2, 4) else q
    return packed, t_min, scale, n_elements


def quantize_kivi(
    tensor: torch.Tensor,
    bits: int = 4,
    group_size: int = 64,   # mantido para compatibilidade; não utilizado
    outlier_channels: int = 0,
    layer_idx: int = 0,     # ignorado; mantém interface compatível com turboquant
) -> tuple[torch.Tensor, dict]:
    """
    Quantização per-channel ao longo de head_dim com outlier FP16.

    tensor: shape (batch, n_kv_heads, seq_len, head_dim)
    outlier_channels > 0: separa os canais de maior magnitude em FP16 antes
    de quantizar os demais, evitando contaminação de escala por outliers.
    """
    assert tensor.ndim == 4, f"Esperado ndim=4, recebido shape={tensor.shape}"
    original_shape = tensor.shape
    original_dtype = tensor.dtype

    outlier_idx: torch.Tensor | None = None
    normal_idx: torch.Tensor | None = None
    outlier_fp16: torch.Tensor | None = None

    if outlier_channels > 0:
        outlier_idx, normal_idx = _detect_outlier_channels(tensor, outlier_channels)
        outlier_fp16 = tensor[:, :, :, outlier_idx].half()
        work = tensor[:, :, :, normal_idx]
    else:
        work = tensor

    packed, t_min, scale, n_elements = _quantize_perchannel(work, bits)

    meta = {
        "t_min": t_min, "scale": scale, "shape": work.shape,
        "original_shape": original_shape,
        "original_dtype": str(original_dtype),
        "bits": bits, "n_elements": n_elements,
        "outlier_idx": outlier_idx, "normal_idx": normal_idx,
        "outlier_fp16": outlier_fp16,
    }
    return packed, meta


def quantize_kivi_value(
    tensor: torch.Tensor,
    bits: int = 4,
    group_size: int = 64,   # mantido para compatibilidade
    outlier_channels: int = 0,
    layer_idx: int = 0,
) -> tuple[torch.Tensor, dict]:
    """
    Quantização per-token para value cache (paper KIVI §3).

    Diferente de quantize_kivi (keys, per-channel): aqui cada token tem sua
    própria escala (min/max ao longo de head_dim). dequantize_kivi é reutilizado
    pois usa meta['scale'] e meta['t_min'] que têm shape correto em ambos os casos.

    tensor: shape (batch, n_kv_heads, seq_len, head_dim)
    """
    assert tensor.ndim == 4, f"Esperado ndim=4, recebido shape={tensor.shape}"
    original_shape = tensor.shape
    original_dtype = tensor.dtype

    outlier_idx: torch.Tensor | None = None
    normal_idx: torch.Tensor | None = None
    outlier_fp16: torch.Tensor | None = None

    if outlier_channels > 0:
        outlier_idx, normal_idx = _detect_outlier_channels(tensor, outlier_channels)
        outlier_fp16 = tensor[:, :, :, outlier_idx].half()
        work = tensor[:, :, :, normal_idx]
    else:
        work = tensor

    packed, t_min, scale, n_elements = _quantize_pertoken(work, bits)

    meta = {
        "t_min": t_min, "scale": scale, "shape": work.shape,
        "original_shape": original_shape,
        "original_dtype": str(original_dtype),
        "bits": bits, "n_elements": n_elements,
        "outlier_idx": outlier_idx, "normal_idx": normal_idx,
        "outlier_fp16": outlier_fp16,
    }
    return packed, meta


def dequantize_kivi(quantized: torch.Tensor, meta: dict) -> torch.Tensor:
    """Reconstrói tensor float a partir de tensor quantizado per-channel."""
    bits = meta.get("bits", 8)
    n_elements = meta.get("n_elements")
    if bits in (2, 4) and n_elements is not None:
        quantized = _unpack_indices_flat(quantized, bits, n_elements)

    dtype = getattr(torch, meta["original_dtype"].replace("torch.", ""))
    work = (quantized.reshape(meta["shape"]).float() * meta["scale"] + meta["t_min"]).to(dtype)

    outlier_idx = meta.get("outlier_idx")
    if outlier_idx is None:
        return work

    orig_shape = meta["original_shape"]
    result = torch.empty(orig_shape, dtype=dtype, device=quantized.device)
    normal_idx = meta["normal_idx"]
    result[:, :, :, normal_idx.to(quantized.device)] = work
    result[:, :, :, outlier_idx.to(quantized.device)] = (
        meta["outlier_fp16"].to(dtype).to(quantized.device)
    )
    return result
