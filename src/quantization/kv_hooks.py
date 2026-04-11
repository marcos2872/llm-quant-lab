"""
src/quantization/kv_hooks.py
-----------------------------
Instala e remove hooks PyTorch que interceptam os tensores K/V
nos módulos de atenção do modelo para aplicar quantização em tempo de execução.

Compatibilidade testada com: Qwen2, Llama-3, Mistral.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import torch

logger = logging.getLogger(__name__)

_ATTN_ATTR_NAMES = ("self_attn", "attn", "attention", "self_attention")


def _find_attention_layers(model: torch.nn.Module) -> list[torch.nn.Module]:
    """Localiza os módulos de atenção de forma agnóstica à arquitetura."""
    decoder_layers = None
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        decoder_layers = model.model.layers
    elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        decoder_layers = model.transformer.h
    elif hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        decoder_layers = model.gpt_neox.layers

    if decoder_layers is None:
        logger.warning("Arquitetura não reconhecida — buscando atenção recursivamente")
        return _find_attention_recursive(model)

    attn_layers = []
    for layer in decoder_layers:
        for attr in _ATTN_ATTR_NAMES:
            if hasattr(layer, attr):
                attn_layers.append(getattr(layer, attr))
                break
    return attn_layers


def _find_attention_recursive(model: torch.nn.Module) -> list[torch.nn.Module]:
    """Busca recursiva por módulos cujo nome contém 'attn' ou 'attention'."""
    return [
        module
        for name, module in model.named_modules()
        if ("attn" in name.lower() or "attention" in name.lower())
        and list(module.children()) == []
    ]


def _is_past_kv(obj: Any) -> bool:
    """Verifica se obj é past_key_value (tupla de tensores ou objeto Cache)."""
    if isinstance(obj, tuple) and len(obj) == 2:
        return all(isinstance(t, torch.Tensor) for t in obj)
    return hasattr(obj, "key_cache") and hasattr(obj, "value_cache")


def _process_past_kv(
    pkv: Any,
    quantize_fn: Callable,
    dequantize_fn: Callable,
    tracker: list[float],
) -> Any:
    """Quantiza e dequantiza K e V dentro de past_key_value."""
    def _tensor_mb(t: torch.Tensor) -> float:
        return t.element_size() * t.numel() / 1024 ** 2

    if isinstance(pkv, tuple):
        k, v = pkv
        qk, meta_k = quantize_fn(k)
        qv, meta_v = quantize_fn(v)
        tracker.append(_tensor_mb(qk) + _tensor_mb(qv))
        return (dequantize_fn(qk, meta_k), dequantize_fn(qv, meta_v))

    for i in range(len(pkv.key_cache)):
        k, v = pkv.key_cache[i], pkv.value_cache[i]
        qk, meta_k = quantize_fn(k)
        qv, meta_v = quantize_fn(v)
        tracker.append(_tensor_mb(qk) + _tensor_mb(qv))
        pkv.key_cache[i] = dequantize_fn(qk, meta_k)
        pkv.value_cache[i] = dequantize_fn(qv, meta_v)
    return pkv


def _make_kv_hook(
    quantize_fn: Callable[[torch.Tensor], tuple[Any, Any]],
    dequantize_fn: Callable[[Any, Any], torch.Tensor],
    kv_mem_tracker: list[float],
) -> Callable[[torch.nn.Module, tuple, tuple], tuple]:
    """Cria forward hook que intercepta e quantiza K/V nas saídas da atenção."""

    def hook(
        module: torch.nn.Module,
        inputs: tuple,  # noqa: ARG001
        outputs: tuple,
    ) -> tuple:
        if not isinstance(outputs, tuple):
            return outputs
        pkv, pkv_idx = None, None
        for idx, out in enumerate(outputs):
            if _is_past_kv(out):
                pkv, pkv_idx = out, idx
                break
        if pkv is None:
            return outputs
        new_pkv = _process_past_kv(pkv, quantize_fn, dequantize_fn, kv_mem_tracker)
        return outputs[:pkv_idx] + (new_pkv,) + outputs[pkv_idx + 1:]

    return hook


def install_kv_hooks(
    model: torch.nn.Module,
    quantize_fn: Callable[[torch.Tensor], tuple[Any, Any]],
    dequantize_fn: Callable[[Any, Any], torch.Tensor],
) -> tuple[list[Any], list[float]]:
    """
    Instala hooks de quantização de KV cache em todos os attention layers.

    Retorna (handles, kv_mem_tracker). Passar handles para remove_kv_hooks.
    """
    attn_layers = _find_attention_layers(model)
    if not attn_layers:
        logger.warning("Nenhum attention layer encontrado — hooks não instalados")
        return [], []

    kv_mem_tracker: list[float] = []
    hook_fn = _make_kv_hook(quantize_fn, dequantize_fn, kv_mem_tracker)
    handles = [layer.register_forward_hook(hook_fn) for layer in attn_layers]
    logger.info("KV hooks instalados em %d camadas", len(handles))
    return handles, kv_mem_tracker


def remove_kv_hooks(handles: list[Any]) -> None:
    """Remove todos os hooks instalados por install_kv_hooks."""
    for h in handles:
        h.remove()
    logger.info("KV hooks removidos (%d)", len(handles))
