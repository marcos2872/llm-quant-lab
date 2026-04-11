"""
src/quantization/kv_hooks.py
-----------------------------
Instala e remove hooks PyTorch que interceptam os tensores K/V
nos módulos de atenção do modelo para aplicar quantização em tempo de execução.

Compatibilidade testada com: Qwen2, Llama-3, Mistral.

Fluxo:
  1. install_kv_hooks(model, quantize_fn, dequantize_fn, ...)
     → percorre model.model.layers, registra forward hooks
  2. Durante model.generate(), cada forward dos attention layers
     chama os hooks que quantizam → dequantizam K e V
  3. remove_kv_hooks(handles) limpa todos os hooks após a geração
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import torch

logger = logging.getLogger(__name__)

# Nomes de atributo usados pelos attention modules de diferentes arquiteturas
_ATTN_ATTR_NAMES = ("self_attn", "attn", "attention", "self_attention")


def _find_attention_layers(model: torch.nn.Module) -> list[torch.nn.Module]:
    """
    Localiza os módulos de atenção do modelo de forma agnóstica à arquitetura.

    Tenta model.model.layers primeiro; faz fallback para busca recursiva.
    """
    # caminho mais comum: Llama / Qwen / Mistral
    decoder_layers = None
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        decoder_layers = model.model.layers
    elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        decoder_layers = model.transformer.h  # GPT-2 / Falcon
    elif hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        decoder_layers = model.gpt_neox.layers  # Pythia / GPT-NeoX

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
    found = []
    for name, module in model.named_modules():
        lname = name.lower()
        if ("attn" in lname or "attention" in lname) and list(module.children()) == []:
            found.append(module)
    return found


def _make_kv_hook(
    quantize_fn: Callable[[torch.Tensor], tuple[Any, Any]],
    dequantize_fn: Callable[[Any, Any], torch.Tensor],
    kv_mem_tracker: list[float],
) -> Callable:
    """
    Cria um forward hook que intercepta saídas do módulo de atenção.

    Arquiteturas Llama/Qwen retornam (attn_output, attn_weights, past_key_value).
    past_key_value é uma tupla (key, value) ou um objeto DynamicCache.
    O hook quantiza e dequantiza K/V para simular o custo de memória reduzido.
    """

    def hook(module, inputs, outputs):  # noqa: ARG001
        if not isinstance(outputs, tuple):
            return outputs

        # Localiza past_key_value no output (posição 1 ou 2)
        pkv = None
        pkv_idx = None
        for idx, out in enumerate(outputs):
            if _is_past_kv(out):
                pkv = out
                pkv_idx = idx
                break

        if pkv is None:
            return outputs

        # quantiza → dequantiza K e V
        new_pkv = _process_past_kv(pkv, quantize_fn, dequantize_fn, kv_mem_tracker)
        outputs = outputs[:pkv_idx] + (new_pkv,) + outputs[pkv_idx + 1:]
        return outputs

    return hook


def _is_past_kv(obj: Any) -> bool:
    """Verifica se obj é past_key_value (tupla de tensores ou objeto Cache)."""
    if isinstance(obj, tuple) and len(obj) == 2:
        return all(isinstance(t, torch.Tensor) for t in obj)
    # Transformers >= 4.36: DynamicCache / StaticCache
    return hasattr(obj, "key_cache") and hasattr(obj, "value_cache")


def _process_past_kv(
    pkv: Any,
    quantize_fn: Callable,
    dequantize_fn: Callable,
    tracker: list[float],
) -> Any:
    """Quantiza e dequantiza K e V dentro de past_key_value."""
    if isinstance(pkv, tuple):
        k, v = pkv
        qk, meta_k = quantize_fn(k)
        qv, meta_v = quantize_fn(v)
        tracker.append(_tensor_mb(qk) + _tensor_mb(qv))
        return (dequantize_fn(qk, meta_k), dequantize_fn(qv, meta_v))

    # objeto Cache — itera por camada
    for i in range(len(pkv.key_cache)):
        k, v = pkv.key_cache[i], pkv.value_cache[i]
        qk, meta_k = quantize_fn(k)
        qv, meta_v = quantize_fn(v)
        tracker.append(_tensor_mb(qk) + _tensor_mb(qv))
        pkv.key_cache[i] = dequantize_fn(qk, meta_k)
        pkv.value_cache[i] = dequantize_fn(qv, meta_v)
    return pkv


def _tensor_mb(t: torch.Tensor) -> float:
    return t.element_size() * t.numel() / 1024 ** 2


# ── API pública ────────────────────────────────────────────────────────────────

def install_kv_hooks(
    model: torch.nn.Module,
    quantize_fn: Callable[[torch.Tensor], tuple[Any, Any]],
    dequantize_fn: Callable[[Any, Any], torch.Tensor],
) -> tuple[list[Any], list[float]]:
    """
    Instala hooks de quantização de KV cache em todos os attention layers.

    Parâmetros
    ----------
    model         : modelo carregado
    quantize_fn   : fn(tensor) → (quantized, metadata)
    dequantize_fn : fn(quantized, metadata) → tensor

    Retorna
    -------
    handles      : lista de hook handles (passar para remove_kv_hooks)
    kv_mem_tracker: lista acumulada de MB de KV quantizado por passo
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
