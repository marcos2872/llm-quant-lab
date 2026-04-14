"""
src/quantization/kv_hooks.py
-----------------------------
Instala e remove hooks PyTorch que interceptam K/V nos módulos de atenção.

Compatibilidade: Qwen2, Llama-3, Mistral, GPT-2, Falcon, Pythia.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import torch

logger = logging.getLogger(__name__)

_ATTN_ATTR_NAMES = ("self_attn", "attn", "attention", "self_attention")

# Nomes de módulos que NÃO são de atenção mesmo sem filhos (evita falsos positivos)
_NON_ATTN_MODULE_TYPES = (
    torch.nn.LayerNorm,
    torch.nn.Dropout,
    torch.nn.Embedding,
    torch.nn.Linear,
    torch.nn.ReLU,
    torch.nn.GELU,
    torch.nn.SiLU,
)


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
    """
    Busca recursiva por módulos de atenção.

    Exclui tipos conhecidos que não são de atenção (LayerNorm, Dropout, etc.)
    para evitar instalação de hooks em módulos incorretos.
    """
    found = []
    for name, module in model.named_modules():
        lname = name.lower()
        is_attn_name = "attn" in lname or "attention" in lname
        is_leaf = list(module.children()) == []
        is_non_attn_type = isinstance(module, _NON_ATTN_MODULE_TYPES)
        if is_attn_name and is_leaf and not is_non_attn_type:
            found.append(module)
    return found


def _is_past_kv(obj: Any) -> bool:
    """Verifica se obj é past_key_value (tupla de tensores ou objeto Cache)."""
    if isinstance(obj, tuple) and len(obj) == 2:
        return all(isinstance(t, torch.Tensor) for t in obj)
    return hasattr(obj, "key_cache") and hasattr(obj, "value_cache")


def _tensor_mb(t: torch.Tensor) -> float:
    return t.element_size() * t.numel() / 1024 ** 2


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


def install_kv_proj_hooks(
    model: torch.nn.Module,
    quantize_fn: Callable[[torch.Tensor], tuple[Any, Any]],
    dequantize_fn: Callable[[Any, Any], torch.Tensor],
) -> tuple[list[Any], list[float]]:
    """
    Instala hooks após k_proj e v_proj de cada camada de atenção.

    Compatível com arquiteturas onde o cache é atualizado internamente
    (ex: Qwen2.5) e o módulo de atenção não retorna past_key_values
    no output tuple. Semanticamente equivalente a quantizar o KV cache:
    quantiza e dequantiza o output das projeções antes do cálculo de atenção.

    Retorna (handles, kv_mem_tracker). Passar handles para remove_kv_hooks.
    """
    attn_layers = _find_attention_layers(model)
    if not attn_layers:
        logger.warning("Nenhum attention layer encontrado — hooks não instalados")
        return [], []

    kv_mem_tracker: list[float] = []
    handles: list[Any] = []

    for attn in attn_layers:
        # detecta n_kv_heads e head_dim do módulo de atenção para reshape correto
        n_kv_heads = (
            getattr(attn, "num_key_value_heads", None)
            or getattr(attn, "num_kv_heads", None)
            or getattr(attn, "num_heads", 1)
        )
        head_dim = getattr(attn, "head_dim", None)

        for proj_name in ("k_proj", "v_proj"):
            proj = getattr(attn, proj_name, None)
            if proj is None:
                continue

            def _make_proj_hook(
                qfn: Callable, dqfn: Callable,
                tracker: list[float],
                n_heads: int, hdim: int | None,
            ) -> Callable:
                def hook(
                    module: torch.nn.Module,
                    inputs: tuple,  # noqa: ARG001
                    output: torch.Tensor,
                ) -> torch.Tensor:
                    """Reshape 3-D proj output → 4-D, quantiza, restaura."""
                    orig_shape = output.shape
                    if output.ndim == 3:
                        b, s, total = orig_shape
                        h = n_heads
                        d = hdim if hdim is not None else total // h
                        t4d = output.view(b, s, h, d).permute(0, 2, 1, 3)
                    else:
                        t4d = output
                    q, meta = qfn(t4d, layer_idx=0)
                    tracker.append(_tensor_mb(q))
                    recon = dqfn(q, meta)
                    if output.ndim == 3:
                        b, s, total = orig_shape
                        return recon.permute(0, 2, 1, 3).reshape(orig_shape)
                    return recon
                return hook

            h = proj.register_forward_hook(
                _make_proj_hook(
                    quantize_fn, dequantize_fn, kv_mem_tracker,
                    n_kv_heads, head_dim,
                )
            )
            handles.append(h)

    logger.info("KV proj hooks instalados em %d projeções", len(handles))
    return handles, kv_mem_tracker


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
