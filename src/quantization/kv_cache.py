"""
src/quantization/kv_cache.py
-----------------------------
Cache KV quantizado que armazena o contexto do prefill em formato comprimido.

Estratégia híbrida:
  - Prefill (N tokens): quantizado uma única vez → redução real de memória GPU
  - Decode (tokens novos): mantidos em FP16 num buffer pequeno (< 256 tokens)

Isso elimina re-quantização a cada passo e preserva a qualidade do decode.
Subclasse de DynamicCache para compatibilidade total com transformers.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from transformers import DynamicCache


def _mb(t: torch.Tensor) -> float:
    """Tamanho do tensor em MB."""
    return t.element_size() * t.numel() / 1024 ** 2


class QuantizedDynamicCache(DynamicCache):
    """
    DynamicCache com armazenamento quantizado do contexto de prefill.

    key_cache / value_cache do pai ficam vazios — usamos _qhist e _fp16k/v.
    """

    def __init__(
        self,
        quantize_fn: Callable[[torch.Tensor], tuple[Any, Any]],
        dequantize_fn: Callable[[Any, Any], torch.Tensor],
        tracker: list[float],
    ) -> None:
        super().__init__()
        self.quantize_fn = quantize_fn
        self.dequantize_fn = dequantize_fn
        self.tracker = tracker
        # Prefill: (qk, meta_k, qv, meta_v) por camada
        self._qhist: list[tuple] = []
        # Decode: buffer FP16 com tokens novos (crescimento lento)
        self._fp16k: list[torch.Tensor | None] = []
        self._fp16v: list[torch.Tensor | None] = []
        self._seq_len: int = 0

    # ── handlers internos ────────────────────────────────────────────────────

    def _handle_prefill(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Quantiza e armazena o prefill com codebook por camada."""
        qk, mk = self.quantize_fn(key_states, layer_idx=layer_idx)
        qv, mv = self.quantize_fn(value_states, layer_idx=layer_idx)
        self._qhist.append((qk, mk, qv, mv))
        self._fp16k.append(None)
        self._fp16v.append(None)
        self.tracker.append(_mb(qk) + _mb(qv))
        return key_states, value_states

    def _handle_decode(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Concatena token novo com histórico dequantizado; mantém FP16 buffer."""
        qk, mk, qv, mv = self._qhist[layer_idx]
        parts_k: list[torch.Tensor] = [self.dequantize_fn(qk, mk)]
        parts_v: list[torch.Tensor] = [self.dequantize_fn(qv, mv)]
        if self._fp16k[layer_idx] is not None:
            parts_k.append(self._fp16k[layer_idx])
            parts_v.append(self._fp16v[layer_idx])
        parts_k.append(key_states)
        parts_v.append(value_states)
        # Acumula token novo no buffer FP16
        prev_k = self._fp16k[layer_idx]
        self._fp16k[layer_idx] = (
            key_states if prev_k is None else torch.cat([prev_k, key_states], dim=-2)
        )
        prev_v = self._fp16v[layer_idx]
        self._fp16v[layer_idx] = (
            value_states if prev_v is None else torch.cat([prev_v, value_states], dim=-2)
        )
        return torch.cat(parts_k, dim=-2), torch.cat(parts_v, dim=-2)

    # ── interface Cache (transformers) ────────────────────────────────────────

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: dict | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Roteia para prefill ou decode e atualiza seq_len."""
        if layer_idx >= len(self._qhist):
            full_k, full_v = self._handle_prefill(key_states, value_states, layer_idx)
        else:
            full_k, full_v = self._handle_decode(key_states, value_states, layer_idx)
        self._seq_len = full_k.shape[-2]
        return full_k, full_v

    def get_seq_length(self, layer_idx: int = 0) -> int:
        return self._seq_len

    def get_usable_length(self, new_seq_length: int, layer_idx: int = 0) -> int:
        return self._seq_len

    def get_max_cache_shape(self) -> tuple | None:
        return None

    # ── propriedades de compatibilidade ──────────────────────────────────────

    @property
    def key_cache(self) -> list[torch.Tensor]:
        """Reconstrói key cache completo (quantizado + buffer FP16)."""
        result = []
        for i, (qk, mk, _, _) in enumerate(self._qhist):
            parts = [self.dequantize_fn(qk, mk)]
            if self._fp16k[i] is not None:
                parts.append(self._fp16k[i])
            result.append(torch.cat(parts, dim=-2))
        return result

    @property
    def value_cache(self) -> list[torch.Tensor]:
        """Reconstrói value cache completo (quantizado + buffer FP16)."""
        result = []
        for i, (_, _, qv, mv) in enumerate(self._qhist):
            parts = [self.dequantize_fn(qv, mv)]
            if self._fp16v[i] is not None:
                parts.append(self._fp16v[i])
            result.append(torch.cat(parts, dim=-2))
        return result
