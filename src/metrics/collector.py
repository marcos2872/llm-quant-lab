"""
src/metrics/collector.py
------------------------
Coleta de métricas de memória e throughput durante inferência LLM.

Memória: torch.cuda (GPU) ou psutil.Process (CPU/RAM).
Throughput: separação explícita prefill × decode com time.perf_counter.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import psutil
import torch

# ── estruturas de dados ────────────────────────────────────────────────────────

@dataclass
class MemorySnapshot:
    """Snapshot de uso de memória em MB."""
    weights_mb: float = 0.0    # memória alocada antes da geração (pesos + estado)
    peak_mb: float = 0.0       # pico durante a geração
    kv_mb: float = 0.0         # delta entre pico e pré-geração (estimativa KV)

    def to_dict(self) -> dict:
        return {
            "weights_mb": round(self.weights_mb, 2),
            "peak_mb": round(self.peak_mb, 2),
            "kv_mb": round(self.kv_mb, 2),
        }


@dataclass
class Throughput:
    """Métricas de velocidade de geração."""
    prefill_tok_s: float = 0.0        # tokens/s na fase de prefill
    decode_tok_s: float = 0.0         # tokens/s na fase de decode
    first_token_latency_s: float = 0.0  # latência até o primeiro token
    total_time_s: float = 0.0         # tempo total da geração
    input_tokens: int = 0
    output_tokens: int = 0

    def to_dict(self) -> dict:
        return {
            "prefill_tok_s": round(self.prefill_tok_s, 2),
            "decode_tok_s": round(self.decode_tok_s, 2),
            "first_token_latency_s": round(self.first_token_latency_s, 4),
            "total_time_s": round(self.total_time_s, 4),
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
        }


@dataclass
class RunMetrics:
    """Métricas completas de uma execução."""
    memory: MemorySnapshot = field(default_factory=MemorySnapshot)
    throughput: Throughput = field(default_factory=Throughput)
    prompt_id: str = ""
    generated_text: str = ""

    def to_dict(self) -> dict:
        return {
            "prompt_id": self.prompt_id,
            "generated_text": self.generated_text,
            **self.memory.to_dict(),
            **self.throughput.to_dict(),
        }


# ── funções de medição ─────────────────────────────────────────────────────────

def _current_memory_mb() -> float:
    """Retorna memória alocada atual em MB (GPU se disponível, senão RAM)."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 ** 2
    return psutil.Process().memory_info().rss / 1024 ** 2


def _reset_peak() -> None:
    """Reseta o contador de pico de memória GPU."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def _peak_memory_mb() -> float:
    """Retorna pico de memória desde o último reset em MB."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024 ** 2
    return psutil.Process().memory_info().rss / 1024 ** 2


def measure_memory_snapshot(after_generation_fn: None = None) -> tuple[float, float]:
    """
    Retorna (weights_mb, peak_mb) medindo antes e durante a geração.
    Uso: chamar _reset_peak() antes, rodar geração, chamar esta função depois.
    """
    weights_mb = _current_memory_mb()
    peak_mb = _peak_memory_mb()
    return weights_mb, peak_mb


def measure_throughput(
    model: object,
    tokenizer: object,
    prompt: str,
    max_new_tokens: int = 256,
    device: str = "cpu",
) -> tuple[Throughput, str]:
    """
    Mede throughput de prefill e decode separadamente.

    Retorna (Throughput, texto_gerado).
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_len = inputs["input_ids"].shape[-1]

    # ── prefill: forward pass sem geração ────────────────────────────────────
    t0 = time.perf_counter()
    with torch.no_grad():
        _ = model(**inputs, use_cache=True)
    prefill_time = time.perf_counter() - t0
    prefill_tok_s = input_len / prefill_time if prefill_time > 0 else 0.0

    # ── decode: geração token a token ─────────────────────────────────────────
    _reset_peak()
    t_gen_start = time.perf_counter()
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
        )
    t_gen_end = time.perf_counter()

    output_len = output_ids.shape[-1] - input_len
    gen_time = t_gen_end - t_gen_start
    decode_tok_s = output_len / gen_time if gen_time > 0 else 0.0

    generated = tokenizer.decode(
        output_ids[0][input_len:], skip_special_tokens=True
    )

    tp = Throughput(
        prefill_tok_s=prefill_tok_s,
        decode_tok_s=decode_tok_s,
        first_token_latency_s=round(gen_time / max(output_len, 1), 4),
        total_time_s=prefill_time + gen_time,
        input_tokens=input_len,
        output_tokens=output_len,
    )
    return tp, generated
