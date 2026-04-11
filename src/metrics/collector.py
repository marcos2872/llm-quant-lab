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
from typing import Any

import psutil
import torch

# ── estruturas de dados ────────────────────────────────────────────────────────

@dataclass
class MemorySnapshot:
    """Snapshot de uso de memória em MB."""
    weights_mb: float = 0.0
    peak_mb: float = 0.0
    kv_mb: float = 0.0

    def to_dict(self) -> dict:
        return {
            "weights_mb": round(self.weights_mb, 2),
            "peak_mb": round(self.peak_mb, 2),
            "kv_mb": round(self.kv_mb, 2),
        }


@dataclass
class Throughput:
    """Métricas de velocidade de geração."""
    prefill_tok_s: float = 0.0
    decode_tok_s: float = 0.0
    first_token_latency_s: float = 0.0
    total_time_s: float = 0.0
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


# ── funções públicas de medição de memória ────────────────────────────────────

def current_memory_mb() -> float:
    """Retorna memória alocada atual em MB (GPU se disponível, senão RAM)."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 ** 2
    return psutil.Process().memory_info().rss / 1024 ** 2


def reset_peak() -> None:
    """Reseta o contador de pico de memória GPU."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def peak_memory_mb() -> float:
    """Retorna pico de memória desde o último reset em MB."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024 ** 2
    return psutil.Process().memory_info().rss / 1024 ** 2


# ── helpers internos de throughput ────────────────────────────────────────────

def _run_prefill(model: Any, inputs: dict) -> tuple[float, float]:
    """Executa prefill e retorna (tok_s, elapsed_s)."""
    n_tokens = inputs["input_ids"].shape[-1]
    t0 = time.perf_counter()
    with torch.no_grad():
        model(**inputs, use_cache=True)
    elapsed = time.perf_counter() - t0
    return (n_tokens / elapsed if elapsed > 0 else 0.0), elapsed


def _run_decode(
    model: Any,
    inputs: dict,
    max_new_tokens: int,
) -> tuple[Any, float]:
    """Executa decode e retorna (output_ids, elapsed_s)."""
    reset_peak()
    t0 = time.perf_counter()
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
        )
    return output_ids, time.perf_counter() - t0


# ── API pública de throughput ──────────────────────────────────────────────────

def measure_throughput(
    model: Any,
    tokenizer: Any,
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

    prefill_tok_s, prefill_time = _run_prefill(model, inputs)
    output_ids, gen_time = _run_decode(model, inputs, max_new_tokens)

    output_len = output_ids.shape[-1] - input_len
    decode_tok_s = output_len / gen_time if gen_time > 0 else 0.0
    generated = tokenizer.decode(output_ids[0][input_len:], skip_special_tokens=True)

    tp = Throughput(
        prefill_tok_s=prefill_tok_s,
        decode_tok_s=decode_tok_s,
        first_token_latency_s=round(gen_time / max(output_len, 1), 4),
        total_time_s=prefill_time + gen_time,
        input_tokens=input_len,
        output_tokens=output_len,
    )
    return tp, generated
