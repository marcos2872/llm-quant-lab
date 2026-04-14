"""
src/runner/_utils.py
--------------------
Utilitários compartilhados entre os runners de inferência.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from src.metrics.collector import (
    RunMetrics,
    current_memory_mb,
    measure_throughput,
    peak_memory_mb,
    reset_peak,
)


def load_prompts(prompts_file: Path) -> list[dict]:
    """Lê arquivo JSONL de prompts e retorna lista de dicts."""
    return [
        json.loads(line)
        for line in prompts_file.read_text().splitlines()
        if line.strip()
    ]


def resolve_device(model: torch.nn.Module) -> str:
    """Obtém string do device do primeiro parâmetro do modelo."""
    try:
        return str(next(model.parameters()).device)
    except StopIteration:
        return "cpu"


def compute_kv_mb(model: object, seq_len: int) -> float:
    """
    Estima tamanho do KV cache analiticamente a partir de model.config.

    Fórmula: 2 (K+V) × n_layers × n_kv_heads × head_dim × seq_len × 2 bytes.
    Retorna 0.0 se model.config não tiver os campos esperados.
    """
    cfg = getattr(model, "config", None)
    if cfg is None or not seq_len:
        return 0.0
    n_layers = getattr(cfg, "num_hidden_layers", 0)
    n_heads  = getattr(cfg, "num_attention_heads", 1) or 1
    n_kv     = getattr(cfg, "num_key_value_heads", n_heads)
    hidden   = getattr(cfg, "hidden_size", 0)
    head_dim = getattr(cfg, "head_dim", hidden // n_heads if hidden else 0)
    if not (n_layers and n_kv and head_dim):
        return 0.0
    return 2 * n_layers * n_kv * head_dim * seq_len * 2 / (1024 ** 2)


def measure_prompt(
    entry: dict,
    model: object,
    tokenizer: object,
    max_new_tokens: int,
    device: str,
    kv_mem_tracker: list[float] | None = None,
    generate_kwargs: dict | None = None,
    analytical_kv: bool = False,
) -> dict:
    """
    Executa um único prompt e retorna dict de métricas.

    kv_mem_tracker: acumula MB de KV quantizado (QuantizedDynamicCache); limpo
    após cada prompt.
    analytical_kv: quando True, usa compute_kv_mb para estimar o KV cache
    analiticamente (recomendado para weight_quant, onde kv_delta inclui
    buffers de dequantização do bitsandbytes).
    generate_kwargs: repassado para measure_throughput (ex: past_key_values).
    """
    reset_peak()
    weights_mb = current_memory_mb()
    throughput, generated, kv_delta = measure_throughput(
        model=model,
        tokenizer=tokenizer,
        prompt=entry["prompt"],
        max_new_tokens=max_new_tokens,
        device=device,
        generate_kwargs=generate_kwargs,
    )
    peak_mb = peak_memory_mb()

    if kv_mem_tracker is not None:
        kv_mb = sum(kv_mem_tracker) if kv_mem_tracker else kv_delta
        kv_mem_tracker.clear()
    elif analytical_kv:
        seq_len = throughput.input_tokens + throughput.output_tokens
        kv_mb   = compute_kv_mb(model, seq_len)
    else:
        kv_mb = kv_delta

    m = RunMetrics(throughput=throughput, prompt_id=entry.get("id", ""))
    m.generated_text = generated
    m.memory.weights_mb = weights_mb
    m.memory.peak_mb = peak_mb
    m.memory.kv_mb = kv_mb
    result = m.to_dict()
    result["kv_theoretical_mb"] = round(
        compute_kv_mb(model, throughput.input_tokens + throughput.output_tokens), 2
    )
    return result


def save_run_json(payload: dict, output_dir: Path, filename: str) -> Path:
    """Salva payload como JSON em output_dir/filename e retorna o caminho."""
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / filename
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    return out_path
