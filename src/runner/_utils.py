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


def measure_prompt(
    entry: dict,
    model: object,
    tokenizer: object,
    max_new_tokens: int,
    device: str,
    kv_mem_tracker: list[float] | None = None,
) -> dict:
    """
    Executa um único prompt e retorna dict de métricas.

    kv_mem_tracker: se fornecido, acumula MB de KV quantizado e é limpo após cada prompt.
    """
    reset_peak()
    weights_mb = current_memory_mb()
    throughput, generated = measure_throughput(
        model=model,
        tokenizer=tokenizer,
        prompt=entry["prompt"],
        max_new_tokens=max_new_tokens,
        device=device,
    )
    peak_mb = peak_memory_mb()

    if kv_mem_tracker is not None:
        kv_mb = sum(kv_mem_tracker) if kv_mem_tracker else max(0.0, peak_mb - weights_mb)
        kv_mem_tracker.clear()
    else:
        kv_mb = max(0.0, peak_mb - weights_mb)

    m = RunMetrics(throughput=throughput, prompt_id=entry.get("id", ""))
    m.generated_text = generated
    m.memory.weights_mb = weights_mb
    m.memory.peak_mb = peak_mb
    m.memory.kv_mb = kv_mb
    return m.to_dict()


def save_run_json(payload: dict, output_dir: Path, filename: str) -> Path:
    """Salva payload como JSON em output_dir/filename e retorna o caminho."""
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / filename
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    return out_path
