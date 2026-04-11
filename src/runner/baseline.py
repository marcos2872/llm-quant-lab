"""
src/runner/baseline.py
----------------------
Runner baseline: carrega o modelo em FP16 sem quantização,
roda os prompts e coleta métricas de memória e throughput.

Saída: results/raw/baseline_<timestamp>.json
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import yaml
from rich.console import Console
from rich.progress import track

from src.metrics.collector import (
    RunMetrics,
    current_memory_mb,
    measure_throughput,
    peak_memory_mb,
    reset_peak,
)
from src.runner._utils import load_prompts, resolve_device
from src.runner.loader import load_model

console = Console()


def _measure_prompt(
    entry: dict,
    model: object,
    tokenizer: object,
    max_new_tokens: int,
    device: str,
) -> dict:
    """Executa um único prompt e retorna dict de métricas."""
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

    m = RunMetrics(throughput=throughput, prompt_id=entry.get("id", ""))
    m.generated_text = generated
    m.memory.weights_mb = weights_mb
    m.memory.peak_mb = peak_mb
    m.memory.kv_mb = max(0.0, peak_mb - weights_mb)
    return m.to_dict()


def run_baseline(
    config_path: Path = Path("configs/baseline.yaml"),
    prompts_file: Path = Path("benchmarks/prompts/basic.jsonl"),
    output_dir: Path = Path("results/raw"),
) -> Path:
    """Executa pipeline baseline e salva JSON com resultados."""
    config = yaml.safe_load(config_path.read_text())
    prompts = load_prompts(prompts_file)
    model, tokenizer = load_model(config)
    device = resolve_device(model)
    max_new_tokens = config.get("max_new_tokens", 256)

    console.print(f"\n[bold green]Baseline — {len(prompts)} prompts[/bold green]")

    results = [
        _measure_prompt(entry, model, tokenizer, max_new_tokens, device)
        for entry in track(prompts, description="Rodando prompts...")
    ]

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"baseline_{int(time.time())}.json"
    payload = {
        "run_type": "baseline",
        "model": config["model"],
        "quant_mode": "baseline",
        "bits": 16,
        "config": config,
        "results": results,
    }
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    console.print(f"[bold green]✓ Salvo:[/bold green] {out_path}")
    return out_path
