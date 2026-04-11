"""
src/runner/baseline.py
----------------------
Runner baseline: carrega o modelo em FP16 sem nenhuma quantização,
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
    _current_memory_mb,
    _peak_memory_mb,
    _reset_peak,
    measure_throughput,
)
from src.runner.loader import load_model

console = Console()


def _load_prompts(prompts_file: Path) -> list[dict]:
    """Lê arquivo JSONL de prompts."""
    return [json.loads(line) for line in prompts_file.read_text().splitlines() if line.strip()]


def _resolve_device(model) -> str:
    """Obtém device do primeiro parâmetro do modelo."""
    try:
        return str(next(model.parameters()).device)
    except StopIteration:
        return "cpu"


def run_baseline(
    config_path: Path = Path("configs/baseline.yaml"),
    prompts_file: Path = Path("benchmarks/prompts/basic.jsonl"),
    output_dir: Path = Path("results/raw"),
) -> Path:
    """
    Executa pipeline baseline e salva JSON com resultados.

    Retorna caminho do arquivo JSON gerado.
    """
    config = yaml.safe_load(config_path.read_text())
    prompts = _load_prompts(prompts_file)

    model, tokenizer = load_model(config)
    device = _resolve_device(model)

    console.print(f"\n[bold green]Baseline — {len(prompts)} prompts[/bold green]")

    results: list[dict] = []
    for entry in track(prompts, description="Rodando prompts..."):
        _reset_peak()
        weights_mb = _current_memory_mb()

        throughput, generated = measure_throughput(
            model=model,
            tokenizer=tokenizer,
            prompt=entry["prompt"],
            max_new_tokens=config.get("max_new_tokens", 256),
            device=device,
        )
        peak_mb = _peak_memory_mb()

        metrics = RunMetrics(throughput=throughput, prompt_id=entry.get("id", ""))
        metrics.generated_text = generated
        metrics.memory.weights_mb = weights_mb
        metrics.memory.peak_mb = peak_mb
        metrics.memory.kv_mb = max(0.0, peak_mb - weights_mb)

        results.append(metrics.to_dict())

    output_dir.mkdir(parents=True, exist_ok=True)
    ts = int(time.time())
    out_path = output_dir / f"baseline_{ts}.json"

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
