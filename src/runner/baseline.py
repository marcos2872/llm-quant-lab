"""
src/runner/baseline.py
----------------------
Runner baseline: carrega o modelo em FP16 sem quantização,
roda os prompts e coleta métricas de memória e throughput.

Saída: results/raw/baseline_<timestamp>.json
"""

from __future__ import annotations

import time
from pathlib import Path

import yaml
from rich.console import Console
from rich.progress import track

from src.runner._utils import load_prompts, measure_prompt, resolve_device, save_run_json
from src.runner.loader import load_model

console = Console()


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
        measure_prompt(entry, model, tokenizer, max_new_tokens, device, analytical_kv=True)
        for entry in track(prompts, description="Rodando prompts...")
    ]

    out_path = save_run_json(
        payload={
            "run_type": "baseline",
            "model": config["model"],
            "quant_mode": "baseline",
            "bits": 16,
            "prompts_file": str(prompts_file),
            "config": config,
            "results": results,
        },
        output_dir=output_dir,
        filename=f"baseline_{int(time.time())}.json",
    )
    console.print(f"[bold green]✓ Salvo:[/bold green] {out_path}")
    return out_path
