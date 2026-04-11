"""
src/runner/weight_quant.py
--------------------------
Runner de quantização de pesos via bitsandbytes.

Itera sobre configurações de bits definidas no config (ou lista fornecida),
roda os prompts e salva métricas por configuração.

Saída: results/raw/weight_quant_<bits>bit_<timestamp>.json
"""

from __future__ import annotations

import copy
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
    return [json.loads(line) for line in prompts_file.read_text().splitlines() if line.strip()]


def _resolve_device(model) -> str:
    try:
        return str(next(model.parameters()).device)
    except StopIteration:
        return "cpu"


def run_weight_quant(
    config_path: Path = Path("configs/weight_quant.yaml"),
    prompts_file: Path = Path("benchmarks/prompts/basic.jsonl"),
    output_dir: Path = Path("results/raw"),
    bits_list: list[int] | None = None,
) -> list[Path]:
    """
    Executa pipeline de quantização de pesos para cada valor de bits.

    Se bits_list não for fornecida, usa o valor único de config.weight_quantization.bits.
    Retorna lista de caminhos dos JSONs gerados.
    """
    base_config = yaml.safe_load(config_path.read_text())
    prompts = _load_prompts(prompts_file)

    wq = base_config.get("weight_quantization", {})
    if not wq.get("enabled", False):
        console.print("[yellow]weight_quantization.enabled=false no config — abortando[/yellow]")
        return []

    bits_to_run = bits_list or [wq.get("bits", 4)]
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_files: list[Path] = []

    for bits in bits_to_run:
        config = copy.deepcopy(base_config)
        config["weight_quantization"]["bits"] = bits

        console.print(f"\n[bold]Weight Quant — {bits} bits — {len(prompts)} prompts[/bold]")
        model, tokenizer = load_model(config)
        device = _resolve_device(model)

        # detecta se o BnB foi silenciado pelo loader (sem CUDA)
        import torch
        bnb_active = torch.cuda.is_available() and config["weight_quantization"].get("enabled", False)

        results: list[dict] = []
        for entry in track(prompts, description=f"INT{bits}..."):
            _reset_peak()
            weights_mb = _current_memory_mb()

            throughput, generated_text = measure_throughput(
                model=model,
                tokenizer=tokenizer,
                prompt=entry["prompt"],
                max_new_tokens=config.get("max_new_tokens", 256),
                device=device,
            )
            peak_mb = _peak_memory_mb()

            m = RunMetrics(throughput=throughput, prompt_id=entry.get("id", ""))
            m.generated_text = generated_text
            m.memory.weights_mb = weights_mb
            m.memory.peak_mb = peak_mb
            m.memory.kv_mb = max(0.0, peak_mb - weights_mb)
            results.append(m.to_dict())

        ts = int(time.time())
        out_path = output_dir / f"weight_quant_{bits}bit_{ts}.json"
        payload = {
            "run_type": "weight_quant",
            "model": config["model"],
            "quant_mode": f"weight_{bits}bit" if bnb_active else f"weight_{bits}bit_fallback_fp32",
            "bits": bits if bnb_active else 32,
            "bnb_active": bnb_active,
            "config": config,
            "results": results,
        }
        out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
        console.print(f"[bold green]✓ Salvo:[/bold green] {out_path}")
        generated_files.append(out_path)

        del model, tokenizer

    return generated_files
