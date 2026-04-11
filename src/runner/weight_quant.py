"""
src/runner/weight_quant.py
--------------------------
Runner de quantização de pesos via bitsandbytes.

Itera sobre configurações de bits, roda os prompts e salva métricas por configuração.
Saída: results/raw/weight_quant_<bits>bit_<timestamp>.json
"""

from __future__ import annotations

import copy
import json
import time
from pathlib import Path

import torch
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


def _run_single_bits(
    bits: int,
    base_config: dict,
    prompts: list[dict],
    output_dir: Path,
) -> Path:
    """Executa weight quant para um valor de bits e salva JSON."""
    config = copy.deepcopy(base_config)
    config["weight_quantization"]["bits"] = bits
    bnb_active = torch.cuda.is_available() and config["weight_quantization"].get("enabled", False)

    console.print(f"\n[bold]Weight Quant — {bits} bits — {len(prompts)} prompts[/bold]")
    model, tokenizer = load_model(config)
    device = resolve_device(model)
    max_new_tokens = config.get("max_new_tokens", 256)

    results = [
        _measure_prompt(entry, model, tokenizer, max_new_tokens, device)
        for entry in track(prompts, description=f"INT{bits}...")
    ]
    del model, tokenizer

    out_path = output_dir / f"weight_quant_{bits}bit_{int(time.time())}.json"
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
    return out_path


def run_weight_quant(
    config_path: Path = Path("configs/weight_quant.yaml"),
    prompts_file: Path = Path("benchmarks/prompts/basic.jsonl"),
    output_dir: Path = Path("results/raw"),
    bits_list: list[int] | None = None,
) -> list[Path]:
    """
    Executa weight quant para cada valor de bits em bits_list.

    Se bits_list não for fornecida, usa o valor de config.weight_quantization.bits.
    Retorna lista de caminhos dos JSONs gerados.
    """
    base_config = yaml.safe_load(config_path.read_text())
    wq = base_config.get("weight_quantization", {})
    if not wq.get("enabled", False):
        console.print("[yellow]weight_quantization.enabled=false — abortando[/yellow]")
        return []

    prompts = load_prompts(prompts_file)
    output_dir.mkdir(parents=True, exist_ok=True)
    bits_to_run = bits_list or [wq.get("bits", 4)]
    return [_run_single_bits(b, base_config, prompts, output_dir) for b in bits_to_run]
