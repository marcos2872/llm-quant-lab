"""
src/runner/weight_quant.py
--------------------------
Runner de quantização de pesos via bitsandbytes.

Saída: results/raw/weight_quant_<bits>bit_<timestamp>.json
"""

from __future__ import annotations

import copy
import time
from pathlib import Path

import torch
import yaml
from rich.console import Console
from rich.progress import track

from src.runner._utils import load_prompts, measure_prompt, resolve_device, save_run_json
from src.runner.loader import load_model

console = Console()


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
        measure_prompt(entry, model, tokenizer, max_new_tokens, device)
        for entry in track(prompts, description=f"INT{bits}...")
    ]
    del model, tokenizer

    out_path = save_run_json(
        payload={
            "run_type": "weight_quant",
            "model": config["model"],
            "quant_mode": f"weight_{bits}bit" if bnb_active else f"weight_{bits}bit_fallback_fp32",
            "bits": bits if bnb_active else 32,
            "bnb_active": bnb_active,
            "config": config,
            "results": results,
        },
        output_dir=output_dir,
        filename=f"weight_quant_{bits}bit_{int(time.time())}.json",
    )
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
    """
    base_config = yaml.safe_load(config_path.read_text())
    wq = base_config.get("weight_quantization", {})
    if not wq.get("enabled", False):
        console.print("[yellow]weight_quantization.enabled=false — abortando[/yellow]")
        return []

    prompts = load_prompts(prompts_file)
    bits_to_run = bits_list or [wq.get("bits", 4)]
    return [_run_single_bits(b, base_config, prompts, output_dir) for b in bits_to_run]
