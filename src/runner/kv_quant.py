"""
src/runner/kv_quant.py
-----------------------
Runner de quantização de KV cache.

Instala hooks nos attention layers antes de cada geração,
coleta métricas de memória e throughput, remove hooks ao final.

Saída: results/raw/kv_quant_<method>_<bits>bit_<timestamp>.json
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from functools import partial
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
from src.quantization.kv_hooks import install_kv_hooks, remove_kv_hooks
from src.runner._utils import load_prompts, resolve_device
from src.runner.loader import load_model

console = Console()


def _get_quant_fns(method: str, bits: int, kv_cfg: dict) -> tuple[Callable, Callable]:
    """Retorna (quantize_fn, dequantize_fn) para o método configurado."""
    if method == "uniform":
        from src.quantization.methods.uniform import dequantize_uniform, quantize_uniform
        return partial(quantize_uniform, bits=bits), dequantize_uniform

    if method == "kivi":
        from src.quantization.methods.kivi import dequantize_kivi, quantize_kivi
        group_size = kv_cfg.get("group_size", 64)
        return partial(quantize_kivi, bits=bits, group_size=group_size), dequantize_kivi

    if method == "turboquant":
        from src.quantization.methods.turboquant import dequantize_turboquant, quantize_turboquant
        return (
            partial(
                quantize_turboquant,
                bits=bits,
                group_size=kv_cfg.get("group_size", 64),
                outlier_channels=kv_cfg.get("outlier_channels", 32),
                rotation_seed=kv_cfg.get("rotation_seed", 42),
            ),
            dequantize_turboquant,
        )

    raise ValueError(f"Método desconhecido: {method!r}. Use uniform | kivi | turboquant")


def _measure_prompt(
    entry: dict,
    model: object,
    tokenizer: object,
    kv_mem_tracker: list[float],
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
    kv_mb = sum(kv_mem_tracker) if kv_mem_tracker else max(0.0, peak_mb - weights_mb)
    kv_mem_tracker.clear()

    m = RunMetrics(throughput=throughput, prompt_id=entry.get("id", ""))
    m.generated_text = generated
    m.memory.weights_mb = weights_mb
    m.memory.peak_mb = peak_mb
    m.memory.kv_mb = kv_mb
    return m.to_dict()


def run_kv_quant(
    config_path: Path = Path("configs/kv_quant.yaml"),
    prompts_file: Path = Path("benchmarks/prompts/basic.jsonl"),
    output_dir: Path = Path("results/raw"),
    config_override: dict | None = None,
) -> Path:
    """
    Executa KV cache quantization.

    Se config_override for fornecido, ignora config_path.
    Garante remoção dos hooks mesmo em caso de exceção (try/finally).
    """
    config = config_override if config_override is not None else yaml.safe_load(config_path.read_text())
    kv_cfg = config.get("kv_quantization", {})

    if not kv_cfg.get("enabled", False):
        console.print("[yellow]kv_quantization.enabled=false — abortando[/yellow]")
        return Path()

    method = kv_cfg.get("method", "uniform")
    bits = kv_cfg.get("bits", 4)
    prompts = load_prompts(prompts_file)

    console.print(f"\n[bold]KV Quant — método={method}, bits={bits} — {len(prompts)} prompts[/bold]")

    base_config = {k: v for k, v in config.items() if k != "kv_quantization"}
    model, tokenizer = load_model(base_config)
    device = resolve_device(model)
    max_new_tokens = config.get("max_new_tokens", 256)

    quantize_fn, dequantize_fn = _get_quant_fns(method, bits, kv_cfg)
    handles, kv_mem_tracker = install_kv_hooks(model, quantize_fn, dequantize_fn)
    console.print(f"[cyan]Hooks instalados:[/cyan] {len(handles)} camadas")

    results: list[dict] = []
    try:
        for entry in track(prompts, description=f"{method} INT{bits}..."):
            results.append(
                _measure_prompt(entry, model, tokenizer, kv_mem_tracker, max_new_tokens, device)
            )
    finally:
        remove_kv_hooks(handles)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"kv_quant_{method}_{bits}bit_{int(time.time())}.json"
    payload = {
        "run_type": "kv_quant",
        "model": config["model"],
        "quant_mode": f"kv_{method}_{bits}bit",
        "bits": bits,
        "method": method,
        "config": config,
        "results": results,
    }
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    console.print(f"[bold green]✓ Salvo:[/bold green] {out_path}")
    return out_path
