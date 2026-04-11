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
from src.quantization.kv_hooks import install_kv_hooks, remove_kv_hooks
from src.runner.loader import load_model

console = Console()

# ── fábrica de funções de quantização por método ──────────────────────────────

def _get_quant_fns(method: str, bits: int, kv_cfg: dict) -> tuple[Callable, Callable]:
    """Retorna (quantize_fn, dequantize_fn) para o método configurado."""
    if method == "uniform":
        from src.quantization.methods.uniform import dequantize_uniform, quantize_uniform

        def q(t):
            return quantize_uniform(t, bits=bits)

        return q, dequantize_uniform

    if method == "kivi":
        from src.quantization.methods.kivi import dequantize_kivi, quantize_kivi

        group_size = kv_cfg.get("group_size", 64)

        def q(t):
            return quantize_kivi(t, bits=bits, group_size=group_size)

        return q, dequantize_kivi

    if method == "turboquant":
        from src.quantization.methods.turboquant import (
            dequantize_turboquant,
            quantize_turboquant,
        )

        group_size = kv_cfg.get("group_size", 64)
        outlier_channels = kv_cfg.get("outlier_channels", 32)
        rotation_seed = kv_cfg.get("rotation_seed", 42)

        def q(t):
            return quantize_turboquant(
                t,
                bits=bits,
                group_size=group_size,
                outlier_channels=outlier_channels,
                rotation_seed=rotation_seed,
            )

        return q, dequantize_turboquant

    raise ValueError(f"Método de KV quant desconhecido: {method!r}. Use uniform | kivi | turboquant")


def _load_prompts(prompts_file: Path) -> list[dict]:
    return [json.loads(line) for line in prompts_file.read_text().splitlines() if line.strip()]


def _resolve_device(model) -> str:
    try:
        return str(next(model.parameters()).device)
    except StopIteration:
        return "cpu"


def run_kv_quant(
    config_path: Path = Path("configs/kv_quant.yaml"),
    prompts_file: Path = Path("benchmarks/prompts/basic.jsonl"),
    output_dir: Path = Path("results/raw"),
) -> Path:
    """
    Executa pipeline de KV cache quantization.

    Carrega modelo baseline (sem weight quant), instala KV hooks,
    roda prompts e salva resultados.
    """
    config = yaml.safe_load(config_path.read_text())
    kv_cfg = config.get("kv_quantization", {})

    if not kv_cfg.get("enabled", False):
        console.print("[yellow]kv_quantization.enabled=false no config — abortando[/yellow]")
        return Path()

    method = kv_cfg.get("method", "uniform")
    bits = kv_cfg.get("bits", 4)
    prompts = _load_prompts(prompts_file)

    console.print(f"\n[bold]KV Quant — método={method}, bits={bits} — {len(prompts)} prompts[/bold]")

    # carrega modelo sem weight quant
    base_config = {k: v for k, v in config.items() if k != "kv_quantization"}
    model, tokenizer = load_model(base_config)
    device = _resolve_device(model)

    quantize_fn, dequantize_fn = _get_quant_fns(method, bits, kv_cfg)
    handles, kv_mem_tracker = install_kv_hooks(model, quantize_fn, dequantize_fn)
    console.print(f"[cyan]Hooks instalados:[/cyan] {len(handles)} camadas")

    results: list[dict] = []
    for entry in track(prompts, description=f"{method} INT{bits}..."):
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

        kv_mb_run = sum(kv_mem_tracker) if kv_mem_tracker else max(0.0, peak_mb - weights_mb)
        kv_mem_tracker.clear()

        m = RunMetrics(throughput=throughput, prompt_id=entry.get("id", ""))
        m.generated_text = generated_text
        m.memory.weights_mb = weights_mb
        m.memory.peak_mb = peak_mb
        m.memory.kv_mb = kv_mb_run
        results.append(m.to_dict())

    remove_kv_hooks(handles)

    output_dir.mkdir(parents=True, exist_ok=True)
    ts = int(time.time())
    out_path = output_dir / f"kv_quant_{method}_{bits}bit_{ts}.json"
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
