"""
src/runner/kv_quant.py
-----------------------
Runner de quantização de KV cache com QuantizedDynamicCache.

Armazena o KV do prefill em formato comprimido (redução real de memória GPU).
Tokens do decode ficam em buffer FP16 pequeno sem re-quantização.

Saída: results/raw/kv_quant_<method>_<bits>bit_<timestamp>.json
"""

from __future__ import annotations

import time
from collections.abc import Callable
from functools import partial
from pathlib import Path

import yaml
from rich.console import Console
from rich.progress import track

from src.runner._utils import load_prompts, measure_prompt, resolve_device, save_run_json
from src.runner.loader import load_model

console = Console()


def _get_quant_fns(
    method: str, bits: int, kv_cfg: dict, model_name: str = ""
) -> tuple[Callable, Callable]:
    """Retorna (quantize_fn, dequantize_fn) para o método configurado."""
    if method == "uniform":
        from src.quantization.methods.uniform import dequantize_uniform, quantize_uniform
        return partial(quantize_uniform, bits=bits), dequantize_uniform

    if method == "kivi":
        from src.quantization.methods.kivi import dequantize_kivi, quantize_kivi
        return partial(quantize_kivi, bits=bits, group_size=kv_cfg.get("group_size", 64)), dequantize_kivi

    if method == "turboquant":
        from src.quantization.methods.turboquant import dequantize_turboquant, quantize_turboquant
        return (
            partial(
                quantize_turboquant,
                bits=bits,
                group_size=kv_cfg.get("group_size", 64),
                outlier_channels=kv_cfg.get("outlier_channels", 32),
                rotation_seed=kv_cfg.get("rotation_seed", 42),
                model_id=model_name,
            ),
            dequantize_turboquant,
        )

    raise ValueError(f"Método desconhecido: {method!r}. Use uniform | kivi | turboquant")


def _run_with_cache(
    model: object,
    tokenizer: object,
    prompts: list[dict],
    quantize_fn: Callable,
    dequantize_fn: Callable,
    max_new_tokens: int,
    device: str,
    label: str,
) -> list[dict]:
    """
    Roda prompts usando QuantizedDynamicCache por prompt.

    Cada prompt recebe um cache fresco; o tracker mede o MB quantizado
    do prefill, refletindo a redução real de memória GPU.
    """
    from src.quantization.kv_cache import QuantizedDynamicCache

    results: list[dict] = []
    for entry in track(prompts, description=label):
        tracker: list[float] = []
        cache = QuantizedDynamicCache(quantize_fn, dequantize_fn, tracker)
        results.append(
            measure_prompt(
                entry, model, tokenizer, max_new_tokens, device,
                kv_mem_tracker=tracker,
                generate_kwargs={"past_key_values": cache},
            )
        )
    return results


def run_kv_quant(
    config_path: Path = Path("configs/kv_quant.yaml"),
    prompts_file: Path = Path("benchmarks/prompts/basic.jsonl"),
    output_dir: Path = Path("results/raw"),
    config_override: dict | None = None,
) -> Path | None:
    """
    Executa KV cache quantization com cache comprimido.

    Retorna None se kv_quantization.enabled=false.
    """
    config = config_override if config_override is not None else yaml.safe_load(config_path.read_text())
    kv_cfg = config.get("kv_quantization", {})

    if not kv_cfg.get("enabled", False):
        console.print("[yellow]kv_quantization.enabled=false — abortando[/yellow]")
        return None

    method = kv_cfg.get("method", "uniform")
    bits = kv_cfg.get("bits", 4)
    model_name = config.get("model", "")
    prompts = load_prompts(prompts_file)
    console.print(f"\n[bold]KV Quant — método={method}, bits={bits} — {len(prompts)} prompts[/bold]")

    base_config = {k: v for k, v in config.items() if k != "kv_quantization"}
    model, tokenizer = load_model(base_config)
    device = resolve_device(model)
    max_new_tokens = config.get("max_new_tokens", 256)

    quantize_fn, dequantize_fn = _get_quant_fns(method, bits, kv_cfg, model_name)
    results = _run_with_cache(
        model, tokenizer, prompts, quantize_fn, dequantize_fn,
        max_new_tokens, device, f"{method} INT{bits}...",
    )

    out_path = save_run_json(
        payload={
            "run_type": "kv_quant", "model": model_name,
            "quant_mode": f"kv_{method}_{bits}bit", "bits": bits,
            "method": method, "config": config, "results": results,
        },
        output_dir=output_dir,
        filename=f"kv_quant_{method}_{bits}bit_{int(time.time())}.json",
    )
    console.print(f"[bold green]✓ Salvo:[/bold green] {out_path}")
    return out_path
