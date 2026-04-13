"""
src/runner/context_sweep.py
----------------------------
Benchmark de escalonamento de contexto.

Para cada comprimento de contexto (512, 1k, 2k, 4k tokens), roda baseline
FP16 e todos os métodos de KV quant, registrando memória e throughput.
Permite visualizar como o KV cache cresce com o contexto e quanto cada
método de quantização economiza em cada tamanho.

Saída: results/raw/context_sweep_<timestamp>.json
"""

from __future__ import annotations

import time
from functools import partial
from pathlib import Path

import torch
import yaml
from rich.console import Console
from rich.progress import track

from src.runner._utils import resolve_device, save_run_json
from src.runner.loader import load_model

console = Console()

_HAYSTACK = (
    "Modern AI systems process long text sequences using attention mechanisms. "
    "The key-value cache stores intermediate representations, growing linearly "
    "with sequence length and consuming significant GPU memory during inference. "
    "Quantization methods aim to compress this cache, reducing memory usage "
    "while maintaining generation quality for long-context applications. "
)


def _build_prompt(target_tokens: int, tokenizer: object) -> str:
    """Constrói prompt sintético com aproximadamente target_tokens tokens."""
    para_len = len(tokenizer.encode(_HAYSTACK))
    n = max(1, (target_tokens - 20) // para_len)
    return _HAYSTACK * n + "\nSummary: The key benefit of KV cache quantization is"


def _measure_baseline(
    model: object, tokenizer: object, prompt: str, device: str, max_new_tokens: int
) -> dict:
    """Mede métricas de memória para o baseline FP16."""
    from src.metrics.collector import current_memory_mb, peak_memory_mb, reset_peak
    reset_peak()
    weights_mb = current_memory_mb()
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=8192).to(device)
    actual = inputs["input_ids"].shape[-1]
    with torch.no_grad():
        model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                        temperature=None, top_p=None)
    peak_mb = peak_memory_mb()
    return {"weights_mb": round(weights_mb, 2), "peak_mb": round(peak_mb, 2),
            "kv_mb": round(max(0.0, peak_mb - weights_mb), 2), "actual_tokens": actual}


def _measure_kv(
    model: object, tokenizer: object, prompt: str, device: str,
    max_new_tokens: int, quantize_fn: object, dequantize_fn: object,
) -> dict:
    """Mede métricas de memória com KV cache quantizado (QuantizedDynamicCache)."""
    from src.metrics.collector import current_memory_mb, peak_memory_mb, reset_peak
    from src.quantization.kv_cache import QuantizedDynamicCache
    reset_peak()
    weights_mb = current_memory_mb()
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=8192).to(device)
    actual = inputs["input_ids"].shape[-1]
    tracker: list[float] = []
    cache = QuantizedDynamicCache(quantize_fn, dequantize_fn, tracker)
    with torch.no_grad():
        model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                        temperature=None, top_p=None, past_key_values=cache)
    peak_mb = peak_memory_mb()
    kv_mb = sum(tracker) if tracker else max(0.0, peak_mb - weights_mb)
    return {"weights_mb": round(weights_mb, 2), "peak_mb": round(peak_mb, 2),
            "kv_mb": round(kv_mb, 2), "actual_tokens": actual}


def _get_quant_fns(method: str, bits: int, model_name: str) -> tuple:
    """Retorna (quantize_fn, dequantize_fn) para o método e bits dados."""
    kv_cfg = {"group_size": 64, "outlier_channels": 32, "rotation_seed": 42}
    from src.runner.kv_quant import _get_quant_fns as _gqf
    return _gqf(method, bits, kv_cfg, model_name)


def run_context_sweep(
    config_path: Path = Path("configs/baseline.yaml"),
    context_lengths: list[int] | None = None,
    methods: list[str] | None = None,
    bits: int = 4,
    output_dir: Path = Path("results/raw"),
) -> Path:
    """
    Executa benchmark de escalonamento de contexto.

    Para cada comprimento em context_lengths, roda baseline e cada método
    de KV quant, registrando kv_mb e peak_mb.
    """
    ctx_lens = context_lengths or [512, 1024, 2048, 4096]
    meths = methods or ["baseline", "uniform", "kivi", "turboquant"]
    config = yaml.safe_load(config_path.read_text())
    model_name = config.get("model", "")
    model, tokenizer = load_model(config)
    device = resolve_device(model)
    max_new_tokens = 32
    results: list[dict] = []

    console.print(f"\n[bold]Context Sweep — {len(ctx_lens)} comprimentos × {len(meths)} métodos[/bold]")
    for ctx_len in track(ctx_lens, description="Contextos..."):
        prompt = _build_prompt(ctx_len, tokenizer)
        for method in meths:
            if method == "baseline":
                m = _measure_baseline(model, tokenizer, prompt, device, max_new_tokens)
            else:
                qfn, dfn = _get_quant_fns(method, bits, model_name)
                m = _measure_kv(model, tokenizer, prompt, device, max_new_tokens, qfn, dfn)
            results.append({"context_len": ctx_len, "method": method,
                             "bits": 16 if method == "baseline" else bits, **m})

    return save_run_json(
        payload={"run_type": "context_sweep", "model": model_name, "bits": bits,
                 "context_lengths": ctx_lens, "methods": meths, "results": results},
        output_dir=output_dir,
        filename=f"context_sweep_{int(time.time())}.json",
    )
