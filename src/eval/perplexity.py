"""
src/eval/perplexity.py
-----------------------
Calcula perplexidade (PPL) e NLL médio com sliding window.

Corpus: benchmarks/perplexity/wikitext.jsonl
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch
from rich.console import Console
from rich.progress import track

console = Console()


def _process_text(
    text: str,
    model: Any,
    tokenizer: Any,
    ctx_len: int,
    stride: int,
    device: str,
) -> tuple[float, int]:
    """Calcula NLL acumulada e total de tokens para um texto."""
    enc = tokenizer(text, return_tensors="pt").to(device)
    input_ids = enc["input_ids"]
    seq_len = input_ids.shape[1]
    if seq_len < 2:
        return 0.0, 0

    total_nll = 0.0
    total_tokens = 0
    prev_end = 0

    for begin in range(0, seq_len, stride):
        end = min(begin + ctx_len, seq_len)
        target_len = end - prev_end
        chunk = input_ids[:, begin:end]
        target_ids = chunk.clone()
        target_ids[:, :-target_len] = -100

        with torch.no_grad():
            out = model(chunk, labels=target_ids)

        total_nll += out.loss.item() * target_len
        total_tokens += target_len
        prev_end = end
        if end == seq_len:
            break

    return total_nll, total_tokens


def eval_perplexity(
    model: Any,
    tokenizer: Any,
    corpus_path: Path = Path("benchmarks/perplexity/wikitext.jsonl"),
    stride: int = 512,
    max_length: int | None = None,
    max_samples: int = 50,
    device: str = "cpu",
    quantize_fn: Any = None,
    dequantize_fn: Any = None,
) -> dict:
    """
    Calcula perplexidade com sliding window em corpus_path.

    quantize_fn / dequantize_fn: quando fornecidos, instala hooks em k_proj/
    v_proj de cada camada de atenção para simular o impacto da quantização de
    KV cache. Os hooks interceptam as projeções antes da atenção, forçando que
    cada forward pass use K/V quantizados+dequantizados — PPL resultante
    reflete a degradação real do método.
    Retorna dict com 'perplexity', 'avg_nll', 'n_samples'.
    """
    lines = [
        json.loads(line)["text"]
        for line in corpus_path.read_text().splitlines()
        if line.strip()
    ][:max_samples]

    ctx_len = min(
        max_length or getattr(model.config, "max_position_embeddings", 2048),
        2048,
    )

    handles: list = []
    if quantize_fn is not None and dequantize_fn is not None:
        from src.quantization.kv_hooks import install_kv_proj_hooks
        handles, _ = install_kv_proj_hooks(model, quantize_fn, dequantize_fn)

    total_nll, total_tokens = 0.0, 0
    try:
        for text in track(lines, description="Calculando perplexidade..."):
            nll, tokens = _process_text(text, model, tokenizer, ctx_len, stride, device)
            total_nll += nll
            total_tokens += tokens
    finally:
        if handles:
            from src.quantization.kv_hooks import remove_kv_hooks
            remove_kv_hooks(handles)

    if total_tokens == 0:
        return {"perplexity": float("inf"), "avg_nll": float("inf"), "n_samples": 0}

    avg_nll = total_nll / total_tokens
    ppl = math.exp(avg_nll)
    console.print(f"[cyan]Perplexidade:[/cyan] {ppl:.2f}  |  NLL: {avg_nll:.4f}  |  tokens: {total_tokens}")
    return {"perplexity": round(ppl, 4), "avg_nll": round(avg_nll, 6), "n_samples": len(lines)}
