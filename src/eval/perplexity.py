"""
src/eval/perplexity.py
-----------------------
Calcula perplexidade (PPL) e NLL médio de um modelo em um corpus fixo.

Usa sliding window para suportar textos mais longos que o contexto do modelo.
Corpus lido de benchmarks/perplexity/wikitext.jsonl (uma linha por texto).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch
from rich.console import Console
from rich.progress import track

console = Console()


def eval_perplexity(
    model: object,
    tokenizer: object,
    corpus_path: Path = Path("benchmarks/perplexity/wikitext.jsonl"),
    stride: int = 512,
    max_length: int | None = None,
    max_samples: int = 50,
    device: str = "cpu",
) -> dict:
    """
    Calcula perplexidade com sliding window em corpus_path.

    Parâmetros
    ----------
    stride       : sobreposição da janela deslizante (tokens)
    max_length   : tamanho máximo de contexto (padrão: model.config.max_position_embeddings)
    max_samples  : máximo de textos a processar

    Retorna dict com 'perplexity', 'avg_nll', 'n_samples'.
    """
    lines = [json.loads(line)["text"] for line in corpus_path.read_text().splitlines() if line.strip()]
    lines = lines[:max_samples]

    ctx_len = max_length or getattr(model.config, "max_position_embeddings", 2048)
    ctx_len = min(ctx_len, 2048)  # limita para não explodir memória em CPU

    nlls: list[float] = []
    token_counts: list[int] = []

    for text in track(lines, description="Calculando perplexidade..."):
        enc = tokenizer(text, return_tensors="pt").to(device)
        input_ids = enc["input_ids"]
        seq_len = input_ids.shape[1]

        if seq_len < 2:
            continue

        prev_end = 0
        for begin in range(0, seq_len, stride):
            end = min(begin + ctx_len, seq_len)
            target_len = end - prev_end

            chunk = input_ids[:, begin:end]
            target_ids = chunk.clone()
            target_ids[:, :-target_len] = -100  # ignora tokens de overlap

            with torch.no_grad():
                out = model(chunk, labels=target_ids)
                nll = out.loss.item()

            nlls.append(nll * target_len)
            token_counts.append(target_len)
            prev_end = end

            if end == seq_len:
                break

    if not nlls:
        return {"perplexity": float("inf"), "avg_nll": float("inf"), "n_samples": 0}

    total_tokens = sum(token_counts)
    avg_nll = sum(nlls) / total_tokens
    ppl = math.exp(avg_nll)

    console.print(f"[cyan]Perplexidade:[/cyan] {ppl:.2f}  |  NLL: {avg_nll:.4f}  |  tokens: {total_tokens}")
    return {"perplexity": round(ppl, 4), "avg_nll": round(avg_nll, 6), "n_samples": len(lines)}
