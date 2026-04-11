"""
src/eval/task_score.py
-----------------------
Avalia qualidade de geração em prompts QA curtos.

Métricas: Exact Match e Token F1 (estilo SQuAD).
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from rich.console import Console
from rich.progress import track

console = Console()


def _normalize(text: str) -> str:
    """Normaliza texto: minúsculo, sem pontuação extra."""
    text = re.sub(r"[^\w\s]", " ", text.lower().strip())
    return re.sub(r"\s+", " ", text).strip()


def _token_f1(prediction: str, reference: str) -> float:
    """Calcula F1 baseado em sobreposição de tokens."""
    pred = Counter(_normalize(prediction).split())
    ref = Counter(_normalize(reference).split())
    if not pred or not ref:
        return float(pred == ref)
    common = pred & ref
    n_common = sum(common.values())
    if n_common == 0:
        return 0.0
    precision = n_common / sum(pred.values())
    recall = n_common / sum(ref.values())
    return 2 * precision * recall / (precision + recall)


def _score_prompt(
    entry: dict,
    model: Any,
    tokenizer: Any,
    max_new_tokens: int,
    device: str,
) -> dict:
    """Executa um prompt e retorna scores."""
    inputs = tokenizer(entry["prompt"], return_tensors="pt").to(device)
    with torch.no_grad():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
        )
    input_len = inputs["input_ids"].shape[-1]
    generated = tokenizer.decode(out_ids[0][input_len:], skip_special_tokens=True)
    return {
        "id": entry["id"],
        "reference": entry["reference"],
        "generated": generated.strip(),
        "f1": round(_token_f1(generated, entry["reference"]), 4),
        "exact_match": _normalize(entry["reference"]) in _normalize(generated),
    }


def eval_task_score(
    model: Any,
    tokenizer: Any,
    prompts_file: Path = Path("benchmarks/prompts/basic.jsonl"),
    max_new_tokens: int = 64,
    device: str = "cpu",
) -> dict:
    """
    Avalia o modelo em todos os prompts de prompts_file.

    Retorna dict com 'avg_f1', 'exact_match_rate', 'scores'.
    """
    entries = [json.loads(line) for line in prompts_file.read_text().splitlines() if line.strip()]
    scores = [
        _score_prompt(entry, model, tokenizer, max_new_tokens, device)
        for entry in track(entries, description="Task score...")
    ]

    avg_f1 = sum(s["f1"] for s in scores) / len(scores) if scores else 0.0
    em_rate = sum(s["exact_match"] for s in scores) / len(scores) if scores else 0.0
    console.print(f"[cyan]Task score:[/cyan] F1={avg_f1:.4f}  |  EM={em_rate:.2%}  |  n={len(scores)}")
    return {"avg_f1": round(avg_f1, 4), "exact_match_rate": round(em_rate, 4), "scores": scores}
