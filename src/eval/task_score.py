"""
src/eval/task_score.py
-----------------------
Avalia qualidade de geração em um conjunto fixo de prompts QA curtos.

Métricas:
  - Exact Match (EM): resposta gerada contém exatamente o texto de referência
  - Token F1: sobreposição de tokens entre gerada e referência (estilo SQuAD)

Lê benchmarks/prompts/basic.jsonl com campos: id, prompt, reference.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import torch
from rich.console import Console
from rich.progress import track

console = Console()


def _normalize(text: str) -> str:
    """Normaliza texto para comparação: minúsculo, sem pontuação extra."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _token_f1(prediction: str, reference: str) -> float:
    """Calcula F1 baseado em sobreposição de tokens (estilo SQuAD)."""
    pred_tokens = Counter(_normalize(prediction).split())
    ref_tokens = Counter(_normalize(reference).split())

    if not pred_tokens or not ref_tokens:
        return float(pred_tokens == ref_tokens)

    common = pred_tokens & ref_tokens
    n_common = sum(common.values())

    if n_common == 0:
        return 0.0

    precision = n_common / sum(pred_tokens.values())
    recall = n_common / sum(ref_tokens.values())
    return 2 * precision * recall / (precision + recall)


def _exact_match(prediction: str, reference: str) -> bool:
    """Verifica se referência está contida na predição (case-insensitive)."""
    return _normalize(reference) in _normalize(prediction)


def eval_task_score(
    model: object,
    tokenizer: object,
    prompts_file: Path = Path("benchmarks/prompts/basic.jsonl"),
    max_new_tokens: int = 64,
    device: str = "cpu",
) -> dict:
    """
    Avalia o modelo em todos os prompts de prompts_file.

    Retorna dict com:
      - avg_f1: F1 token médio
      - exact_match_rate: taxa de exact match
      - scores: lista de resultados por prompt
    """
    entries = [json.loads(line) for line in prompts_file.read_text().splitlines() if line.strip()]

    scores: list[dict] = []

    for entry in track(entries, description="Task score..."):
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

        f1 = _token_f1(generated, entry["reference"])
        em = _exact_match(generated, entry["reference"])

        scores.append({
            "id": entry["id"],
            "reference": entry["reference"],
            "generated": generated.strip(),
            "f1": round(f1, 4),
            "exact_match": em,
        })

    avg_f1 = sum(s["f1"] for s in scores) / len(scores) if scores else 0.0
    em_rate = sum(s["exact_match"] for s in scores) / len(scores) if scores else 0.0

    console.print(
        f"[cyan]Task score:[/cyan] F1={avg_f1:.4f}  |  EM={em_rate:.2%}  |  n={len(scores)}"
    )
    return {
        "avg_f1": round(avg_f1, 4),
        "exact_match_rate": round(em_rate, 4),
        "scores": scores,
    }
