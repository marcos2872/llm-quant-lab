"""
src/eval/needle.py
-------------------
Avaliação Needle-in-a-Haystack.

Para cada entrada em needle.jsonl:
  1. Constrói contexto longo preenchendo com texto repetitivo até atingir
     aproximadamente context_tokens tokens.
  2. Insere a frase "needle" no meio do contexto.
  3. Adiciona a pergunta ao final.
  4. Gera resposta e verifica se o "answer" esperado está presente.

Retorna recall_rate por tamanho de contexto e score global.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import torch
from rich.console import Console
from rich.progress import track

console = Console()

# Texto de haystack neutro e repetitivo
_HAYSTACK_PARAGRAPH = (
    "The following is a series of documents covering various topics in science, "
    "history, technology, and culture. Each document contains factual information "
    "about the world and is intended to provide background context for analysis. "
    "Researchers often use long documents to study information retrieval and language "
    "understanding in large language models. The content is neutral and informative. "
)


def _build_context(needle: str, target_tokens: int, tokenizer: object) -> str:
    """Constrói contexto longo com needle inserido no meio."""
    # Estima tamanho do parágrafo em tokens
    para_tokens = len(tokenizer.encode(_HAYSTACK_PARAGRAPH))
    n_paragraphs = max(1, target_tokens // para_tokens)

    half = n_paragraphs // 2
    before = _HAYSTACK_PARAGRAPH * half
    after = _HAYSTACK_PARAGRAPH * (n_paragraphs - half)

    return f"{before}\n\n{needle}\n\n{after}"


def _contains_answer(generated: str, answer: str) -> bool:
    """Verifica se a resposta gerada contém o trecho esperado (case-insensitive)."""
    return answer.lower().strip() in generated.lower()


def eval_needle(
    model: object,
    tokenizer: object,
    needle_file: Path = Path("benchmarks/long_context/needle.jsonl"),
    max_new_tokens: int = 128,
    device: str = "cpu",
) -> dict:
    """
    Roda Needle-in-a-Haystack em todas as entradas do arquivo.

    Retorna dict com:
      - overall_recall: taxa de acerto global
      - by_context_len: dict context_tokens → recall
      - details: lista com resultado por entrada
    """
    entries = [json.loads(line) for line in needle_file.read_text().splitlines() if line.strip()]

    details: list[dict] = []
    by_context: dict[int, list[bool]] = defaultdict(list)

    for entry in track(entries, description="Needle test..."):
        context = _build_context(entry["needle"], entry["context_tokens"], tokenizer)
        prompt = f"{context}\n\nQuestion: {entry['question']}\nAnswer:"

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)
        actual_tokens = inputs["input_ids"].shape[-1]

        with torch.no_grad():
            out_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
            )
        generated = tokenizer.decode(out_ids[0][actual_tokens:], skip_special_tokens=True)
        hit = _contains_answer(generated, entry["answer"])

        details.append({
            "id": entry["id"],
            "context_tokens_target": entry["context_tokens"],
            "context_tokens_actual": actual_tokens,
            "answer": entry["answer"],
            "generated": generated.strip(),
            "hit": hit,
        })
        by_context[entry["context_tokens"]].append(hit)

    overall_recall = sum(d["hit"] for d in details) / len(details) if details else 0.0
    by_context_recall = {k: round(sum(v) / len(v), 4) for k, v in by_context.items()}

    console.print(
        f"[cyan]Needle recall:[/cyan] {overall_recall:.2%}  |  "
        f"por contexto: {by_context_recall}"
    )
    return {
        "overall_recall": round(overall_recall, 4),
        "by_context_len": by_context_recall,
        "details": details,
    }
