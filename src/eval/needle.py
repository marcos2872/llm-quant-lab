"""
src/eval/needle.py
-------------------
Avaliação Needle-in-a-Haystack.

Para cada entrada de needle.jsonl, constrói um contexto longo com a frase
"needle" no meio, gera resposta e verifica se o "answer" está presente.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from rich.console import Console
from rich.progress import track

console = Console()

_HAYSTACK_PARAGRAPH = (
    "The following is a series of documents covering various topics in science, "
    "history, technology, and culture. Each document contains factual information "
    "about the world and is intended to provide background context for analysis. "
    "Researchers often use long documents to study information retrieval and language "
    "understanding in large language models. The content is neutral and informative. "
)


def _build_context(needle: str, target_tokens: int, tokenizer: Any) -> str:
    """Constrói contexto longo com needle inserido no meio."""
    para_tokens = len(tokenizer.encode(_HAYSTACK_PARAGRAPH))
    n_paragraphs = max(1, target_tokens // para_tokens)
    half = n_paragraphs // 2
    before = _HAYSTACK_PARAGRAPH * half
    after = _HAYSTACK_PARAGRAPH * (n_paragraphs - half)
    return f"{before}\n\n{needle}\n\n{after}"


def _run_single_entry(
    entry: dict,
    model: Any,
    tokenizer: Any,
    max_new_tokens: int,
    device: str,
) -> dict:
    """Executa uma única entrada needle e retorna resultado."""
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
    hit = entry["answer"].lower().strip() in generated.lower()

    return {
        "id": entry["id"],
        "context_tokens_target": entry["context_tokens"],
        "context_tokens_actual": actual_tokens,
        "answer": entry["answer"],
        "generated": generated.strip(),
        "hit": hit,
    }


def eval_needle(
    model: Any,
    tokenizer: Any,
    needle_file: Path = Path("benchmarks/long_context/needle.jsonl"),
    max_new_tokens: int = 128,
    device: str = "cpu",
) -> dict:
    """
    Roda Needle-in-a-Haystack em todas as entradas do arquivo.

    Retorna dict com 'overall_recall', 'by_context_len', 'details'.
    """
    entries = [json.loads(line) for line in needle_file.read_text().splitlines() if line.strip()]
    details: list[dict] = []
    by_context: dict[int, list[bool]] = defaultdict(list)

    for entry in track(entries, description="Needle test..."):
        result = _run_single_entry(entry, model, tokenizer, max_new_tokens, device)
        details.append(result)
        by_context[entry["context_tokens"]].append(result["hit"])

    overall_recall = sum(d["hit"] for d in details) / len(details) if details else 0.0
    by_context_recall = {k: round(sum(v) / len(v), 4) for k, v in by_context.items()}

    console.print(
        f"[cyan]Needle recall:[/cyan] {overall_recall:.2%}  |  por contexto: {by_context_recall}"
    )
    return {
        "overall_recall": round(overall_recall, 4),
        "by_context_len": by_context_recall,
        "details": details,
    }
