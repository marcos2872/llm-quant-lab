"""
src/runner/_utils.py
--------------------
Utilitários compartilhados entre os runners de inferência.

Centraliza funções duplicadas que existiam em baseline, kv_quant e weight_quant.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch


def load_prompts(prompts_file: Path) -> list[dict]:
    """Lê arquivo JSONL de prompts e retorna lista de dicts."""
    return [
        json.loads(line)
        for line in prompts_file.read_text().splitlines()
        if line.strip()
    ]


def resolve_device(model: torch.nn.Module) -> str:
    """Obtém string do device do primeiro parâmetro do modelo."""
    try:
        return str(next(model.parameters()).device)
    except StopIteration:
        return "cpu"
