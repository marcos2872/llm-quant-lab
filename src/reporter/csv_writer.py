"""
src/reporter/csv_writer.py
---------------------------
Agrega todos os JSONs de results/raw/ em um CSV consolidado.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from rich.console import Console

console = Console()

_COLUMNS = [
    "model", "quant_mode", "bits", "run_type", "method",
    "peak_mem_mb", "kv_mem_mb", "weights_mb",
    "prefill_tok_s", "decode_tok_s", "first_token_latency_s", "total_time_s",
    "input_tokens", "output_tokens",
    "perplexity", "needle_recall", "task_f1",
    "source_file",
]


def _extract_row(payload: dict, source: str) -> dict | None:
    """Extrai uma linha resumo de um payload JSON de run. Retorna None se vazio."""
    results = payload.get("results", [])
    if not results:
        return None

    def avg(key: str) -> float:
        return sum(r.get(key, 0) for r in results) / len(results)

    return {
        "model": payload.get("model", ""),
        "quant_mode": payload.get("quant_mode", ""),
        "bits": payload.get("bits", 16),
        "run_type": payload.get("run_type", ""),
        "method": payload.get("method", "none"),
        "peak_mem_mb": round(avg("peak_mb"), 2),
        "kv_mem_mb": round(avg("kv_mb"), 2),
        "weights_mb": round(avg("weights_mb"), 2),
        "prefill_tok_s": round(avg("prefill_tok_s"), 2),
        "decode_tok_s": round(avg("decode_tok_s"), 2),
        "first_token_latency_s": round(avg("first_token_latency_s"), 4),
        "total_time_s": round(avg("total_time_s"), 4),
        "input_tokens": round(avg("input_tokens"), 1),
        "output_tokens": round(avg("output_tokens"), 1),
        "perplexity": payload.get("perplexity"),
        "needle_recall": payload.get("needle_recall"),
        "task_f1": payload.get("task_f1"),
        "source_file": source,
    }


def aggregate_results(
    raw_dir: Path = Path("results/raw"),
    output_path: Path = Path("results/reports/summary.csv"),
) -> pd.DataFrame:
    """
    Lê todos os JSONs de raw_dir e salva CSV consolidado em output_path.

    Arquivos com erro são registrados e pulados; o total de erros é reportado
    ao final para que o usuário saiba que o CSV pode estar incompleto.
    """
    json_files = sorted(raw_dir.glob("*.json"))
    if not json_files:
        console.print(f"[yellow]Nenhum JSON encontrado em {raw_dir}[/yellow]")
        return pd.DataFrame(columns=_COLUMNS)

    rows: list[dict] = []
    errors: list[str] = []

    for jf in json_files:
        try:
            payload = json.loads(jf.read_text())
            row = _extract_row(payload, jf.name)
            if row:
                rows.append(row)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{jf.name}: {exc}")

    if errors:
        console.print(f"[yellow]⚠ {len(errors)} arquivo(s) com erro (CSV pode estar incompleto):[/yellow]")
        for e in errors:
            console.print(f"  [dim]{e}[/dim]")

    df = pd.DataFrame(rows, columns=_COLUMNS)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    console.print(f"[bold green]✓ CSV salvo:[/bold green] {output_path}  ({len(df)} linhas)")
    return df
