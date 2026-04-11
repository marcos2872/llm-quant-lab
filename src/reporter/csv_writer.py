"""
src/reporter/csv_writer.py
---------------------------
Agrega todos os JSONs de results/raw/ em um CSV consolidado.

Colunas do CSV:
  model, quant_mode, bits, run_type, method,
  peak_mem_mb, kv_mem_mb, weights_mb,
  prefill_tok_s, decode_tok_s, first_token_latency_s, total_time_s,
  input_tokens, output_tokens,
  perplexity, needle_recall, task_f1,
  source_file
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


def _extract_row(payload: dict, source: str) -> dict:
    """Extrai uma linha resumo de um payload JSON de run."""
    results = payload.get("results", [])
    if not results:
        return {}

    peak_mem = sum(r.get("peak_mb", 0) for r in results) / len(results)
    kv_mem = sum(r.get("kv_mb", 0) for r in results) / len(results)
    weights_mem = sum(r.get("weights_mb", 0) for r in results) / len(results)
    prefill = sum(r.get("prefill_tok_s", 0) for r in results) / len(results)
    decode = sum(r.get("decode_tok_s", 0) for r in results) / len(results)
    lat = sum(r.get("first_token_latency_s", 0) for r in results) / len(results)
    total_t = sum(r.get("total_time_s", 0) for r in results) / len(results)
    in_tok = sum(r.get("input_tokens", 0) for r in results) / len(results)
    out_tok = sum(r.get("output_tokens", 0) for r in results) / len(results)

    return {
        "model": payload.get("model", ""),
        "quant_mode": payload.get("quant_mode", ""),
        "bits": payload.get("bits", 16),
        "run_type": payload.get("run_type", ""),
        "method": payload.get("method", "none"),
        "peak_mem_mb": round(peak_mem, 2),
        "kv_mem_mb": round(kv_mem, 2),
        "weights_mb": round(weights_mem, 2),
        "prefill_tok_s": round(prefill, 2),
        "decode_tok_s": round(decode, 2),
        "first_token_latency_s": round(lat, 4),
        "total_time_s": round(total_t, 4),
        "input_tokens": round(in_tok, 1),
        "output_tokens": round(out_tok, 1),
        "perplexity": payload.get("perplexity", None),
        "needle_recall": payload.get("needle_recall", None),
        "task_f1": payload.get("task_f1", None),
        "source_file": source,
    }


def aggregate_results(
    raw_dir: Path = Path("results/raw"),
    output_path: Path = Path("results/reports/summary.csv"),
) -> pd.DataFrame:
    """
    Lê todos os JSONs de raw_dir e salva CSV consolidado em output_path.

    Retorna o DataFrame gerado.
    """
    json_files = sorted(raw_dir.glob("*.json"))
    if not json_files:
        console.print(f"[yellow]Nenhum JSON encontrado em {raw_dir}[/yellow]")
        return pd.DataFrame(columns=_COLUMNS)

    rows: list[dict] = []
    for jf in json_files:
        try:
            payload = json.loads(jf.read_text())
            row = _extract_row(payload, jf.name)
            if row:
                rows.append(row)
        except Exception as exc:  # noqa: BLE001
            console.print(f"[yellow]Aviso: erro ao ler {jf.name}: {exc}[/yellow]")

    df = pd.DataFrame(rows, columns=_COLUMNS)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    console.print(f"[bold green]✓ CSV salvo:[/bold green] {output_path}  ({len(df)} linhas)")
    return df
