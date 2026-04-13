"""
src/reporter/context_plots.py
------------------------------
Gráfico de escalonamento de contexto a partir de context_sweep_*.json.

context_scaling.png — duas subplots:
  1. Tokens de contexto × KV Cache (MB)  — uma linha por método
  2. Tokens de contexto × Memória Pico (MB) — uma linha por método
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from rich.console import Console

console = Console()

_COLORS = {
    "baseline":   "#C44E52",
    "uniform":    "#4C72B0",
    "kivi":       "#55A868",
    "turboquant": "#DD8452",
}


def _load_sweep_data(raw_dir: Path) -> list[dict]:
    """Lê todos os context_sweep_*.json e retorna lista de payloads."""
    files = sorted(raw_dir.glob("context_sweep_*.json"))
    return [json.loads(f.read_text()) for f in files]


def _method_label(method: str, bits: int) -> str:
    """Label legível para legenda."""
    return f"{method} FP16" if method == "baseline" else f"{method} {bits}-bit"


def _plot_axis(
    ax: plt.Axes,
    df: pd.DataFrame,
    y_col: str,
    y_label: str,
    title: str,
    bits: int,
) -> None:
    """Plota linhas de cada método num eixo."""
    for method in df["method"].unique():
        sub = df[df["method"] == method].sort_values("actual_tokens")
        color = _COLORS.get(method, "#888888")
        style = "--" if method == "baseline" else "-"
        label = _method_label(method, bits)
        ax.plot(sub["actual_tokens"], sub[y_col], style, color=color, marker="o",
                linewidth=1.8, markersize=5, label=label)
    ax.set_xlabel("Tokens de contexto")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


def plot_context_scaling(
    payload: dict,
    output_dir: Path = Path("results/reports"),
) -> Path:
    """
    Gera context_scaling.png a partir de um payload de context_sweep.

    Exibe crescimento do KV cache e memória pico em função do contexto.
    """
    results = payload.get("results", [])
    if not results:
        console.print("[yellow]context_sweep sem resultados — pulando gráfico[/yellow]")
        return Path()

    df = pd.DataFrame(results)
    bits = payload.get("bits", 4)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    _plot_axis(axes[0], df, "kv_mb",   "KV Cache (MB)",          "KV Cache × Comprimento de Contexto", bits)
    _plot_axis(axes[1], df, "peak_mb", "Memória Pico GPU (MB)",  "Memória Pico × Comprimento de Contexto", bits)

    model = payload.get("model", "")
    if model:
        fig.suptitle(f"Escalonamento de Contexto — {model}", fontsize=11, y=1.02)

    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / "context_scaling.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    console.print(f"[green]✓[/green] {out}")
    return out


def generate_context_report(
    raw_dir: Path = Path("results/raw"),
    output_dir: Path = Path("results/reports"),
) -> list[Path]:
    """
    Lê todos os context_sweep_*.json em raw_dir e gera context_scaling.png.

    Usa o arquivo mais recente se houver múltiplos.
    """
    payloads = _load_sweep_data(raw_dir)
    if not payloads:
        console.print(f"[yellow]Nenhum context_sweep_*.json encontrado em {raw_dir}[/yellow]")
        return []
    payload = payloads[-1]  # mais recente
    console.print(f"[cyan]Usando:[/cyan] context_sweep com {len(payload.get('results', []))} entradas")
    return [plot_context_scaling(payload, output_dir)]
