"""
src/reporter/plots.py
----------------------
Gera gráficos comparativos a partir do DataFrame de resultados consolidados.

Gráficos:
  1. memory_comparison.png   — barras: pesos + KV por configuração
  2. throughput_comparison.png — barras: tok/s prefill e decode
  3. quality_tradeoff.png    — scatter: compressão × qualidade
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from rich.console import Console

console = Console()
_OUTPUT_DIR = Path("results/reports")


def _mode_label(row: pd.Series) -> str:
    method = row.get("method", "none")
    if method and method != "none":
        return f"{method}\n{row['bits']}bit"
    return row.get("quant_mode", "") or "baseline"


def _savefig(fig: plt.Figure, output_dir: Path, name: str) -> Path:
    out = output_dir / name
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    console.print(f"[green]✓[/green] {out}")
    return out


def plot_memory_comparison(df: pd.DataFrame, output_dir: Path = _OUTPUT_DIR) -> Path:
    """Barras empilhadas: pesos + KV cache por configuração."""
    if df.empty:
        return Path()
    df = df.copy()
    df["label"] = df.apply(_mode_label, axis=1)
    x = range(len(df))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x, df["weights_mb"], label="Pesos (MB)", color="#4C72B0")
    ax.bar(x, df["kv_mem_mb"], bottom=df["weights_mb"], label="KV Cache (MB)", color="#DD8452")
    ax.set_xticks(list(x))
    ax.set_xticklabels(df["label"], rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Memória (MB)")
    ax.set_title("Uso de Memória por Configuração")
    ax.legend()
    plt.tight_layout()
    return _savefig(fig, output_dir, "memory_comparison.png")


def plot_throughput_comparison(df: pd.DataFrame, output_dir: Path = _OUTPUT_DIR) -> Path:
    """Barras agrupadas: tok/s prefill e decode por configuração."""
    if df.empty:
        return Path()
    df = df.copy()
    df["label"] = df.apply(_mode_label, axis=1)
    x = list(range(len(df)))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar([i - width / 2 for i in x], df["prefill_tok_s"], width, label="Prefill (tok/s)", color="#4C72B0")
    ax.bar([i + width / 2 for i in x], df["decode_tok_s"], width, label="Decode (tok/s)", color="#55A868")
    ax.set_xticks(x)
    ax.set_xticklabels(df["label"], rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Tokens / segundo")
    ax.set_title("Throughput por Configuração")
    ax.legend()
    plt.tight_layout()
    return _savefig(fig, output_dir, "throughput_comparison.png")


def _scatter_subplot(
    ax: plt.Axes,
    df: pd.DataFrame,
    y_col: str,
    y_label: str,
    title: str,
) -> None:
    """Renderiza um subplot scatter compressão × métrica de qualidade."""
    sub = df.dropna(subset=[y_col])
    if sub.empty:
        return
    sc = ax.scatter(sub["compression_ratio"], sub[y_col], c=sub["bits"], cmap="viridis", s=100, zorder=3)
    for _, row in sub.iterrows():
        ax.annotate(row["label"], (row["compression_ratio"], row[y_col]),
                    textcoords="offset points", xytext=(5, 5), fontsize=7)
    plt.colorbar(sc, ax=ax, label="bits")
    ax.set_xlabel("Compressão de memória (×)")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)


def plot_quality_tradeoff(df: pd.DataFrame, output_dir: Path = _OUTPUT_DIR) -> Path:
    """Scatter: compressão de memória × perplexidade e needle recall."""
    if df.empty:
        return Path()
    df = df.copy()
    df["label"] = df.apply(_mode_label, axis=1)
    baseline_mem = df.loc[df["run_type"] == "baseline", "peak_mem_mb"]
    ref_mem = baseline_mem.iloc[0] if not baseline_mem.empty else df["peak_mem_mb"].max()
    df["compression_ratio"] = ref_mem / df["peak_mem_mb"].replace(0, float("nan"))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    _scatter_subplot(axes[0], df, "perplexity", "Perplexidade (↓ melhor)", "Compressão × Perplexidade")
    _scatter_subplot(axes[1], df, "needle_recall", "Needle Recall (↑ melhor)", "Compressão × Needle Recall")
    plt.tight_layout()
    return _savefig(fig, output_dir, "quality_tradeoff.png")


def generate_all_plots(df: pd.DataFrame, output_dir: Path = _OUTPUT_DIR) -> list[Path]:
    """Gera todos os gráficos e retorna lista de caminhos."""
    return [
        plot_memory_comparison(df, output_dir),
        plot_throughput_comparison(df, output_dir),
        plot_quality_tradeoff(df, output_dir),
    ]
