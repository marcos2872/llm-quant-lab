"""
src/reporter/plots.py
----------------------
Gera gráficos comparativos a partir do DataFrame de resultados consolidados.

Gráficos gerados:
  1. memory_comparison.png   — barra: pesos + KV por configuração
  2. throughput_comparison.png — barra: tok/s (prefill e decode) por configuração
  3. quality_tradeoff.png    — scatter: compressão × qualidade (PPL / needle recall)
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
    mode = row.get("quant_mode", "")
    if method and method != "none":
        return f"{method}\n{row['bits']}bit"
    return mode or "baseline"


def plot_memory_comparison(df: pd.DataFrame, output_dir: Path = _OUTPUT_DIR) -> Path:
    """Gráfico de barras empilhadas: memória de pesos + KV cache por configuração."""
    if df.empty:
        console.print("[yellow]DataFrame vazio — pulando gráfico de memória[/yellow]")
        return Path()

    df = df.copy()
    df["label"] = df.apply(_mode_label, axis=1)

    fig, ax = plt.subplots(figsize=(10, 5))
    x = range(len(df))
    ax.bar(x, df["weights_mb"], label="Pesos (MB)", color="#4C72B0")
    ax.bar(x, df["kv_mem_mb"], bottom=df["weights_mb"], label="KV Cache (MB)", color="#DD8452")

    ax.set_xticks(list(x))
    ax.set_xticklabels(df["label"], rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Memória (MB)")
    ax.set_title("Uso de Memória por Configuração de Quantização")
    ax.legend()
    plt.tight_layout()

    out = output_dir / "memory_comparison.png"
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    console.print(f"[green]✓[/green] {out}")
    return out


def plot_throughput_comparison(df: pd.DataFrame, output_dir: Path = _OUTPUT_DIR) -> Path:
    """Gráfico de barras agrupadas: tok/s prefill e decode por configuração."""
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
    ax.set_title("Throughput por Configuração de Quantização")
    ax.legend()
    plt.tight_layout()

    out = output_dir / "throughput_comparison.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    console.print(f"[green]✓[/green] {out}")
    return out


def plot_quality_tradeoff(df: pd.DataFrame, output_dir: Path = _OUTPUT_DIR) -> Path:
    """
    Scatter: eixo X = compressão de memória relativa ao baseline,
             eixo Y = qualidade (perplexidade ou needle_recall).
    """
    if df.empty:
        return Path()

    df = df.copy()
    df["label"] = df.apply(_mode_label, axis=1)

    baseline_mem = df.loc[df["run_type"] == "baseline", "peak_mem_mb"]
    if baseline_mem.empty:
        baseline_mem_val = df["peak_mem_mb"].max()
    else:
        baseline_mem_val = baseline_mem.iloc[0]

    df["compression_ratio"] = baseline_mem_val / df["peak_mem_mb"].replace(0, float("nan"))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # PPL subplot
    ppl_df = df.dropna(subset=["perplexity"])
    if not ppl_df.empty:
        sc = axes[0].scatter(
            ppl_df["compression_ratio"], ppl_df["perplexity"],
            c=ppl_df["bits"], cmap="viridis", s=100, zorder=3,
        )
        for _, row in ppl_df.iterrows():
            axes[0].annotate(row["label"], (row["compression_ratio"], row["perplexity"]),
                             textcoords="offset points", xytext=(5, 5), fontsize=7)
        plt.colorbar(sc, ax=axes[0], label="bits")
    axes[0].set_xlabel("Compressão de memória (×)")
    axes[0].set_ylabel("Perplexidade (↓ melhor)")
    axes[0].set_title("Trade-off: Compressão × Perplexidade")
    axes[0].grid(True, alpha=0.3)

    # Needle recall subplot
    nr_df = df.dropna(subset=["needle_recall"])
    if not nr_df.empty:
        sc2 = axes[1].scatter(
            nr_df["compression_ratio"], nr_df["needle_recall"],
            c=nr_df["bits"], cmap="viridis", s=100, zorder=3,
        )
        for _, row in nr_df.iterrows():
            axes[1].annotate(row["label"], (row["compression_ratio"], row["needle_recall"]),
                             textcoords="offset points", xytext=(5, 5), fontsize=7)
        plt.colorbar(sc2, ax=axes[1], label="bits")
    axes[1].set_xlabel("Compressão de memória (×)")
    axes[1].set_ylabel("Needle Recall (↑ melhor)")
    axes[1].set_title("Trade-off: Compressão × Needle Recall")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    out = output_dir / "quality_tradeoff.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    console.print(f"[green]✓[/green] {out}")
    return out


def generate_all_plots(df: pd.DataFrame, output_dir: Path = _OUTPUT_DIR) -> list[Path]:
    """Gera todos os gráficos e retorna lista de caminhos."""
    return [
        plot_memory_comparison(df, output_dir),
        plot_throughput_comparison(df, output_dir),
        plot_quality_tradeoff(df, output_dir),
    ]
