"""
src/reporter/plots.py
----------------------
Gera gráficos comparativos a partir do DataFrame de resultados consolidados.

Gráficos gerados por generate_all_plots:
  1. memory_comparison.png    — barras empilhadas: pesos + KV por configuração
  2. throughput_comparison.png — barras agrupadas: tok/s prefill e decode
  3. quality_tradeoff.png     — scatter: compressão × perplexidade, needle e F1
  4. kv_cache_detail.png      — kv_mem_mb por método com linha de referência FP16
  5. latency_breakdown.png    — TTFT + TPOT por configuração
  6. pareto_frontier.png      — fronteira de Pareto: memória × perplexidade
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from rich.console import Console

console = Console()
_OUTPUT_DIR = Path("results/reports")


def _mode_label(row: pd.Series) -> str:
    """Gera label legível por configuração."""
    run_type = str(row.get("run_type", ""))
    method = str(row.get("method") or "none")
    bits = int(row.get("bits", 16))
    if run_type == "baseline":
        return "baseline"
    if run_type == "weight_quant":
        return f"weight\nINT{bits}"
    if run_type == "kv_quant" and method != "none":
        return f"{method}\n{bits}bit"
    return str(row.get("quant_mode", ""))


def _fig_width(n: int, per_bar: float = 2.2, min_w: float = 8.0) -> float:
    """Largura de figura proporcional ao número de barras."""
    return max(min_w, n * per_bar)


def _savefig(fig: plt.Figure, output_dir: Path, name: str) -> Path:
    out = output_dir / name
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    console.print(f"[green]✓[/green] {out}")
    return out


def _bar_value_labels(ax: plt.Axes, bars: list, fmt: str = "{:.0f}", fontsize: int = 7) -> None:
    """Adiciona rótulo de valor no topo de cada barra."""
    for bar in bars:
        h = bar.get_height()
        if h > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + h * 0.01,
                fmt.format(h),
                ha="center", va="bottom", fontsize=fontsize,
            )


def plot_memory_comparison(df: pd.DataFrame, output_dir: Path = _OUTPUT_DIR) -> Path:
    """Barras empilhadas: pesos + KV cache por configuração."""
    if df.empty:
        return Path()
    df = df.copy()
    df["label"] = df.apply(_mode_label, axis=1)
    n = len(df)
    x = range(n)
    fig, ax = plt.subplots(figsize=(_fig_width(n), 5))
    ax.bar(x, df["weights_mb"], label="Pesos (MB)", color="#4C72B0")
    ax.bar(x, df["kv_mem_mb"], bottom=df["weights_mb"], label="KV Cache (MB)", color="#DD8452")
    # rótulo do total no topo de cada barra
    for i, (w, k) in enumerate(zip(df["weights_mb"], df["kv_mem_mb"])):
        total = w + k
        ax.text(i, total + total * 0.01, f"{total:.0f}", ha="center", va="bottom", fontsize=7)
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
    n = len(df)
    x = list(range(n))
    width = 0.35
    fig, ax = plt.subplots(figsize=(_fig_width(n), 5))
    bars_p = ax.bar(
        [i - width / 2 for i in x], df["prefill_tok_s"],
        width, label="Prefill (tok/s)", color="#4C72B0",
    )
    bars_d = ax.bar(
        [i + width / 2 for i in x], df["decode_tok_s"],
        width, label="Decode (tok/s)", color="#55A868",
    )
    _bar_value_labels(ax, bars_p)
    _bar_value_labels(ax, bars_d)
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
    """
    Scatter compressão × métrica de qualidade.

    Pontos com coordenadas idênticas têm labels combinados para evitar
    sobreposição. Grupos distintos alternam lado do rótulo.
    """
    sub = df.dropna(subset=[y_col]).copy()
    if sub.empty:
        return

    sc = ax.scatter(
        sub["compression_ratio"], sub[y_col],
        c=sub["bits"], cmap="viridis", s=90, zorder=3,
    )

    # Agrupa labels por coordenada arredondada
    sub["_key"] = (
        sub["compression_ratio"].round(3).astype(str)
        + "_"
        + sub[y_col].round(3).astype(str)
    )
    groups: dict[str, dict] = {}
    for _, row in sub.iterrows():
        k = row["_key"]
        if k not in groups:
            groups[k] = {"x": row["compression_ratio"], "y": row[y_col], "labels": []}
        groups[k]["labels"].append(row["label"])

    # Alterna posição dos rótulos para grupos diferentes
    _offsets = [(7, 7), (7, -17), (-62, 7), (-62, -17), (7, 22), (7, -30)]
    for i, g in enumerate(groups.values()):
        ox, oy = _offsets[i % len(_offsets)]
        combined = " /\n".join(g["labels"])
        has_arrow = len(g["labels"]) > 1
        ax.annotate(
            combined,
            (g["x"], g["y"]),
            textcoords="offset points",
            xytext=(ox, oy),
            fontsize=7,
            arrowprops=dict(arrowstyle="-", color="#aaaaaa", lw=0.5) if has_arrow else None,
        )

    plt.colorbar(sc, ax=ax, label="bits")
    ax.set_xlabel("Compressão de memória (×)")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)


def plot_quality_tradeoff(df: pd.DataFrame, output_dir: Path = _OUTPUT_DIR) -> Path:
    """Scatter: compressão × perplexidade, needle recall e task F1."""
    if df.empty:
        return Path()
    df = df.copy()
    df["label"] = df.apply(_mode_label, axis=1)
    baseline_mem = df.loc[df["run_type"] == "baseline", "peak_mem_mb"]
    ref_mem = baseline_mem.iloc[0] if not baseline_mem.empty else df["peak_mem_mb"].max()
    df["compression_ratio"] = ref_mem / df["peak_mem_mb"].replace(0, float("nan"))

    fig, axes = plt.subplots(1, 3, figsize=(21, 5))
    _scatter_subplot(axes[0], df, "perplexity",    "Perplexidade (↓ melhor)",  "Compressão × Perplexidade")
    _scatter_subplot(axes[1], df, "needle_recall", "Needle Recall (↑ melhor)", "Compressão × Needle Recall")
    _scatter_subplot(axes[2], df, "task_f1",       "Task F1 (↑ melhor)",       "Compressão × Task F1")
    plt.tight_layout()
    return _savefig(fig, output_dir, "quality_tradeoff.png")


def plot_kv_cache_detail(df: pd.DataFrame, output_dir: Path = _OUTPUT_DIR) -> Path:
    """Barras de kv_mem_mb para cada método KV quant + linha de referência FP16."""
    kv_df = df[df["run_type"] == "kv_quant"].copy()
    if kv_df.empty:
        return Path()
    kv_df["label"] = kv_df.apply(_mode_label, axis=1)
    baseline_rows = df[df["run_type"] == "baseline"]
    baseline_kv = baseline_rows["kv_mem_mb"].iloc[0] if not baseline_rows.empty else None

    n = len(kv_df)
    x = range(n)
    fig, ax = plt.subplots(figsize=(_fig_width(n), 5))
    bars = ax.bar(x, kv_df["kv_mem_mb"], color="#DD8452")
    _bar_value_labels(ax, bars, fmt="{:.1f}")
    if baseline_kv:
        ax.axhline(
            baseline_kv, color="#C44E52", linestyle="--", linewidth=1.5,
            label=f"baseline FP16 ({baseline_kv:.0f} MB)",
        )
        ax.legend()
    ax.set_xticks(list(x))
    ax.set_xticklabels(kv_df["label"], rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("KV Cache (MB)")
    ax.set_title("Compressão do KV Cache por Método")
    plt.tight_layout()
    return _savefig(fig, output_dir, "kv_cache_detail.png")


def plot_latency_breakdown(df: pd.DataFrame, output_dir: Path = _OUTPUT_DIR) -> Path:
    """Barras agrupadas: TTFT (first token) e TPOT (tempo por token de saída)."""
    if df.empty:
        return Path()
    df = df.copy()
    df["label"] = df.apply(_mode_label, axis=1)
    df["ttft_ms"] = (df["first_token_latency_s"] * 1000).round(1)
    df["tpot_ms"] = ((df["total_time_s"] / df["output_tokens"].replace(0, 1)) * 1000).round(1)

    n = len(df)
    x = list(range(n))
    width = 0.35
    fig, ax = plt.subplots(figsize=(_fig_width(n), 5))
    bars_t = ax.bar([i - width / 2 for i in x], df["ttft_ms"], width, label="TTFT (ms)", color="#C44E52")
    bars_p = ax.bar([i + width / 2 for i in x], df["tpot_ms"], width, label="TPOT (ms/tok)", color="#8172B2")
    _bar_value_labels(ax, bars_t, fmt="{:.1f}")
    _bar_value_labels(ax, bars_p, fmt="{:.1f}")
    ax.set_xticks(x)
    ax.set_xticklabels(df["label"], rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Latência (ms)")
    ax.set_title("Latência: Primeiro Token (TTFT) e Por Token (TPOT)")
    ax.legend()
    plt.tight_layout()
    return _savefig(fig, output_dir, "latency_breakdown.png")


def _pareto_front(df: pd.DataFrame, x_col: str, y_col: str) -> pd.DataFrame:
    """Retorna pontos da fronteira de Pareto (minimiza x e y simultaneamente)."""
    sorted_df = df.sort_values(x_col)
    pareto: list[pd.Series] = []
    min_y = float("inf")
    for _, row in sorted_df.iterrows():
        if row[y_col] <= min_y:
            pareto.append(row)
            min_y = row[y_col]
    return pd.DataFrame(pareto)


def plot_pareto_frontier(df: pd.DataFrame, output_dir: Path = _OUTPUT_DIR) -> Path:
    """Scatter memória × perplexidade com fronteira de Pareto ótima destacada."""
    sub = df.dropna(subset=["perplexity"]).copy()
    if sub.empty:
        return Path()
    sub["label"] = sub.apply(_mode_label, axis=1)
    pareto = _pareto_front(sub, "peak_mem_mb", "perplexity")

    fig, ax = plt.subplots(figsize=(10, 6))
    sc = ax.scatter(
        sub["peak_mem_mb"], sub["perplexity"],
        c=sub["bits"], cmap="viridis", s=100, zorder=3,
    )
    if len(pareto) > 1:
        ax.plot(
            pareto["peak_mem_mb"], pareto["perplexity"],
            "r--", linewidth=1.5, label="Fronteira de Pareto", zorder=2,
        )
    for _, row in sub.iterrows():
        ax.annotate(
            row["label"], (row["peak_mem_mb"], row["perplexity"]),
            textcoords="offset points", xytext=(6, 4), fontsize=7,
        )
    plt.colorbar(sc, ax=ax, label="bits")
    ax.set_xlabel("Memória Pico (MB)")
    ax.set_ylabel("Perplexidade (↓ melhor)")
    ax.set_title("Fronteira de Pareto: Memória × Qualidade")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return _savefig(fig, output_dir, "pareto_frontier.png")


def generate_all_plots(df: pd.DataFrame, output_dir: Path = _OUTPUT_DIR) -> list[Path]:
    """Gera todos os gráficos e retorna lista de caminhos."""
    return [
        plot_memory_comparison(df, output_dir),
        plot_throughput_comparison(df, output_dir),
        plot_quality_tradeoff(df, output_dir),
        plot_kv_cache_detail(df, output_dir),
        plot_latency_breakdown(df, output_dir),
        plot_pareto_frontier(df, output_dir),
    ]
