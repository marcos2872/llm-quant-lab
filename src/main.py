"""
src/main.py
-----------
CLI principal do LLM Quant Lab.

Uso:
    uv run python -m src.main <comando> [opções]
    # ou, após `uv sync`:
    lab <comando> [opções]

Comandos disponíveis por fase:
    Fase 1:  baseline
    Fase 2:  weight-quant
    Fase 3:  kv-quant
    Fase 4:  eval-ppl, eval-needle, eval-tasks
    Fase 5:  report
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Annotated

import typer
import yaml
from dotenv import load_dotenv
from rich.console import Console

load_dotenv()

# suprime FutureWarning e UserWarning do transformers (Cache API, do_sample, etc.)
# usa message= porque o transformers emite com stacklevel alto, fazendo o warning
# aparecer originado no nosso código — o filtro module="transformers" não pega
warnings.filterwarnings("ignore", message=".*past_key_values.*", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*do_sample.*top_k.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*Cache.*", category=FutureWarning)

app = typer.Typer(
    name="lab",
    help="LLM Quant Lab — benchmark local de quantização de LLMs",
    add_completion=False,
)
console = Console()


# ── helpers ────────────────────────────────────────────────────────────────────

def _load_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text()) or {}


def _patch_model(config: dict, model: str | None) -> dict:
    """Sobrescreve model no config se fornecido via CLI."""
    if model:
        config["model"] = model
    return config


# ══════════════════════════════════════════════════════════════════════════════
# FASE 1 — Baseline
# ══════════════════════════════════════════════════════════════════════════════

@app.command()
def baseline(
    config: Annotated[Path, typer.Option("--config", "-c")] = Path("configs/baseline.yaml"),
    prompts: Annotated[Path, typer.Option("--prompts", "-p")] = Path("benchmarks/prompts/basic.jsonl"),
    output_dir: Annotated[Path, typer.Option("--output-dir", "-o")] = Path("results/raw"),
    model: Annotated[str | None, typer.Option("--model", "-m", help="Sobrescreve model do config")] = None,
) -> None:
    """Roda inferência baseline (FP16 sem quantização) e salva métricas."""
    cfg = _load_config(config)
    cfg = _patch_model(cfg, model)
    config.write_text(yaml.dump(cfg)) if model else None

    from src.runner.baseline import run_baseline
    run_baseline(config_path=config, prompts_file=prompts, output_dir=output_dir)


# ══════════════════════════════════════════════════════════════════════════════
# FASE 2 — Weight Quantization
# ══════════════════════════════════════════════════════════════════════════════

@app.command(name="weight-quant")
def weight_quant(
    config: Annotated[Path, typer.Option("--config", "-c")] = Path("configs/weight_quant.yaml"),
    prompts: Annotated[Path, typer.Option("--prompts", "-p")] = Path("benchmarks/prompts/basic.jsonl"),
    output_dir: Annotated[Path, typer.Option("--output-dir", "-o")] = Path("results/raw"),
    bits: Annotated[str, typer.Option("--bits", help="Bits separados por vírgula. Ex: 4,8")] = "",
    model: Annotated[str | None, typer.Option("--model", "-m")] = None,
) -> None:
    """Roda quantização de pesos (INT8/INT4 via bitsandbytes) e salva métricas."""
    cfg = _load_config(config)
    cfg = _patch_model(cfg, model)

    bits_list = [int(b.strip()) for b in bits.split(",")] if bits else None

    from src.runner.weight_quant import run_weight_quant
    run_weight_quant(
        config_path=config,
        prompts_file=prompts,
        output_dir=output_dir,
        bits_list=bits_list,
    )


# ══════════════════════════════════════════════════════════════════════════════
# FASE 3 — KV Cache Quantization
# ══════════════════════════════════════════════════════════════════════════════

@app.command(name="kv-quant")
def kv_quant(
    config: Annotated[Path, typer.Option("--config", "-c")] = Path("configs/kv_quant.yaml"),
    prompts: Annotated[Path, typer.Option("--prompts", "-p")] = Path("benchmarks/prompts/basic.jsonl"),
    output_dir: Annotated[Path, typer.Option("--output-dir", "-o")] = Path("results/raw"),
    method: Annotated[str | None, typer.Option("--method", help="uniform | kivi | turboquant")] = None,
    bits: Annotated[int | None, typer.Option("--bits")] = None,
    model: Annotated[str | None, typer.Option("--model", "-m")] = None,
) -> None:
    """Roda quantização de KV cache com hooks PyTorch e salva métricas."""
    cfg = _load_config(config)
    cfg = _patch_model(cfg, model)

    if method:
        cfg.setdefault("kv_quantization", {})["method"] = method
        cfg.setdefault("kv_quantization", {})["enabled"] = True
    if bits:
        cfg.setdefault("kv_quantization", {})["bits"] = bits

    from src.runner.kv_quant import run_kv_quant
    result = run_kv_quant(prompts_file=prompts, output_dir=output_dir, config_override=cfg)
    if result is None:
        raise typer.Exit(0)


# ══════════════════════════════════════════════════════════════════════════════
# FASE 4 — Avaliação de qualidade
# ══════════════════════════════════════════════════════════════════════════════

@app.command(name="eval-ppl")
def eval_ppl(
    config: Annotated[Path, typer.Option("--config", "-c")] = Path("configs/baseline.yaml"),
    corpus: Annotated[Path, typer.Option("--corpus")] = Path("benchmarks/perplexity/wikitext.jsonl"),
    result_json: Annotated[Path | None, typer.Option("--result-json", help="JSON de run para anotar PPL")] = None,
    max_samples: Annotated[int, typer.Option("--max-samples")] = 50,
    model: Annotated[str | None, typer.Option("--model", "-m")] = None,
) -> None:
    """Calcula perplexidade do modelo no corpus WikiText-2."""
    cfg = _load_config(config)
    cfg = _patch_model(cfg, model)

    from src.runner._utils import resolve_device
    from src.runner.loader import load_model
    llm, tokenizer = load_model(cfg)
    device = resolve_device(llm)

    from src.eval.perplexity import eval_perplexity
    result = eval_perplexity(llm, tokenizer, corpus_path=corpus, max_samples=max_samples, device=device)

    console.print(result)

    if result_json and result_json.exists():
        import json
        payload = json.loads(result_json.read_text())
        payload["perplexity"] = result["perplexity"]
        payload["avg_nll"] = result["avg_nll"]
        result_json.write_text(json.dumps(payload, indent=2))
        console.print(f"[green]✓[/green] PPL anotado em {result_json}")


@app.command(name="eval-needle")
def eval_needle_cmd(
    config: Annotated[Path, typer.Option("--config", "-c")] = Path("configs/baseline.yaml"),
    needle_file: Annotated[Path, typer.Option("--needle-file")] = Path("benchmarks/long_context/needle.jsonl"),
    result_json: Annotated[Path | None, typer.Option("--result-json")] = None,
    model: Annotated[str | None, typer.Option("--model", "-m")] = None,
) -> None:
    """Avalia recall no teste Needle-in-a-Haystack."""
    cfg = _load_config(config)
    cfg = _patch_model(cfg, model)

    from src.runner._utils import resolve_device
    from src.runner.loader import load_model
    llm, tokenizer = load_model(cfg)
    device = resolve_device(llm)

    from src.eval.needle import eval_needle
    result = eval_needle(llm, tokenizer, needle_file=needle_file, device=device)

    if result_json and result_json.exists():
        import json
        payload = json.loads(result_json.read_text())
        payload["needle_recall"] = result["overall_recall"]
        payload["needle_by_context"] = result["by_context_len"]
        result_json.write_text(json.dumps(payload, indent=2))
        console.print(f"[green]✓[/green] Needle recall anotado em {result_json}")


@app.command(name="eval-tasks")
def eval_tasks(
    config: Annotated[Path, typer.Option("--config", "-c")] = Path("configs/baseline.yaml"),
    prompts: Annotated[Path, typer.Option("--prompts", "-p")] = Path("benchmarks/prompts/basic.jsonl"),
    result_json: Annotated[Path | None, typer.Option("--result-json")] = None,
    model: Annotated[str | None, typer.Option("--model", "-m")] = None,
) -> None:
    """Avalia F1 e exact match em conjunto fixo de prompts QA."""
    cfg = _load_config(config)
    cfg = _patch_model(cfg, model)

    from src.runner._utils import resolve_device
    from src.runner.loader import load_model
    llm, tokenizer = load_model(cfg)
    device = resolve_device(llm)

    from src.eval.task_score import eval_task_score
    result = eval_task_score(llm, tokenizer, prompts_file=prompts, device=device)

    if result_json and result_json.exists():
        import json
        payload = json.loads(result_json.read_text())
        payload["task_f1"] = result["avg_f1"]
        payload["exact_match_rate"] = result["exact_match_rate"]
        result_json.write_text(json.dumps(payload, indent=2))
        console.print(f"[green]✓[/green] Task score anotado em {result_json}")


# ══════════════════════════════════════════════════════════════════════════════
# FASE 5 — Relatório
# ══════════════════════════════════════════════════════════════════════════════

@app.command()
def report(
    raw_dir: Annotated[Path, typer.Option("--raw-dir")] = Path("results/raw"),
    output_dir: Annotated[Path, typer.Option("--output-dir")] = Path("results/reports"),
) -> None:
    """Agrega todos os JSONs de resultados em CSV e gera gráficos comparativos."""
    from src.reporter.csv_writer import aggregate_results
    from src.reporter.plots import generate_all_plots

    df = aggregate_results(raw_dir=raw_dir, output_path=output_dir / "summary.csv")

    if df.empty:
        console.print("[yellow]Nenhum resultado para plotar. Rode baseline/weight-quant/kv-quant primeiro.[/yellow]")
        raise typer.Exit(0)

    generate_all_plots(df=df, output_dir=output_dir)
    console.print(f"\n[bold green]✓ Relatório completo em {output_dir}/[/bold green]")



# ══════════════════════════════════════════════════════════════════════════════
# FASE 6 — Context Sweep
# ══════════════════════════════════════════════════════════════════════════════

@app.command(name="context-sweep")
def context_sweep(
    config: Annotated[Path, typer.Option("--config", "-c")] = Path("configs/baseline.yaml"),
    output_dir: Annotated[Path, typer.Option("--output-dir", "-o")] = Path("results/raw"),
    bits: Annotated[int, typer.Option("--bits")] = 4,
    model: Annotated[str | None, typer.Option("--model", "-m")] = None,
) -> None:
    """Benchmark de escalonamento de contexto (512→4096 tokens) com KV quant."""
    cfg = _load_config(config)
    cfg = _patch_model(cfg, model)

    from src.runner.context_sweep import run_context_sweep
    out = run_context_sweep(config_path=config, bits=bits, output_dir=output_dir)
    console.print(f"[bold green]✓ Salvo:[/bold green] {out}")


@app.command(name="context-report")
def context_report(
    raw_dir: Annotated[Path, typer.Option("--raw-dir")] = Path("results/raw"),
    output_dir: Annotated[Path, typer.Option("--output-dir")] = Path("results/reports"),
) -> None:
    """Gera context_scaling.png a partir dos dados de context-sweep."""
    from src.reporter.context_plots import generate_context_report
    paths = generate_context_report(raw_dir=raw_dir, output_dir=output_dir)
    if paths:
        console.print(f"[bold green]✓ Gráfico em {output_dir}/[/bold green]")


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    app()
