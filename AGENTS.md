# AGENTS.md

> Atualizado em 2026-04-11 após refatoração completa para LLM Quant Lab.

## Projeto

- **Nome:** llm-quant-lab
- **Descrição:** Benchmark local reproduzível para comparar quantização de pesos (INT8/INT4 via bitsandbytes) e quantização de KV cache (uniform, KIVI, TurboQuant) em modelos de linguagem locais. Mede trade-off entre compressão de memória, throughput e qualidade (perplexidade, Needle-in-a-Haystack, QA).

## Stack

- **Linguagem:** Python ≥ 3.10
- **Frameworks:** transformers + PyTorch (inferência e hooks), bitsandbytes (weight quant), accelerate (device map), datasets (WikiText-2), Typer (CLI), Matplotlib (gráficos), psutil (memória RAM)

## Gerenciamento de Dependências

- **Instalar tudo:** `uv sync`
- **Adicionar pacote:** `uv add <pacote>`
- **Remover pacote:** `uv remove <pacote>`

## Comandos Essenciais

- **Setup completo:** `make setup`
- **Apenas instalar deps:** `make env`
- **Baseline FP16:** `make baseline [MODEL=...]`
- **Weight quant INT4/INT8:** `make weight-quant [BITS=4]`
- **KV cache quant:** `make kv-quant [METHOD=turboquant] [BITS=4]`
- **Avaliar perplexidade:** `make eval-ppl`
- **Avaliar Needle-in-a-Haystack:** `make eval-needle`
- **Avaliar F1/EM em QA:** `make eval-tasks`
- **Todas as avaliações:** `make all-eval`
- **Relatório CSV + gráficos:** `make report`
- **Pipeline completo:** `make all`
- **Limpar artefatos:** `make clean`
- **Listar targets:** `make help`

## Estrutura de Diretórios

- **Código principal:** `src/`
- **Testes:** `tests/` (ainda não criado — adicionar com `uv add --dev pytest pytest-cov`)
- **Configs:** `configs/` — YAMLs por modo (baseline, weight\_quant, kv\_quant)
- **Benchmarks:** `benchmarks/` — prompts fixos, needle, wikitext
- **Resultados brutos:** `results/raw/` — JSON por execução
- **Relatórios:** `results/reports/` — CSV consolidado + PNGs
- **Modelos locais:** `models/` — cache opcional (não versionado)
- **Notebooks:** `notebooks/`

## Módulos

- **`src/main.py`** — CLI Typer: `baseline`, `weight-quant`, `kv-quant`, `eval-ppl`, `eval-needle`, `eval-tasks`, `report`
- **`src/runner/loader.py`** — Carrega modelo + tokenizer com suporte a BitsAndBytesConfig
- **`src/runner/baseline.py`** — Pipeline baseline FP16: run prompts + coleta métricas
- **`src/runner/weight_quant.py`** — Pipeline weight quant: itera bits, usa bitsandbytes
- **`src/runner/kv_quant.py`** — Pipeline KV quant: instala hooks, roda prompts, remove hooks
- **`src/quantization/kv_hooks.py`** — `install_kv_hooks` / `remove_kv_hooks` via `register_forward_hook`
- **`src/quantization/methods/uniform.py`** — Quantização uniforme min-max por tensor
- **`src/quantization/methods/kivi.py`** — Quantização por grupo de canais estilo KIVI
- **`src/quantization/methods/turboquant.py`** — Rotação ortogonal Haar + Lloyd-Max + outlier FP16
- **`src/metrics/collector.py`** — `MemorySnapshot`, `Throughput`, `measure_throughput()`, `measure_memory_snapshot()`
- **`src/eval/perplexity.py`** — PPL com sliding window em corpus WikiText-2
- **`src/eval/needle.py`** — Needle-in-a-Haystack: constrói contexto longo, mede recall
- **`src/eval/task_score.py`** — F1 token-level e exact match em QA curta
- **`src/reporter/csv_writer.py`** — Agrega JSONs de `results/raw/` → `summary.csv`
- **`src/reporter/plots.py`** — 3 gráficos: memória, throughput, quality tradeoff

## Arquitetura

- **Estilo:** Pipeline modular de 5 fases
- **Descrição:** Cada fase é independente. `src/main.py` orquestra via CLI Typer. Makefile encadeia para pipelines. Resultados persistidos como JSON (por execução) e CSV (consolidado).

```
configs/*.yaml
  → [loader] → model + tokenizer
  → [runner/baseline|weight_quant|kv_quant] → results/raw/*.json
  → [eval/perplexity|needle|task_score] → anota JSON existente
  → [reporter/csv_writer + plots] → results/reports/summary.csv + PNGs
```

## Variáveis de Ambiente

> Copie `.env.example` para `.env` e ajuste.

- `MODEL_NAME` — modelo HuggingFace padrão
- `HF_TOKEN` — token para modelos gated (Llama, etc.)
- `DEVICE` — `auto` | `cpu` | `cuda` | `mps`
- `RANDOM_SEED` — semente global
- `KV_QUANT_SEED` — semente para rotação TurboQuant

## Testes

- **Framework:** pytest
- **Diretório:** `tests/` ⚠️ ainda não criado
- **Executar:** `uv run pytest tests/`
- **Com cobertura:** `uv run pytest tests/ --cov=src --cov-report=term-missing`

## Convenções de Código

- **Tamanho máximo de função:** 40 linhas
- **Tamanho máximo de arquivo:** 300 linhas
- **Aninhamento máximo:** 3 níveis
- **Docstrings / comentários:** Português brasileiro
- **Identificadores:** Inglês
- Python: `X | None`, `list[str]` — nunca `Optional`/`Union` de `typing`
- Prefira `np.random.default_rng(seed)` em vez de `np.random.seed()` (API moderna do NumPy)
- Resultados sempre salvos como `results/raw/<tipo>_<detalhes>_<timestamp>.json`
- Configurações carregadas via `configs/*.yaml` + variáveis de ambiente; nunca hardcode de caminhos
- Hooks PyTorch sempre removidos via `remove_kv_hooks(handles)` após cada run

## Commits

Este projeto segue o padrão **Conventional Commits**.
Antes de commitar, carregue a skill de commit:

```
/skill:git-commit-push
```

## Agentes e Skills

| Agente    | Função                                         | Modo                   |
|-----------|------------------------------------------------|------------------------|
| `build`   | Implementa funcionalidades e corrige bugs      | escrita completa       |
| `ask`     | Responde perguntas somente-leitura             | somente-leitura        |
| `plan`    | Cria planos detalhados em `.pi/plans/`         | escrita em .pi/plans/  |
| `quality` | Auditoria de qualidade de código               | bash + leitura         |
| `qa`      | Análise de bugs e edge cases                   | bash + leitura         |
| `test`    | Cria e mantém testes automatizados             | escrita em tests/      |
| `doc`     | Cria documentação técnica em `docs/`           | escrita em docs/       |
