# LLM Quant Lab

> ⚠️ **Em progresso** — projeto em desenvolvimento ativo. APIs, estrutura de arquivos e resultados podem mudar a qualquer momento.

Benchmark local reproduzível para comparar **quantização de pesos** e **quantização de KV cache** em modelos de linguagem locais, medindo o trade-off entre compressão de memória, throughput e qualidade de geração.

Inspirado no paper **TurboQuant** (Xu et al., 2025 — arXiv:2504.19874).

---

## Requisitos

- **GPU NVIDIA** com drivers CUDA (obrigatório — CPU e MPS não são suportados)
- Python ≥ 3.10
- [uv](https://github.com/astral-sh/uv) instalado

> Testado em RTX 4000 Ada Generation, CUDA 12.x, Ubuntu 24.04.

---

## O que este projeto mede

| Grupo | Métricas |
|---|---|
| **Memória** | Pesos (MB), KV cache (MB), pico durante geração |
| **Velocidade** | tok/s prefill, tok/s decode, latência até 1º token |
| **Qualidade intrínseca** | Perplexidade (WikiText-2), NLL médio |
| **Qualidade de tarefa** | Needle-in-a-Haystack recall, F1 / Exact Match em QA |

---

## Modos de quantização

| Modo | Descrição |
|---|---|
| `baseline` | FP16 sem quantização — referência |
| `weight_quant` | Quantização de pesos INT8 / INT4 via bitsandbytes (NF4, double quant) |
| `kv_uniform` | KV cache: quantização uniforme min-max por tensor |
| `kv_kivi` | KV cache: quantização por grupo (KIVI-style) INT2/INT4 |
| `kv_turboquant` | KV cache: rotação ortogonal Haar + Lloyd-Max + outlier FP16 |

---

## Quickstart

```bash
# 1. instalar dependências
make setup

# 2. rodar baseline FP16
make baseline

# 3. quantizar pesos (INT4)
make weight-quant BITS=4

# 4. quantizar KV cache (TurboQuant, 4 bits)
make kv-quant METHOD=turboquant BITS=4

# 5. anotar JSONs com métricas de qualidade e gerar relatório
make annotate-all
make report
```

Resultados salvos em `results/raw/*.json`, `results/reports/summary.csv` e `results/reports/*.png`.

---

## Pipelines prontos

| Comando | Descrição |
|---|---|
| `make all` | Pipeline básico — 1 config por modo: `baseline → weight-quant → kv-quant → annotate-all → report` |
| `make sweep-all` | Benchmark completo — todos os modos e bits. Aceita `PROMPTS=`, `CONFIG=`, `RAW_DIR=`, `OUTPUT_DIR=` |
| `make benchmark-7b` | `sweep-all` pré-configurado para `Qwen/Qwen2.5-7B-Instruct` em `results/7b/` |
| `make benchmark-long` | `sweep-all` com contexto longo 4k+ tokens em `results/long/` (limpa dados anteriores) |

---

## Comandos disponíveis

```bash
make help
```

### Execução de inferência

| Comando | Descrição | Variáveis opcionais |
|---|---|---|
| `make baseline` | Run FP16 sem quantização | `MODEL=` `PROMPTS=` `RAW_DIR=` |
| `make weight-quant` | Quantização de pesos via bitsandbytes | `BITS=4\|8\|4,8` `PROMPTS=` `RAW_DIR=` |
| `make kv-quant` | Quantização de KV cache (1 método/bits) | `METHOD=turboquant\|kivi\|uniform` `BITS=4\|2` `PROMPTS=` `RAW_DIR=` |
| `make kv-quant-long` | KV cache quant com contexto longo 4k+ (3 métodos × bits 4 e 2) | `MODEL=` `RAW_DIR=` |

### Sweeps automáticos

| Comando | Descrição | Variáveis opcionais |
|---|---|---|
| `make sweep-weight` | Weight quant INT4 + INT8 em sequência | `MODEL=` `PROMPTS=` `RAW_DIR=` |
| `make sweep-kv` | KV cache quant — 3 métodos × bits 4 e 2 (6 runs) | `MODEL=` `PROMPTS=` `RAW_DIR=` |
| `make sweep-all` | Benchmark completo — baseline + sweep-weight + sweep-kv + annotate + report | `MODEL=` `PROMPTS=` `CONFIG=` `RAW_DIR=` `OUTPUT_DIR=` |

### Benchmark de escalonamento de contexto

| Comando | Descrição | Variáveis opcionais |
|---|---|---|
| `make context-sweep` | Mede throughput/memória para contextos de 512 → 4096 tokens | `MODEL=` `CONFIG=` `RAW_DIR=` |
| `make context-report` | Gera `context_scaling.png` a partir dos dados do `context-sweep` | `RAW_DIR=` `OUTPUT_DIR=` |

### Avaliação de qualidade

| Comando | Descrição | Variáveis opcionais |
|---|---|---|
| `make annotate-all` | Anota todos os JSONs em `RAW_DIR` com PPL + Needle + F1 | `RAW_DIR=` |
| `make all-eval` | Alias para `annotate-all` | `RAW_DIR=` |
| `make eval-ppl` | Calcula perplexidade individualmente | `CONFIG=` `RESULT_JSON=` |
| `make eval-needle` | Avalia Needle-in-a-Haystack individualmente | `CONFIG=` `RESULT_JSON=` |
| `make eval-tasks` | Avalia F1/EM em prompts QA individualmente | `CONFIG=` `RESULT_JSON=` |

### Relatório e utilitários

| Comando | Descrição | Variáveis opcionais |
|---|---|---|
| `make report` | Gera `summary.csv` + 3 gráficos PNG | `RAW_DIR=` `OUTPUT_DIR=` |
| `make clean` | Remove JSONs, CSVs e PNGs gerados | — |
| `make env` | Instala dependências sem criar `.env` | — |
| `make setup` | Instala deps, cria `.env` e pastas de trabalho | — |

### Usando diretórios e prompts customizados

```bash
# Salvar runs em diretório próprio
make baseline     RAW_DIR=experimentos/run1
make weight-quant BITS=4 RAW_DIR=experimentos/run1

# Usar conjunto de prompts alternativo
make baseline PROMPTS=benchmarks/prompts/meu_dataset.jsonl

# Anotar e gerar relatório no mesmo diretório
make annotate-all RAW_DIR=experimentos/run1
make report       RAW_DIR=experimentos/run1 OUTPUT_DIR=experimentos/run1/report

# Benchmark completo para modelo 7B
make benchmark-7b

# Benchmark completo com contexto longo (limpa results/long/ antes)
make benchmark-long
```

---

## Estrutura

```
src/
├── main.py                  CLI Typer
├── runner/                  Carregamento e execução de inferência
│   ├── loader.py            Carrega modelo + tokenizer (CUDA obrigatório)
│   ├── baseline.py          Run FP16
│   ├── weight_quant.py      Run bitsandbytes INT8/INT4
│   └── kv_quant.py          Run com KV cache quantizado
├── quantization/
│   ├── kv_hooks.py          Hooks PyTorch para attention layers
│   └── methods/
│       ├── uniform.py       Quantização uniforme min-max
│       ├── kivi.py          KIVI-style por grupo de canais
│       └── turboquant.py    Rotação ortogonal + Lloyd-Max
├── eval/
│   ├── perplexity.py        PPL com sliding window
│   ├── needle.py            Needle-in-a-Haystack
│   └── task_score.py        F1 / EM em QA curta
├── metrics/
│   └── collector.py         Memória (torch/psutil) + throughput
└── reporter/
    ├── csv_writer.py        Agrega JSONs → CSV
    └── plots.py             Gráficos comparativos (memória, throughput, qualidade)

configs/
├── baseline.yaml            Configuração FP16 (device: cuda)
├── weight_quant.yaml        Configuração bitsandbytes INT4/INT8
└── kv_quant.yaml            Configuração KV cache quant

benchmarks/
├── prompts/basic.jsonl      30 prompts QA curtos
├── long_context/needle.jsonl 10 entradas Needle-in-a-Haystack
└── perplexity/wikitext.jsonl 200 amostras WikiText-2

results/
├── raw/                     JSON por execução (anotado com métricas após eval)
└── reports/                 summary.csv + memory_comparison.png
                             throughput_comparison.png + quality_tradeoff.png
```

---

## Variáveis de ambiente

Copie `.env.example` para `.env` e ajuste:

| Variável | Descrição | Padrão |
|---|---|---|
| `MODEL_NAME` | Modelo HuggingFace padrão | `Qwen/Qwen2.5-1.5B-Instruct` |
| `HF_TOKEN` | Token para modelos gated (Llama, Gemma…) | — |
| `DEVICE` | Device alvo (`cuda` ou `cuda:N`) | `cuda` |
| `RANDOM_SEED` | Semente global | `42` |
| `KV_QUANT_SEED` | Semente para rotação TurboQuant | `42` |

---

## Dependências principais

| Pacote | Versão | Função |
|---|---|---|
| `transformers` | `>=4.46,<5` | Carregamento de modelos e tokenizers |
| `torch` | `>=2.2` | Inferência e hooks PyTorch |
| `bitsandbytes` | `>=0.43` | Quantização de pesos INT8/INT4 |
| `accelerate` | `>=0.30` | Device map e carregamento distribuído |
| `datasets` | `>=2.19` | WikiText-2 para perplexidade |

---

## Referências

- **TurboQuant**: Xu et al. (2025) — [arXiv:2504.19874](https://arxiv.org/abs/2504.19874)
- **KIVI**: Liu et al. (2024) — [arXiv:2402.02750](https://arxiv.org/abs/2402.02750)
- **bitsandbytes**: Dettmers et al. — INT8/INT4 weight quantization
