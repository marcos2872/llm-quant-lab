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

## Pipeline completo

```bash
make all
```

Equivale a: `baseline → weight-quant → kv-quant → annotate-all → report`.

---

## Comandos disponíveis

```bash
make help
```

### Execução de inferência

| Comando | Descrição | Variáveis opcionais |
|---|---|---|
| `make baseline` | Run FP16 sem quantização | `MODEL=` `RAW_DIR=` |
| `make weight-quant` | Quantização de pesos via bitsandbytes | `BITS=4` `BITS=4,8` `RAW_DIR=` |
| `make kv-quant` | Quantização de KV cache | `METHOD=turboquant\|kivi\|uniform` `BITS=4` `RAW_DIR=` |

### Avaliação de qualidade

| Comando | Descrição | Variáveis opcionais |
|---|---|---|
| `make annotate-all` | Descobre todos os JSONs em `RAW_DIR` e os anota com PPL + Needle + F1 | `RAW_DIR=` |
| `make all-eval` | Alias para `annotate-all` | `RAW_DIR=` |
| `make eval-ppl` | Calcula perplexidade individualmente | `CONFIG=` `RESULT_JSON=` |
| `make eval-needle` | Avalia Needle-in-a-Haystack individualmente | `CONFIG=` `RESULT_JSON=` |
| `make eval-tasks` | Avalia F1/EM em prompts QA individualmente | `CONFIG=` `RESULT_JSON=` |

### Relatório e utilitários

| Comando | Descrição | Variáveis opcionais |
|---|---|---|
| `make report` | Gera `summary.csv` + 3 gráficos PNG | `RAW_DIR=` `OUTPUT_DIR=` |
| `make all` | Pipeline completo | `MODEL=` `BITS=` `RAW_DIR=` `OUTPUT_DIR=` |
| `make clean` | Remove JSONs, CSVs e PNGs gerados | — |

### Usando diretórios customizados

```bash
# Salvar runs em diretório próprio
make baseline     RAW_DIR=experimentos/run1
make weight-quant BITS=4 RAW_DIR=experimentos/run1

# Anotar e gerar relatório no mesmo diretório
make annotate-all RAW_DIR=experimentos/run1
make report       RAW_DIR=experimentos/run1 OUTPUT_DIR=experimentos/run1/report
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
