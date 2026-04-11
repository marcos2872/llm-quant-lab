# LLM Quant Lab

> ⚠️ **Em progresso** — projeto em desenvolvimento ativo. APIs, estrutura de arquivos e resultados podem mudar a qualquer momento.

Benchmark local reproduzível para comparar **quantização de pesos** e **quantização de KV cache** em modelos de linguagem locais, medindo o trade-off entre compressão de memória, throughput e qualidade de geração.

Inspirado no paper **TurboQuant** (Xu et al., 2025 — arXiv:2504.19874).

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
make baseline MODEL=Qwen/Qwen2.5-1.5B-Instruct

# 3. quantizar pesos (INT4)
make weight-quant BITS=4

# 4. quantizar KV cache (TurboQuant, 4 bits)
make kv-quant METHOD=turboquant BITS=4

# 5. avaliar qualidade
make all-eval

# 6. gerar relatório CSV + gráficos
make report
```

Resultados salvos em `results/reports/summary.csv` e `results/reports/*.png`.

---

## Estrutura

```
src/
├── main.py                  CLI Typer
├── runner/                  Carregamento e execução de inferência
│   ├── loader.py            Carrega modelo + tokenizer
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
    └── plots.py             Gráficos comparativos

configs/                     YAMLs por modo de quantização
benchmarks/
├── prompts/basic.jsonl      30 prompts QA curtos
├── long_context/needle.jsonl 10 entradas Needle-in-a-Haystack
└── perplexity/wikitext.jsonl 200 amostras WikiText-2

results/
├── raw/                     JSON por execução
└── reports/                 summary.csv + PNGs
```

---

## Comandos disponíveis

```
make help
```

| Comando | Descrição |
|---|---|
| `make baseline` | Run FP16 baseline |
| `make weight-quant BITS=4` | Quantização de pesos INT4 |
| `make kv-quant METHOD=turboquant BITS=4` | KV cache TurboQuant 4-bit |
| `make eval-ppl` | Perplexidade WikiText-2 |
| `make eval-needle` | Needle-in-a-Haystack |
| `make eval-tasks` | F1/EM em prompts QA |
| `make report` | CSV + gráficos |
| `make all` | Pipeline completo |
| `make clean` | Remove artefatos |

---

## Referências

- **TurboQuant**: Xu et al. (2025) — [arXiv:2504.19874](https://arxiv.org/abs/2504.19874)
- **KIVI**: Liu et al. (2024) — [arXiv:2402.02750](https://arxiv.org/abs/2402.02750)
- **bitsandbytes**: Dettmers et al. — INT8/INT4 weight quantization
