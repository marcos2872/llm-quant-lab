# Plano: Runtime CLI com TurboQuant (projeto isolado)

**Data:** 2026-04-14
**Autor:** agente-plan
**Status:** aprovado

---

## Objetivo

Criar `runtime/` como um **projeto uv completamente independente**, com seu próprio
`pyproject.toml`, `.venv`, `uv.lock` e `Makefile`. Os comandos de inferência (`download`,
`run`, `run-turboquant`, `run-full-quant`) são acessíveis apenas via `make` dentro de
`runtime/`. O projeto reutiliza `src/` do projeto pai via dependência de caminho editável
(`../`), sem duplicar código. Um painel Rich Live exibe métricas ao vivo durante o chat.

---

## Escopo

**Dentro do escopo:**
- `make download MODEL=<model>` — baixa modelo do HuggingFace Hub
- `make run MODEL=<model>` — chat interativo FP16 sem quantização
- `make run-turboquant MODEL=<model> [BITS=4]` — chat com KV TurboQuant
- `make run-full-quant MODEL=<model> [BITS=4]` — chat INT4 pesos (NF4) + KV TurboQuant
- Painel Rich Live: VRAM usada/total, RAM, GPU%, prefill tok/s, decode tok/s,
  first-token latency, KV cache MB, pesos MB — atualizados em tempo real
- Streaming de tokens no painel de conversa durante a geração
- Suporte a `apply_chat_template` quando disponível no tokenizer
- `system_prompt` configurável via flag ou variável de ambiente
- Isolamento total: `runtime/` tem seu próprio `.venv/` e `uv.lock`
- Reutilização de `src/` do projeto pai via path dependency editable (`../`)

**Fora do escopo:**
- Histórico de conversa persistente em disco
- Servidor HTTP / API REST
- Quantização INT8 de pesos neste runtime
- Testes automatizados (escopo do agente `test`)
- Alterações em qualquer arquivo do projeto pai (`src/`, `pyproject.toml` raiz, etc.)

---

## Estrutura de Diretórios (resultado final)

```
runtime/                        ← raiz do projeto isolado
├── pyproject.toml              ← projeto "llm-runtime", depende de "../" (llm-quant-lab)
├── uv.lock                     ← gerado por `uv sync` (não versionar .venv)
├── Makefile                    ← targets: setup, download, run, run-turboquant, run-full-quant, help
├── .env.example                ← MODEL_NAME, HF_TOKEN, DEVICE, SYSTEM_PROMPT, MODELS_DIR
├── models/                     ← modelos baixados localmente (não versionado)
├── .gitignore                  ← ignora .venv/
└── llm_runtime/                ← pacote Python (nome sem conflito com src/ do pai)
    ├── __init__.py
    ├── cli.py                  ← Typer app com 4 comandos
    ├── engine.py               ← InferenceEngine (carrega modelo, gera com streaming)
    ├── display.py              ← RuntimeDisplay (Rich Live, painel métricas + chat)
    ├── monitor.py              ← GPUMonitor (thread daemon, VRAM/RAM/GPU%/temp)
    └── downloader.py           ← download_model() com progresso Rich
```

---

## Dependências do `runtime/pyproject.toml`

```toml
[project]
name = "llm-runtime"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
  "llm-quant-lab",        # projeto pai — provê src.runner.loader, src.quantization.*
  "pynvml>=11.5",         # GPU utilization % e temperatura (NVIDIA NVML)
  "typer[all]>=0.12",
  "rich>=13",
  "python-dotenv>=1.0",
  "psutil>=5.9",
]

[project.scripts]
llm-runtime = "llm_runtime.cli:app"

[tool.uv.sources]
llm-quant-lab = { path = "../", editable = true }
```

> **Por que path dependency editable?**
> `from src.runner.loader import load_model` e `from src.quantization.kv_cache import
> QuantizedDynamicCache` funcionam porque o pai é instalado com seus pacotes (`src/`)
> no path do `.venv` de `runtime/`. Zero duplicação de código.

> **Por que `pynvml` aqui e não no pai?**
> É uma dependência exclusiva do runtime interativo. `torch.cuda` expõe apenas memória
> alocada; GPU utilization % e temperatura exigem NVML. `pynvml` é a binding oficial e
> não tem subdependências pesadas.

> **Conflito de nome `src`:** O pai exporta o pacote `src` (configurado em
> `[tool.hatch.build.targets.wheel] packages = ["src"]`). O runtime **não cria** nenhum
> diretório `src/` próprio — por isso o pacote Python do runtime se chama `llm_runtime/`.

---

## Arquivos Afetados

| Arquivo | Ação | Motivo |
|---------|------|--------|
| `runtime/pyproject.toml` | criar | define projeto isolado + path dep para pai |
| `runtime/Makefile` | criar | interface de uso — targets make |
| `runtime/.env.example` | criar | variáveis de ambiente documentadas |
| `runtime/models/.gitkeep` | criar | garante que a pasta exista no repositório |
| `runtime/.gitignore` | criar | exclui `.venv/` e `uv.lock` do versionamento |
| `runtime/llm_runtime/__init__.py` | criar | torna o diretório um pacote |
| `runtime/llm_runtime/cli.py` | criar | Typer app com 4 comandos |
| `runtime/llm_runtime/engine.py` | criar | InferenceEngine |
| `runtime/llm_runtime/display.py` | criar | RuntimeDisplay (Rich Live) |
| `runtime/llm_runtime/monitor.py` | criar | GPUMonitor (thread daemon) |
| `runtime/llm_runtime/downloader.py` | criar | download_model() |

**Nenhum arquivo fora de `runtime/` é criado ou modificado.**

---

## Arquitetura do Painel (Rich Live)

```
┌─────────────────────────────────────────────────────────────────┐
│  🚀 LLM Runtime │ Qwen/Qwen2.5-7B-Instruct │ mode: TurboQuant  │
├──────────────────────┬──────────────────────────────────────────┤
│  📊 Métricas         │  💬 Conversa                             │
│                      │                                          │
│  VRAM  14.5 / 24 GB  │  [você] Qual é a capital da França?      │
│  RAM    4.2 GB       │                                          │
│  GPU    87% @ 72°C   │  [modelo] A capital da França é Paris.   │
│                      │  Paris é uma cidade localizada no        │
│  Prefill  31k tok/s  │  norte da França, às margens do rio      │
│  Decode   21.5 tok/s │  Sena...▌                                │
│  1st tok  0.046s     │                                          │
│  Geração  3.4s       │                                          │
│                      │                                          │
│  KV cache  217 MB    │                                          │
│  Pesos    5.3 GB     │                                          │
│  Tokens gerados  64  │                                          │
└──────────────────────┴──────────────────────────────────────────┘
```

---

## Decomposição do Layout CLI

### Árvore de comandos

```
llm-runtime
├── download  <MODEL>                     Baixa modelo do HuggingFace Hub
│   ├── --token / -t      TEXT            HF token (ou env HF_TOKEN)
│   └── --cache-dir       PATH  [./models]  Diretório local de modelos (ou env MODELS_DIR)
│
├── run       --model MODEL               Chat FP16 sem quantização
│   ├── --model / -m      TEXT  [req]     Repo ID (ex: Qwen/Qwen2.5-7B-Instruct)
│   ├── --max-tokens      INT   [512]     Máximo de tokens gerados por resposta
│   ├── --system-prompt   TEXT  [None]    Prompt de sistema (ou env SYSTEM_PROMPT)
│   └── --device          TEXT  [cuda]    Dispositivo: cuda | cpu | mps | auto
│
├── run-turboquant --model MODEL          Chat com KV cache TurboQuant
│   ├── --model / -m      TEXT  [req]
│   ├── --bits            INT   [4]       Bits de quantização do KV (4 | 8)
│   ├── --max-tokens      INT   [512]
│   ├── --system-prompt   TEXT  [None]
│   └── --device          TEXT  [cuda]
│
└── run-full-quant --model MODEL          Chat INT4 pesos (NF4) + KV TurboQuant
    ├── --model / -m      TEXT  [req]
    ├── --bits            INT   [4]
    ├── --max-tokens      INT   [512]
    ├── --system-prompt   TEXT  [None]
    └── --device          TEXT  [cuda]
```

---

### Telas do CLI

#### `make help` / `llm-runtime --help`

```
 Usage: llm-runtime [OPTIONS] COMMAND [ARGS]...

 Runtime LLM interativo com quantização TurboQuant.

╭─ Commands ──────────────────────────────────────────────────────────────────╮
│ download        Baixa modelo do HuggingFace Hub                             │
│ run             Chat interativo FP16 sem quantização                        │
│ run-turboquant  Chat interativo com KV cache TurboQuant                     │
│ run-full-quant  Chat com pesos INT4 (NF4) + KV TurboQuant                  │
╰─────────────────────────────────────────────────────────────────────────────╯
╭─ Options ───────────────────────────────────────────────────────────────────╮
│ --help    Show this message and exit.                                       │
╰─────────────────────────────────────────────────────────────────────────────╯
```

---

#### `make download MODEL=<repo>` em execução

```
⬇  Baixando Qwen/Qwen2.5-7B-Instruct ...

Fetching 32 files: 100%|████████████████| 32/32 [00:03<00:00]
Downloading config.json:      100%|████| 662/662 [00:00<00:00, 1.24MB/s]
Downloading tokenizer.json:   100%|████| 7.03M/7.03M [00:01<00:00, 4.2MB/s]
Downloading model-00001.safetensors: 100%|███| 4.97G/4.97G [02:14<00:00, 37.0MB/s]
...

✅ Modelo salvo em: runtime/models/models--Qwen--Qwen2.5-7B-Instruct
```

---

#### `make run` / `make run-turboquant` / `make run-full-quant` — tela inicial (carregamento)

```
⚙  Carregando Qwen/Qwen2.5-7B-Instruct [mode: turboquant, bits: 4] ...
   device: cuda  │  dtype: fp16
```

> Exibido antes de o `Live` iniciar, enquanto `load_model()` roda.

---

#### Estrutura de `runtime/models/` após download

```
runtime/models/
└── Qwen--Qwen2.5-7B-Instruct/          ← repo_id com "/" substituído por "--"
    ├── config.json
    ├── tokenizer.json
    ├── tokenizer_config.json
    ├── special_tokens_map.json
    ├── generation_config.json
    ├── model-00001-of-00004.safetensors
    ├── model-00002-of-00004.safetensors
    ├── model-00003-of-00004.safetensors
    └── model-00004-of-00004.safetensors
```

> `snapshot_download` respeita a `cache_dir` passada e cria subdiretórios
> no padrão `models--<owner>--<repo>`. O `load_model()` recebe o path local
> diretamente — sem precisar do HF Hub online após o primeiro download.

---

#### Painel Rich Live durante o chat (anotado)

```
┌─ 🚀 LLM Runtime │ Qwen/Qwen2.5-7B-Instruct │ mode: turboquant ──────────────┐
│                                                                               │
│ ┌── 📊 Métricas ──────────┐  ┌── 💬 Conversa ──────────────────────────────┐ │
│ │                         │  │                                              │ │
│ │ VRAM    14.5 / 24.0 GB  │  │ [você]                                       │ │
│ │ RAM      4.2 GB         │  │  Qual é a capital da França?                 │ │
│ │ GPU      87% @ 72 °C    │  │                                              │ │
│ │ ─────────────────────── │  │ [modelo]                                     │ │
│ │ Prefill  31,000 tok/s   │  │  A capital da França é Paris. Paris é uma    │ │
│ │ Decode    21.5 tok/s    │  │  cidade localizada no norte da França, às    │ │
│ │ 1st tok    0.046 s      │  │  margens do rio Sena...▌                     │ │
│ │ Geração    3.4 s        │  │                                              │ │
│ │ Tokens      64          │  │                                              │ │
│ │ ─────────────────────── │  │                                              │ │
│ │ KV cache   217 MB       │  │                                              │ │
│ │ Pesos    5,309 MB       │  │                                              │ │
│ └─────────────────────────┘  └──────────────────────────────────────────────┘ │
└───────────────────────────────────────────────────────────────────────────────┘
```

> **`▌`** — cursor piscante acumulando tokens durante streaming.
> O painel de **Métricas** (esquerda, ratio=1) é atualizado a cada 200 ms pela thread do
> `GPUMonitor`. As linhas de throughput/tokens são atualizadas só após cada geração completa.
> O painel de **Conversa** (direita, ratio=2) cresce linha a linha conforme tokens chegam.

---

#### Prompt de input (Live pausado)

```
[você] ▶ _
```

> O Live é interrompido com `Live.stop()` antes de `input()`, evitando artefatos visuais.
> Após `Enter`, o Live é retomado com `Live.start()` e a resposta começa a fluir.

---

#### Saída de encerramento (`exit` / `Ctrl+C`)

```
👋 Encerrando sessão.
   Tokens gerados na sessão : 347
   Tempo total de geração   : 16.2 s
   Decode médio             : 21.4 tok/s
```

---

### Regras de cor (Rich markup)

| Elemento | Estilo Rich |
|---|---|
| Cabeçalho (header Rule) | `bold cyan` |
| Label de métrica | `dim white` |
| Valor de métrica (normal) | `bold white` |
| Valor de métrica (alerta ≥ 90% VRAM) | `bold red` |
| Prefixo `[você]` | `bold green` |
| Prefixo `[modelo]` | `bold yellow` |
| Tokens em streaming | `white` |
| Cursor `▌` | `bold magenta blink` |
| Mensagem de encerramento | `dim italic` |

---

## Sequência de Execução

### 1. Criar `runtime/.gitignore`

**O que fazer:**
```
.venv/
__pycache__/
*.pyc
.env
models/
```

**Dependências:** nenhuma

---

### 2. Criar `runtime/pyproject.toml`

**O que fazer:** definir projeto standalone com path dep editable para o pai.

Campos obrigatórios:
- `name = "llm-runtime"`, `requires-python = ">=3.10"`
- `dependencies`: lista acima (llm-quant-lab, pynvml, typer, rich, python-dotenv, psutil)
- `[project.scripts]`: `llm-runtime = "llm_runtime.cli:app"`
- `[tool.uv.sources]`: `llm-quant-lab = { path = "../", editable = true }`
- `[build-system]`: hatchling
- `[tool.hatch.build.targets.wheel]`: `packages = ["llm_runtime"]`

**Dependências:** passo 1

---

### 3. Criar `runtime/Makefile`

**O que fazer:**

```makefile
MODEL   ?= Qwen/Qwen2.5-7B-Instruct
BITS    ?= 4
DEVICE  ?= cuda
CMD      = uv run llm-runtime

.PHONY: setup download run run-turboquant run-full-quant help

help:           ## Mostra esta ajuda
    @grep -E '^[a-zA-Z_-]+:.*##' $(MAKEFILE_LIST) | \
        awk 'BEGIN {FS = ":.*##"}; {printf "  %-20s %s\n", $$1, $$2}'

setup:          ## Instala dependências (uv sync)
    uv sync

download:       ## Baixa modelo do HuggingFace (MODEL=<repo_id>)
    $(CMD) download $(MODEL)

run:            ## Chat FP16 sem quantização (MODEL=<repo_id>)
    $(CMD) run --model $(MODEL) --device $(DEVICE)

run-turboquant: ## Chat com KV TurboQuant (MODEL=... BITS=4)
    $(CMD) run-turboquant --model $(MODEL) --bits $(BITS) --device $(DEVICE)

run-full-quant: ## Chat INT4 pesos + KV TurboQuant (MODEL=... BITS=4)
    $(CMD) run-full-quant --model $(MODEL) --bits $(BITS) --device $(DEVICE)
```

**Dependências:** passo 2 (precisa do entry point definido)

---

### 4. Criar `runtime/.env.example`

**O que fazer:**
```bash
MODEL_NAME=Qwen/Qwen2.5-7B-Instruct
HF_TOKEN=
DEVICE=cuda
SYSTEM_PROMPT=
MODELS_DIR=./models
```

**Dependências:** nenhuma

---

### 5. Criar `runtime/llm_runtime/__init__.py`

**O que fazer:** arquivo vazio.

**Dependências:** nenhuma

---

### 5b. Criar `runtime/models/.gitkeep`

**O que fazer:** arquivo vazio — garante que `runtime/models/` exista no repositório
em branco, sem versionar os pesos.

**Dependências:** nenhuma

---

### 6. Criar `runtime/llm_runtime/monitor.py`

**O que fazer:** `GPUMonitor` — thread daemon que coleta métricas a cada 200ms.

```python
class GPUMonitor:
    """Monitora VRAM, RAM, GPU utilization e temperatura em thread de background."""

    vram_used_mb: float    # torch.cuda.memory_allocated() / 1024²
    vram_total_mb: float   # nvmlDeviceGetMemoryInfo().total / 1024²
    gpu_util_pct: float    # nvmlDeviceGetUtilizationRates().gpu  (0 se não disponível)
    gpu_temp_c: float      # nvmlDeviceGetTemperature()           (0 se não disponível)
    ram_mb: float          # psutil.Process().memory_info().rss / 1024²

    def start(self) -> None: ...  # lança threading.Thread(daemon=True, target=_poll)
    def stop(self) -> None: ...   # seta threading.Event(); join()
```

Detalhe de implementação:
- `__init__`: tenta `pynvml.nvmlInit()`; se falhar, seta flag `_nvml_ok = False`
- `_poll()`: loop com `time.sleep(0.2)`, atualiza atributos de instância
- Se `_nvml_ok is False`: `vram_total_mb` via `torch.cuda.get_device_properties(0).total_memory`;
  `gpu_util_pct = 0.0`, `gpu_temp_c = 0.0`
- `stop()`: chama `pynvml.nvmlShutdown()` se `_nvml_ok`

**Dependências:** passo 5

---

### 7. Criar `runtime/llm_runtime/display.py`

**O que fazer:** `RuntimeDisplay` — contexto Rich Live, dois painéis.

```python
class RuntimeDisplay:
    def __init__(self, model_name: str, mode: str, monitor: GPUMonitor) -> None: ...
    def __enter__(self) -> RuntimeDisplay: ...    # Live.__enter__()
    def __exit__(self, *_) -> None: ...           # Live.__exit__()

    def update_metrics(self, m: dict) -> None:
        # m: prefill_tok_s, decode_tok_s, first_token_latency_s,
        #    total_time_s, kv_mb, weights_mb, output_tokens
        # atualiza Table no painel esquerdo

    def append_user(self, text: str) -> None:   # adiciona linha "[você] texto"
    def append_token(self, token: str) -> None: # acumula token no buffer atual
    def finalize_response(self) -> None:        # fecha buffer e limpa para próxima resposta
    def prompt_input(self) -> str:              # para Live, lê input, retoma Live
```

Layout:
```
Layout("root")
 ├─ Layout("header", size=3)      ← Rule com nome do modelo e modo
 └─ Layout("body")
      ├─ Layout("metrics", ratio=1)  ← Panel("📊 Métricas") + Table de 2 colunas
      └─ Layout("chat", ratio=2)     ← Panel("💬 Conversa") + Text (buffer de conversa)
```

Painel de métricas — Table com colunas `Métrica` / `Valor`:
```
VRAM        14.5 / 24 GB
RAM         4.2 GB
GPU         87% @ 72°C
──────────────────────
Prefill     31,000 tok/s
Decode      21.5 tok/s
1st token   0.046 s
Geração     3.4 s
Tokens out  64
──────────────────────
KV cache    217 MB
Pesos       5,309 MB
```

O monitor atualiza VRAM/RAM/GPU continuamente (via `Live.refresh()` na thread do monitor).
As métricas de throughput são atualizadas após cada geração.

**Nota sobre `prompt_input()`:** chama `self._live.stop()`, lê via `input()` ou
`Console().input()`, depois chama `self._live.start()`. Isso evita conflito visual entre
o painel e o cursor de input.

**Dependências:** passo 6 (GPUMonitor)

---

### 8. Criar `runtime/llm_runtime/downloader.py`

**O que fazer:** `download_model()` com progresso Rich.

```python
from pathlib import Path

DEFAULT_MODELS_DIR = Path(__file__).parent.parent / "models"  # runtime/models/

def download_model(
    model_id: str,
    token: str | None = None,
    cache_dir: Path | str | None = None,
) -> str:
    """Baixa modelo via huggingface_hub.snapshot_download.

    Salva por padrão em runtime/models/ (ou MODELS_DIR do ambiente).
    """
    import os
    from huggingface_hub import snapshot_download

    resolved = Path(
        cache_dir
        or os.environ.get("MODELS_DIR", str(DEFAULT_MODELS_DIR))
    )
    resolved.mkdir(parents=True, exist_ok=True)

    path = snapshot_download(repo_id=model_id, token=token, cache_dir=str(resolved))
    # rich.print confirmação com caminho final
    return path
```

**Prioridade de `cache_dir`:**
1. Flag `--cache-dir` explícita na CLI
2. Variável de ambiente `MODELS_DIR`
3. Default: `runtime/models/` (relativo ao pacote)

`snapshot_download` já exibe progresso via tqdm internamente. Não é necessário adaptador
custom — apenas garantir que `rich` não suprima o output do tqdm.

**Dependências:** passo 5

---

### 9. Criar `runtime/llm_runtime/engine.py`

**O que fazer:** `InferenceEngine` — carrega modelo no modo correto, gera com streaming.

```python
class InferenceEngine:
    def __init__(
        self,
        model_name: str,
        mode: str,           # "fp16" | "turboquant" | "full_quant"
        bits: int = 4,
        device: str = "cuda",
        trust_remote_code: bool = False,
    ) -> None:
        # Monta config dict conforme o modo:
        #   fp16:       {model, dtype:"fp16", device}
        #   turboquant: {model, dtype:"fp16", device}
        #   full_quant: {model, dtype:"fp16", device,
        #                weight_quantization:{enabled:True, bits:4, double_quant:True, quant_type:"nf4"}}
        # Chama src.runner.loader.load_model(config)
        # Se mode != "fp16": monta quantize_fn / dequantize_fn via functools.partial

    @property
    def weights_mb(self) -> float:
        # src.metrics.collector.current_memory_mb()

    def generate(
        self,
        messages: list[dict],           # [{"role": "user"|"system"|"assistant", "content": str}]
        max_new_tokens: int,
        streamer: "TextIteratorStreamer",
    ) -> dict:
        # Retorna: {prefill_tok_s, decode_tok_s, first_token_latency_s,
        #           total_time_s, kv_mb, weights_mb, output_tokens}
```

Fluxo interno de `generate()`:
1. Formatar prompt:
   - `tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)`
     se `hasattr(tokenizer, "apply_chat_template")`
   - Fallback: `"\n".join(f"{m['role']}: {m['content']}" for m in messages)`
2. Tokenizar e mover para device
3. `reset_peak()` → medir prefill: `model(**inputs, use_cache=True)` com `time.perf_counter`
4. Criar `QuantizedDynamicCache(quantize_fn, dequantize_fn, tracker)` se mode != "fp16"
5. Lançar `model.generate(..., streamer=streamer, past_key_values=cache)` em `threading.Thread`
6. `thread.join()` → coletar métricas e retornar dict

**Dependências:** passos 5, 6

---

### 10. Criar `runtime/llm_runtime/cli.py`

**O que fazer:** Typer app com 4 comandos + loop de chat compartilhado.

```python
app = typer.Typer(name="llm-runtime", add_completion=False,
                  help="Runtime LLM interativo com quantização TurboQuant")

@app.command()
def download(
    model: Annotated[str, typer.Argument()],
    token: Annotated[str | None, typer.Option("--token", "-t", envvar="HF_TOKEN")] = None,
    cache_dir: Annotated[str | None, typer.Option(
        "--cache-dir", envvar="MODELS_DIR",
        help="Diretório local onde salvar o modelo (default: runtime/models/)"
    )] = None,
) -> None:
    """Baixa modelo do HuggingFace Hub e salva em runtime/models/."""

@app.command()
def run(model, max_tokens, system_prompt, device) -> None:
    """Chat interativo FP16 sem quantização."""
    _chat_loop(model, "fp16", 4, max_tokens, system_prompt, device)

@app.command(name="run-turboquant")
def run_turboquant(model, bits, max_tokens, system_prompt, device) -> None:
    """Chat interativo com KV cache TurboQuant."""
    _chat_loop(model, "turboquant", bits, max_tokens, system_prompt, device)

@app.command(name="run-full-quant")
def run_full_quant(model, bits, max_tokens, system_prompt, device) -> None:
    """Chat com pesos INT4 (bitsandbytes NF4) + KV TurboQuant."""
    _chat_loop(model, "full_quant", bits, max_tokens, system_prompt, device)
```

`_chat_loop()` — função privada compartilhada:

```
def _chat_loop(model_name, mode, bits, max_tokens, system_prompt, device):
    monitor = GPUMonitor(); monitor.start()
    engine  = InferenceEngine(model_name, mode, bits, device)

    with RuntimeDisplay(model_name, mode, monitor) as display:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        while True:
            try:
                user_text = display.prompt_input()   # pausa Live, lê, retoma
            except (KeyboardInterrupt, EOFError):
                break
            if user_text.strip().lower() in ("exit", "quit", "sair"):
                break

            display.append_user(user_text)
            messages.append({"role": "user", "content": user_text})

            streamer = TextIteratorStreamer(engine.tokenizer, skip_special_tokens=True)
            gen_thread = Thread(target=engine.generate,
                                args=(messages, max_tokens, streamer), daemon=True)
            gen_thread.start()

            response_tokens = []
            for token in streamer:
                display.append_token(token)
                response_tokens.append(token)

            gen_thread.join()
            metrics = engine.last_metrics          # engine armazena métricas do último generate
            display.update_metrics(metrics)
            display.finalize_response()
            messages.append({"role": "assistant", "content": "".join(response_tokens)})

    monitor.stop()
```

**Dependências:** passos 7, 8, 9

---

## Riscos e Mitigações

| Risco | Probabilidade | Mitigação |
|-------|---------------|-----------|
| `pynvml` não inicializa (GPU não NVIDIA ou drivers antigos) | baixa | `try/except` em `GPUMonitor.__init__`; exibe `GPU: N/A` sem crash |
| `apply_chat_template` ausente em modelos antigos | média | `hasattr` check + fallback de concatenação simples |
| Rich Live + `input()` no mesmo terminal gera artefatos visuais | média | `display.prompt_input()` para e retoma o Live explicitamente |
| Conflito de pacote `src` entre pai e runtime | alta se mal implementado | solucionado: runtime usa `llm_runtime/` como pacote próprio, sem `src/` local |
| `uv sync` em `runtime/` precisa de acesso ao pai em `../` | baixa (path relativo) | path dep relativa `"../"` funciona em desenvolvimento local; documentado no README |
| `TextIteratorStreamer` bloqueia indefinidamente se generate falhar | baixa | `gen_thread.join(timeout=300)` + tratamento de exceção no callback |
| KV TurboQuant em `full_quant` aumenta `first_token_latency` ~20ms | confirmado | documentado no `--help` do comando |

---

## Fluxo de Uso (do zero)

```bash
cd runtime/
cp .env.example .env          # editar MODEL_NAME, HF_TOKEN, MODELS_DIR se necessário
make setup                    # uv sync → cria .venv/ local

make download MODEL=Qwen/Qwen2.5-7B-Instruct
# → salvo em runtime/models/models--Qwen--Qwen2.5-7B-Instruct/

make run-full-quant MODEL=Qwen/Qwen2.5-7B-Instruct BITS=4
# → carrega direto de runtime/models/ (sem acesso à internet)
```

---

## Critérios de Conclusão

- [ ] `cd runtime && make help` lista todos os targets com descrição
- [ ] `cd runtime && make setup` executa `uv sync` e cria `runtime/.venv/` isolado
- [ ] `cd runtime && make download MODEL=Qwen/Qwen2.5-0.5B-Instruct` baixa com progresso em `runtime/models/`
- [ ] `cd runtime && make run MODEL=...` abre chat FP16 com painel de métricas
- [ ] `cd runtime && make run-turboquant MODEL=... BITS=4` abre chat; KV MB < 250
- [ ] `cd runtime && make run-full-quant MODEL=...` abre chat; pesos ~5.3 GB + KV TurboQuant
- [ ] Painel exibe VRAM, RAM, GPU%, prefill tok/s, decode tok/s, latency, KV MB, pesos MB
- [ ] Tokens aparecem em streaming no painel durante geração
- [ ] `Ctrl+C` encerra limpo sem stack trace
- [ ] `make run` a partir da raiz do projeto pai **não funciona** (comandos isolados em `runtime/`)
- [ ] Nenhum arquivo fora de `runtime/` foi criado ou modificado
- [ ] Cada arquivo Python tem ≤ 300 linhas

---

## Fora do Escopo (deliberado)

- Histórico de conversa persistente em disco
- Múltiplos modelos em simultâneo
- Endpoint HTTP / REST
- Weight INT8 neste runtime
- Testes automatizados
- Alterações em qualquer arquivo do projeto pai
