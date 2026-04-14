# ── variáveis ──────────────────────────────────────────────────────────────
PYTHON     := uv run python
SRC        := src.main

# Carrega .env se existir (define MODEL_NAME, HF_TOKEN, DEVICE…)
-include .env
export MODEL_NAME

# Sobrescrevíveis na linha de comando
MODEL      ?= $(MODEL_NAME)
BITS       ?=
METHOD     ?= turboquant
CONFIG     ?=
RAW_DIR    ?= results/raw
OUTPUT_DIR ?= results/reports
RESULT_JSON ?=
PROMPTS    ?= benchmarks/prompts/basic.jsonl

# Prompts de contexto longo (usados pelos pipelines long)
PROMPTS_LONG := benchmarks/prompts/long_context.jsonl
CONFIG_LONG  := configs/long_context.yaml

.PHONY: setup env \
        baseline weight-quant kv-quant \
        sweep-weight sweep-kv sweep-all \
        eval-ppl eval-needle eval-tasks annotate-all all-eval \
        context-sweep context-report \
        report \
        benchmark-7b benchmark-long \
        all clean clean-long clean-all help

# ── setup ───────────────────────────────────────────────────────────────────
setup:           ## Instala dependências e cria .env + pastas
	uv sync
	@[ -f .env ] || cp .env.example .env && echo "✓ .env criado"
	@mkdir -p results/raw results/reports results/long results/7b \
	          models benchmarks/prompts benchmarks/long_context benchmarks/perplexity

env:             ## Só instala dependências (sem criar .env)
	uv sync

# ── fase 1: baseline ────────────────────────────────────────────────────────
baseline:        ## Inferência FP16 baseline  (MODEL=... PROMPTS=... RAW_DIR=... opcionais)
	$(PYTHON) -m $(SRC) baseline \
	  $(if $(MODEL),--model $(MODEL),) \
	  $(if $(CONFIG),--config $(CONFIG),) \
	  --prompts $(PROMPTS) \
	  --output-dir $(RAW_DIR)

# ── fase 2: quantização de pesos ────────────────────────────────────────────
weight-quant:    ## Weight quant via bitsandbytes  (BITS=4|8|4,8  PROMPTS=... RAW_DIR=... opcionais)
	$(PYTHON) -m $(SRC) weight-quant \
	  $(if $(MODEL),--model $(MODEL),) \
	  $(if $(BITS),--bits $(BITS),) \
	  $(if $(CONFIG),--config $(CONFIG),) \
	  --prompts $(PROMPTS) \
	  --output-dir $(RAW_DIR)

sweep-weight:    ## Weight quant INT4 + INT8 em sequência
	$(MAKE) weight-quant BITS=4,8 \
	  $(if $(MODEL),MODEL=$(MODEL),) \
	  $(if $(CONFIG),CONFIG=$(CONFIG),) \
	  PROMPTS=$(PROMPTS) RAW_DIR=$(RAW_DIR)

# ── fase 3: quantização de KV cache ─────────────────────────────────────────
kv-quant:        ## KV cache quant  (METHOD=uniform|kivi|turboquant  BITS=4|2  PROMPTS=...  RAW_DIR=...)
	$(PYTHON) -m $(SRC) kv-quant \
	  $(if $(MODEL),--model $(MODEL),) \
	  $(if $(METHOD),--method $(METHOD),) \
	  $(if $(BITS),--bits $(BITS),) \
	  $(if $(CONFIG),--config $(CONFIG),) \
	  --prompts $(PROMPTS) \
	  --output-dir $(RAW_DIR)

sweep-kv:        ## KV cache quant — 3 métodos × 2 bits (6 runs)
	$(MAKE) kv-quant METHOD=uniform    BITS=4 $(if $(MODEL),MODEL=$(MODEL),) PROMPTS=$(PROMPTS) RAW_DIR=$(RAW_DIR) $(if $(CONFIG),CONFIG=$(CONFIG),)
	$(MAKE) kv-quant METHOD=kivi       BITS=4 $(if $(MODEL),MODEL=$(MODEL),) PROMPTS=$(PROMPTS) RAW_DIR=$(RAW_DIR) $(if $(CONFIG),CONFIG=$(CONFIG),)
	$(MAKE) kv-quant METHOD=turboquant BITS=4 $(if $(MODEL),MODEL=$(MODEL),) PROMPTS=$(PROMPTS) RAW_DIR=$(RAW_DIR) $(if $(CONFIG),CONFIG=$(CONFIG),)
	$(MAKE) kv-quant METHOD=uniform    BITS=2 $(if $(MODEL),MODEL=$(MODEL),) PROMPTS=$(PROMPTS) RAW_DIR=$(RAW_DIR) $(if $(CONFIG),CONFIG=$(CONFIG),)
	$(MAKE) kv-quant METHOD=kivi       BITS=2 $(if $(MODEL),MODEL=$(MODEL),) PROMPTS=$(PROMPTS) RAW_DIR=$(RAW_DIR) $(if $(CONFIG),CONFIG=$(CONFIG),)
	$(MAKE) kv-quant METHOD=turboquant BITS=2 $(if $(MODEL),MODEL=$(MODEL),) PROMPTS=$(PROMPTS) RAW_DIR=$(RAW_DIR) $(if $(CONFIG),CONFIG=$(CONFIG),)

# ── fase 4: avaliação de qualidade ──────────────────────────────────────────
# Notas sobre eval-tasks:
#   - Quando chamado via annotate-all (--result-json), o prompts_file é lido
#     diretamente do JSON de run — sem necessidade de --prompts explícito.
#   - Runs gerados por benchmark-long usam long_context.jsonl automaticamente.
#   - Runs gerados pelo pipeline padrão usam basic.jsonl (legado, QA curto).

eval-ppl:        ## Perplexidade no corpus WikiText-2  (CONFIG=... RESULT_JSON=... opcionais)
	$(PYTHON) -m $(SRC) eval-ppl \
	  $(if $(MODEL),--model $(MODEL),) \
	  $(if $(CONFIG),--config $(CONFIG),) \
	  $(if $(RESULT_JSON),--result-json $(RESULT_JSON),)

eval-needle:     ## Needle-in-a-Haystack (2k/4k/8k tokens)  (CONFIG=... RESULT_JSON=... opcionais)
	$(PYTHON) -m $(SRC) eval-needle \
	  $(if $(MODEL),--model $(MODEL),) \
	  $(if $(CONFIG),--config $(CONFIG),) \
	  $(if $(RESULT_JSON),--result-json $(RESULT_JSON),)

eval-tasks:      ## F1 / Exact Match em prompts QA  (CONFIG=... RESULT_JSON=... opcionais)
	$(PYTHON) -m $(SRC) eval-tasks \
	  $(if $(MODEL),--model $(MODEL),) \
	  $(if $(CONFIG),--config $(CONFIG),) \
	  $(if $(RESULT_JSON),--result-json $(RESULT_JSON),)

annotate-all:    ## Anota todos os JSONs em RAW_DIR com PPL + Needle + Task score
	@echo "Anotando JSONs em $(RAW_DIR) ..."
	@for f in \
	    $(RAW_DIR)/baseline_*.json \
	    $(RAW_DIR)/weight_quant_*.json \
	    $(RAW_DIR)/kv_quant_*.json; do \
	  [ -f "$$f" ] || continue; \
	  echo "  $$f"; \
	  $(PYTHON) -m $(SRC) eval-ppl    --result-json $$f; \
	  $(PYTHON) -m $(SRC) eval-needle --result-json $$f; \
	  $(PYTHON) -m $(SRC) eval-tasks  --result-json $$f; \
	done
	@echo "✓ Anotação concluída."

all-eval: annotate-all  ## Alias para annotate-all

# ── fase 5: relatório ────────────────────────────────────────────────────────
report:          ## Gera summary.csv + gráficos  (RAW_DIR=... OUTPUT_DIR=... opcionais)
	$(PYTHON) -m $(SRC) report \
	  --raw-dir $(RAW_DIR) \
	  --output-dir $(OUTPUT_DIR)

# ── context sweep (opcional) ─────────────────────────────────────────────────
context-sweep:   ## Escalonamento de contexto 512→4096 tokens  (CONFIG=... RAW_DIR=... opcionais)
	$(PYTHON) -m $(SRC) context-sweep \
	  $(if $(MODEL),--model $(MODEL),) \
	  $(if $(CONFIG),--config $(CONFIG),) \
	  --output-dir $(RAW_DIR)

context-report:  ## Gera context_scaling.png a partir dos dados de context-sweep
	$(PYTHON) -m $(SRC) context-report \
	  --raw-dir $(RAW_DIR) \
	  --output-dir $(OUTPUT_DIR)

# ── pipelines completos ───────────────────────────────────────────────────────
sweep-all:       ## Pipeline completo: baseline + weight + KV quant + eval + report
	$(MAKE) baseline     $(if $(MODEL),MODEL=$(MODEL),) PROMPTS=$(PROMPTS) $(if $(CONFIG),CONFIG=$(CONFIG),) RAW_DIR=$(RAW_DIR)
	$(MAKE) sweep-weight $(if $(MODEL),MODEL=$(MODEL),) PROMPTS=$(PROMPTS) $(if $(CONFIG),CONFIG=$(CONFIG),) RAW_DIR=$(RAW_DIR)
	$(MAKE) sweep-kv     $(if $(MODEL),MODEL=$(MODEL),) PROMPTS=$(PROMPTS) $(if $(CONFIG),CONFIG=$(CONFIG),) RAW_DIR=$(RAW_DIR)
	$(MAKE) annotate-all RAW_DIR=$(RAW_DIR)
	$(MAKE) report       RAW_DIR=$(RAW_DIR) OUTPUT_DIR=$(OUTPUT_DIR)
	@echo "✓ sweep-all concluído → $(OUTPUT_DIR)/"

benchmark-long:  ## Benchmark completo com prompts de ~4k tokens (Needle + PPL + Task-long)
	@echo "Limpando resultados anteriores em results/long/ ..."
	@rm -rf results/long
	$(MAKE) sweep-all \
	  PROMPTS=$(PROMPTS_LONG) \
	  CONFIG=$(CONFIG_LONG) \
	  RAW_DIR=results/long \
	  OUTPUT_DIR=results/long/report

benchmark-7b:    ## Benchmark completo para Qwen2.5-7B-Instruct (prompts padrão)
	$(MAKE) sweep-all \
	  MODEL=Qwen/Qwen2.5-7B-Instruct \
	  RAW_DIR=results/7b \
	  OUTPUT_DIR=results/7b/report

all:             ## Pipeline básico com 1 config por modo (prompts padrão)
	$(MAKE) baseline     PROMPTS=$(PROMPTS) RAW_DIR=$(RAW_DIR)
	$(MAKE) weight-quant PROMPTS=$(PROMPTS) RAW_DIR=$(RAW_DIR)
	$(MAKE) kv-quant     PROMPTS=$(PROMPTS) RAW_DIR=$(RAW_DIR)
	$(MAKE) annotate-all RAW_DIR=$(RAW_DIR)
	$(MAKE) report       RAW_DIR=$(RAW_DIR) OUTPUT_DIR=$(OUTPUT_DIR)
	@echo "✓ Pipeline básico finalizado → $(OUTPUT_DIR)/"

# ── utilidades ────────────────────────────────────────────────────────────────
clean:           ## Remove JSONs e relatórios de results/raw/
	rm -f results/raw/*.json
	rm -f results/reports/*.csv results/reports/*.png

clean-long:      ## Remove todos os artefatos de results/long/
	rm -rf results/long
	@mkdir -p results/long

clean-all:       ## Remove todos os artefatos de resultados (raw, long, 7b, reports)
	rm -rf results/raw results/reports results/long results/7b
	@mkdir -p results/raw results/reports results/long results/7b

help:            ## Lista todos os targets disponíveis
	@grep -hE '^[a-zA-Z_-]+:.*##' $(MAKEFILE_LIST) \
	  | awk 'BEGIN {FS = ":.*##"}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'
