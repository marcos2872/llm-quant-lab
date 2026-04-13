# ── variáveis ──────────────────────────────────────────────────────────────
PYTHON     := uv run python
RUN        := uv run
SRC        := src.main

# defaults configuráveis via linha de comando
MODEL      ?=
BITS       ?=
METHOD     ?= turboquant
CONFIG     ?=
RAW_DIR    ?= results/raw
OUTPUT_DIR ?= results/reports
RESULT_JSON ?=

.PHONY: setup env \
        baseline weight-quant kv-quant \
        eval-ppl eval-needle eval-tasks all-eval annotate-all \
        report \
        all clean help

# ── setup ───────────────────────────────────────────────────────────────────
setup:           ## Instala dependências e cria .env + pastas
	uv sync
	@[ -f .env ] || cp .env.example .env && echo "✓ .env criado"
	@mkdir -p results/raw results/reports models benchmarks/prompts benchmarks/long_context benchmarks/perplexity

env:             ## Só instala dependências (sem criar .env)
	uv sync

# ── fase 1: baseline ────────────────────────────────────────────────────────
baseline:        ## Roda inferência FP16 baseline  (MODEL=... RAW_DIR=... opcionais)
	$(PYTHON) -m $(SRC) baseline \
	  $(if $(MODEL),--model $(MODEL),) \
	  $(if $(CONFIG),--config $(CONFIG),) \
	  --output-dir $(RAW_DIR)

# ── fase 2: quantização de pesos ────────────────────────────────────────────
weight-quant:    ## Roda weight quant  (BITS=4|4,8  RAW_DIR=... opcionais)
	$(PYTHON) -m $(SRC) weight-quant \
	  $(if $(MODEL),--model $(MODEL),) \
	  $(if $(BITS),--bits $(BITS),) \
	  $(if $(CONFIG),--config $(CONFIG),) \
	  --output-dir $(RAW_DIR)

# ── fase 3: quantização de KV cache ─────────────────────────────────────────
kv-quant:        ## Roda KV cache quant  (METHOD=turboquant|kivi|uniform  BITS=4  RAW_DIR=...)
	$(PYTHON) -m $(SRC) kv-quant \
	  $(if $(MODEL),--model $(MODEL),) \
	  $(if $(METHOD),--method $(METHOD),) \
	  $(if $(BITS),--bits $(BITS),) \
	  $(if $(CONFIG),--config $(CONFIG),) \
	  --output-dir $(RAW_DIR)

# ── fase 4: avaliação de qualidade ──────────────────────────────────────────
eval-ppl:        ## Calcula perplexidade  (CONFIG=... RESULT_JSON=... opcionais)
	$(PYTHON) -m $(SRC) eval-ppl \
	  $(if $(MODEL),--model $(MODEL),) \
	  $(if $(CONFIG),--config $(CONFIG),) \
	  $(if $(RESULT_JSON),--result-json $(RESULT_JSON),)

eval-needle:     ## Avalia Needle-in-a-Haystack  (CONFIG=... RESULT_JSON=... opcionais)
	$(PYTHON) -m $(SRC) eval-needle \
	  $(if $(MODEL),--model $(MODEL),) \
	  $(if $(CONFIG),--config $(CONFIG),) \
	  $(if $(RESULT_JSON),--result-json $(RESULT_JSON),)

eval-tasks:      ## Avalia F1 / exact match  (CONFIG=... RESULT_JSON=... opcionais)
	$(PYTHON) -m $(SRC) eval-tasks \
	  $(if $(MODEL),--model $(MODEL),) \
	  $(if $(CONFIG),--config $(CONFIG),) \
	  $(if $(RESULT_JSON),--result-json $(RESULT_JSON),)

annotate-all:    ## Anota todos os JSONs em RAW_DIR com as 3 métricas de qualidade
	@echo "Anotando JSONs em $(RAW_DIR) ..."
	@for f in $(RAW_DIR)/baseline_*.json; do \
	  [ -f "$$f" ] || continue; \
	  echo "  [baseline] $$f"; \
	  $(PYTHON) -m $(SRC) eval-ppl    --result-json $$f; \
	  $(PYTHON) -m $(SRC) eval-needle --result-json $$f; \
	  $(PYTHON) -m $(SRC) eval-tasks  --result-json $$f; \
	done
	@for f in $(RAW_DIR)/weight_quant_*.json; do \
	  [ -f "$$f" ] || continue; \
	  echo "  [weight_quant] $$f"; \
	  $(PYTHON) -m $(SRC) eval-ppl    --config configs/weight_quant.yaml --result-json $$f; \
	  $(PYTHON) -m $(SRC) eval-needle --config configs/weight_quant.yaml --result-json $$f; \
	  $(PYTHON) -m $(SRC) eval-tasks  --config configs/weight_quant.yaml --result-json $$f; \
	done
	@for f in $(RAW_DIR)/kv_quant_*.json; do \
	  [ -f "$$f" ] || continue; \
	  echo "  [kv_quant] $$f"; \
	  $(PYTHON) -m $(SRC) eval-ppl    --config configs/kv_quant.yaml --result-json $$f; \
	  $(PYTHON) -m $(SRC) eval-needle --config configs/kv_quant.yaml --result-json $$f; \
	  $(PYTHON) -m $(SRC) eval-tasks  --config configs/kv_quant.yaml --result-json $$f; \
	done
	@echo "✓ Anotação concluída."

all-eval: annotate-all  ## Alias para annotate-all

# ── fase 5: relatório ────────────────────────────────────────────────────────
report:          ## Gera summary.csv + gráficos  (RAW_DIR=... OUTPUT_DIR=... opcionais)
	$(PYTHON) -m $(SRC) report \
	  --raw-dir $(RAW_DIR) \
	  --output-dir $(OUTPUT_DIR)

# ── pipelines completos ───────────────────────────────────────────────────────
all: baseline weight-quant kv-quant annotate-all report  ## Pipeline completo
	@echo "✓ Pipeline completo finalizado → $(OUTPUT_DIR)/"

# ── utilidades ────────────────────────────────────────────────────────────────
clean:           ## Remove artefatos gerados (raw JSONs, CSVs, PNGs)
	rm -f results/raw/*.json
	rm -f results/reports/*.csv results/reports/*.png

help:            ## Lista todos os targets disponíveis
	@grep -E '^[a-zA-Z_-]+:.*##' $(MAKEFILE_LIST) \
	  | awk 'BEGIN {FS = ":.*##"}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'
