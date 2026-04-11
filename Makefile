# ── variáveis ──────────────────────────────────────────────────────────────
PYTHON  := uv run python
RUN     := uv run
SRC     := src.main

# defaults configuráveis via linha de comando
MODEL   ?=
BITS    ?=
METHOD  ?= turboquant
CONFIG  ?=

.PHONY: setup env \
        baseline weight-quant kv-quant \
        eval-ppl eval-needle eval-tasks all-eval \
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
baseline:        ## Roda inferência FP16 baseline  (MODEL=... opcional)
	$(PYTHON) -m $(SRC) baseline \
	  $(if $(MODEL),--model $(MODEL),) \
	  $(if $(CONFIG),--config $(CONFIG),)

# ── fase 2: quantização de pesos ────────────────────────────────────────────
weight-quant:    ## Roda weight quant  (BITS=4 ou BITS=4,8 opcional)
	$(PYTHON) -m $(SRC) weight-quant \
	  $(if $(MODEL),--model $(MODEL),) \
	  $(if $(BITS),--bits $(BITS),) \
	  $(if $(CONFIG),--config $(CONFIG),)

# ── fase 3: quantização de KV cache ─────────────────────────────────────────
kv-quant:        ## Roda KV cache quant  (METHOD=turboquant|kivi|uniform  BITS=4)
	$(PYTHON) -m $(SRC) kv-quant \
	  $(if $(MODEL),--model $(MODEL),) \
	  $(if $(METHOD),--method $(METHOD),) \
	  $(if $(BITS),--bits $(BITS),) \
	  $(if $(CONFIG),--config $(CONFIG),)

# ── fase 4: avaliação de qualidade ──────────────────────────────────────────
eval-ppl:        ## Calcula perplexidade no WikiText-2  (RESULT_JSON=... para anotar)
	$(PYTHON) -m $(SRC) eval-ppl \
	  $(if $(MODEL),--model $(MODEL),) \
	  $(if $(RESULT_JSON),--result-json $(RESULT_JSON),)

eval-needle:     ## Avalia Needle-in-a-Haystack  (RESULT_JSON=... para anotar)
	$(PYTHON) -m $(SRC) eval-needle \
	  $(if $(MODEL),--model $(MODEL),) \
	  $(if $(RESULT_JSON),--result-json $(RESULT_JSON),)

eval-tasks:      ## Avalia F1 / exact match em prompts QA  (RESULT_JSON=... para anotar)
	$(PYTHON) -m $(SRC) eval-tasks \
	  $(if $(MODEL),--model $(MODEL),) \
	  $(if $(RESULT_JSON),--result-json $(RESULT_JSON),)

all-eval: eval-ppl eval-needle eval-tasks  ## Roda todas as avaliações de qualidade

# ── fase 5: relatório ────────────────────────────────────────────────────────
report:          ## Gera summary.csv + gráficos em results/reports/
	$(PYTHON) -m $(SRC) report

# ── pipelines completos ───────────────────────────────────────────────────────
all: baseline weight-quant kv-quant all-eval report  ## Pipeline completo
	@echo "✓ Pipeline completo finalizado → results/reports/"

# ── utilidades ────────────────────────────────────────────────────────────────
clean:           ## Remove artefatos gerados (raw JSONs, CSVs, PNGs)
	rm -f results/raw/*.json
	rm -f results/reports/*.csv results/reports/*.png

help:            ## Lista todos os targets disponíveis
	@grep -E '^[a-zA-Z_-]+:.*##' $(MAKEFILE_LIST) \
	  | awk 'BEGIN {FS = ":.*##"}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'
