## Relatório de QA — Pós-Refatoração
**Data:** 2026-04-11
**Escopo:** repositório completo (`src/`)
**Baseline:** auditoria anterior de 2026-04-11

---

### Ferramentas Automáticas

#### ruff check src/
Nenhum problema encontrado.

#### pytest tests/
Diretório `tests/` inexistente — zero testes executados.

---

### Progresso desde a última auditoria

| Item anterior | Status |
|---|---|
| `try/finally` em `run_kv_quant` | ✅ corrigido |
| `_load_prompts` / `_resolve_device` triplicados | ✅ centralizados em `_utils.py` |
| Símbolos privados importados externamente | ✅ renomeados para API pública |
| Tempfile sem cleanup em `main.py` | ✅ eliminado (`config_override`) |
| Codebook lloyd-max recomputado por token | ✅ cacheado por `(bits, group_size)` |
| Funções > 40 linhas (12 funções) | ✅ reduzido para 2 (run_kv_quant=57, load_model=43) |
| Type hints em closures `q(t)` | ✅ substituídas por `functools.partial` |
| Parâmetro morto `after_generation_fn` | ✅ removido |
| Import lazy de `torch` em loop | ✅ movido para topo |

---

### Bugs e Inconsistências

#### Risco ALTO

- [ALTO] `src/quantization/methods/turboquant.py:68` — `_split_channels` — quando
  `outlier_channels >= head_dim`, `torch.tensor([])` produz `dtype=float32` em vez de
  `int64`. Usar `float32` como índice de tensor lança `IndexError` imediatamente.
  Este caso ocorre naturalmente quando `outlier_channels` no config (padrão 32) for
  maior ou igual ao `head_dim` do modelo (ex: modelos com `head_dim=16` ou 32).
  **Risco:** `make kv-quant` trava com IndexError sem mensagem de diagnóstico útil.
  **Cenário de teste:** `quantize_turboquant(torch.randn(1,2,4,32), outlier_channels=64)`.
  **Sugestão:** `torch.tensor(lst, dtype=torch.long)` ou adicionar `normal_idx = normal_idx.long()`.

#### Risco MÉDIO

- [MÉDIO] `src/quantization/methods/turboquant.py` — `_codebook_cache` é um dicionário
  global com chave `(bits, group_size)`. O codebook é treinado na primeira chamada
  com esses parâmetros e reutilizado em todas as chamadas subsequentes **na mesma
  sessão Python**, incluindo runs com modelos diferentes. Se `make baseline` e
  `make kv-quant` forem executados na mesma sessão com parâmetros iguais, o codebook
  treinado na distribuição do modelo A será usado para o modelo B.
  Na prática, o CLI cria um processo por comando (não há sessão compartilhada), então
  o risco real é baixo. Mas em notebooks ou scripts que encadeiam múltiplas runs, pode
  produzir resultados silenciosamente incorretos.
  **Sugestão:** Incluir `model_name` na chave do cache ou documentar a limitação claramente.

- [MÉDIO] `src/runner/kv_quant.py:95` `run_kv_quant` ainda tem 57 linhas (limite: 40).
  Igualmente `src/runner/loader.py:99` `load_model` tem 43 linhas.
  Não crítico mas viola a convenção declarada.

- [MÉDIO] `src/runner/kv_quant.py` e `src/runner/baseline.py` e `src/runner/weight_quant.py`
  ainda têm `_measure_prompt` com implementações quase idênticas (apenas a assinatura
  da versão `kv_quant` inclui `kv_mem_tracker`). A lógica central é idêntica.
  **Sugestão:** Mover para `_utils.py` com `kv_mem_tracker: list[float] | None = None`.

- [MÉDIO] `src/runner/kv_quant.py:113` — `run_kv_quant` retorna `Path()` (equivalente
  a `Path(".")`) quando `enabled=false`. `Path()` é truthy e `str(Path()) == "."`, então
  chamadores que fazem `if result:` ou `result.exists()` não detectam o caso de "não rodou".
  **Sugestão:** Retornar `None` e anotar o retorno como `Path | None`, ou levantar `typer.Exit(0)`.

#### Risco BAIXO

- [BAIXO] `src/eval/perplexity.py:32`, `src/eval/needle.py:53`, `src/eval/task_score.py:53`,
  `src/metrics/collector.py:141` — todos chamam `tokenizer(...).to(device)`. O retorno de
  um `AutoTokenizer` é um `BatchEncoding` (subclasse de `dict`) que tem método `.to()`.
  Isso funciona, mas se um tokenizer customizado retornar `dict` puro, o `.to()` não existirá.
  Risco baixo pois o projeto usa apenas tokenizers HuggingFace.

- [BAIXO] `src/reporter/csv_writer.py:91` — `except Exception as exc` faz `console.print`
  do erro mas continua processando. O arquivo CSV resultante pode ter menos linhas do que
  o esperado sem sinalização clara ao usuário no relatório final.

- [BAIXO] `src/quantization/kv_hooks.py:62` — `_find_attention_recursive` usa
  `list(module.children()) == []` para detectar folhas. Isso pode capturar módulos que
  não são de atenção mas não têm filhos (ex: `nn.LayerNorm`, `nn.Dropout`).
  Quando a arquitetura não é reconhecida e a busca recursiva é usada, hooks podem ser
  instalados em módulos incorretos, corrompendo silenciosamente a inferência.

---

### Vulnerabilidades de Segurança
Nenhum problema encontrado.

---

### Manutenção

- [SUGESTÃO] `tests/` ainda inexistente. O bug de `torch.tensor([])` teria sido capturado
  por um único teste parametrizado em `head_dim < outlier_channels`. Com a lógica de
  quantização matematicamente delicada, a ausência de testes é o maior risco de longo prazo.

- [SUGESTÃO] `src/main.py` tem 252 linhas. Considerando o limite de 300, está próximo.
  Extrair os handlers de avaliação (`eval_ppl`, `eval_needle`, `eval_tasks`) para
  `src/main_eval.py` manteria o arquivo principal abaixo do limite.

- [SUGESTÃO] A `_rotation_cache` e `_codebook_cache` em `turboquant.py` são globais de
  módulo sem nenhuma função de limpeza pública. Em sessões longas, a memória cresce
  indefinidamente com novas combinações `(dim, seed)` e `(bits, group_size)`.
  Expor `clear_caches()` seria boa prática.

---

### Resumo

| Severidade | Anterior | Agora |
|---|---|---|
| **Erros críticos** | 1 | 1 (novo — `torch.tensor([])`) |
| **Risco médio** | 4 | 4 |
| **Risco baixo** | 2 | 3 |
| **Sugestões** | 3 | 3 |

**Pior problema do ciclo anterior** (try/finally) foi corrigido.
**Pior problema atual:** `_split_channels` retorna índice `float32` quando `outlier_channels >= head_dim`, causando `IndexError` silencioso em modelos com `head_dim` pequeno.

**Ação imediata recomendada:** corrigir `torch.tensor(lst, dtype=torch.long)` em `_split_channels`.
