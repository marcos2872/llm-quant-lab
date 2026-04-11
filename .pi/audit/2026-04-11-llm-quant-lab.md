## Relatório de Qualidade de Código
**Data:** 2026-04-11
**Escopo:** repositório completo (`src/`)
**Stack detectada:** Python 3.11, PyTorch, Transformers, Typer

---

### Ferramentas Automáticas

#### ruff check src/
Nenhum problema encontrado.

#### pytest tests/
Diretório `tests/` inexistente — zero testes executados.

---

### Arquitetura

- [AVISO] `src/runner/baseline.py`, `src/runner/kv_quant.py`, `src/runner/weight_quant.py` — funções `_load_prompts` e `_resolve_device` duplicadas nos três arquivos (mesma implementação). Viola o princípio DRY; deveriam viver em `src/runner/loader.py` ou em um módulo `src/runner/_utils.py`.
- [AVISO] `src/runner/kv_quant.py:23`, `src/runner/baseline.py:20`, `src/runner/weight_quant.py:23` — importam símbolos privados `_current_memory_mb`, `_peak_memory_mb`, `_reset_peak` de `src.metrics.collector`. Símbolos prefixados com `_` são convencionalmente de uso interno do módulo. Torná-los públicos (remover o `_`) ou encapsulá-los em uma API pública é o caminho correto.
- [AVISO] `src/main.py:137-144` — `kv_quant()` grava config em arquivo temporário com `NamedTemporaryFile(delete=False)` para passar ao runner. Abordagem desnecessariamente complexa: `run_kv_quant` poderia aceitar um `dict` diretamente, eliminando o arquivo temporário.

---

### Python — Estilo e Convenções

#### Tamanho de funções (limite: 40 linhas)
- [AVISO] `src/runner/weight_quant.py:46` — `run_weight_quant` = **75 linhas**
- [AVISO] `src/runner/kv_quant.py:92` — `run_kv_quant` = **74 linhas**
- [AVISO] `src/quantization/methods/turboquant.py:76` — `quantize_turboquant` = **72 linhas**
- [AVISO] `src/eval/perplexity.py:23` — `eval_perplexity` = **66 linhas**
- [AVISO] `src/runner/loader.py:80` — `load_model` = **62 linhas**
- [AVISO] `src/eval/needle.py:56` — `eval_needle` = **60 linhas**
- [AVISO] `src/reporter/plots.py:90` — `plot_quality_tradeoff` = **59 linhas**
- [AVISO] `src/eval/task_score.py:58` — `eval_task_score` = **56 linhas**
- [AVISO] `src/metrics/collector.py:105` — `measure_throughput` = **52 linhas**
- [AVISO] `src/runner/baseline.py:45` — `run_baseline` = **55 linhas**
- [AVISO] `src/runner/kv_quant.py:37` — `_get_quant_fns` = **42 linhas**
- [AVISO] `src/quantization/methods/kivi.py:19` — `quantize_kivi` = **46 linhas**

#### Type hints
- [AVISO] `src/runner/kv_quant.py:42,52,67` — closures `q(t)` sem anotação de argumento e retorno. São funções públicas de facto (passadas como callbacks); anotar como `(t: torch.Tensor) -> tuple[torch.Tensor, dict]`.
- [AVISO] `src/quantization/kv_hooks.py:82` — `hook(module, inputs, outputs)` sem anotações. Embora seja closure interna, o padrão de hooks PyTorch tem assinatura conhecida.

#### Dead code / API confusa
- [AVISO] `src/metrics/collector.py:95` — `measure_memory_snapshot(after_generation_fn: None = None)` — parâmetro `after_generation_fn` não é utilizado dentro da função nem em nenhum chamador. É dead code e confunde a interface.
- [AVISO] `src/runner/weight_quant.py:79` — `import torch` dentro do loop `for bits in bits_to_run`, ou seja, dentro de função mas poderia estar no topo do arquivo. Import lazy sem motivo aparente.

#### Segurança de recursos
- [ERRO] `src/runner/kv_quant.py:92` — `run_kv_quant` não usa `try/finally` para garantir que `remove_kv_hooks(handles)` seja chamado em caso de exceção. Se `measure_throughput` lançar erro, os hooks ficam instalados permanentemente no modelo, corrompendo todas as inferências subsequentes na mesma sessão.
- [AVISO] `src/main.py:137` — `tempfile.NamedTemporaryFile(delete=False)` sem `try/finally`. Se `run_kv_quant` lançar exceção, o arquivo temporário `.yaml` fica em disco.

---

### Segurança
Nenhum problema encontrado. Sem tokens, senhas ou dados sensíveis hardcoded. `.env` está no `.gitignore`.

---

### Manutenção

- [AVISO] `src/quantization/methods/turboquant.py:120` — `_lloyd_max_codebook` é chamado **a cada invocação de `quantize_turboquant`**, ou seja, a cada token gerado durante `model.generate()`. O codebook é treinado do zero em ~20 iterações k-means para cada par K/V. Em sequências longas isso é significativo. O codebook deveria ser cacheado por `(bits, group_size)` após o primeiro cálculo, ou pré-computado antes do loop de geração.
- [SUGESTÃO] `configs/baseline.yaml` — as chaves foram reordenadas alfabeticamente pelo ruff (via git diff anterior), removendo os comentários de cabeçalho (`# Configuração do runner baseline...`). Os configs `weight_quant.yaml` e `kv_quant.yaml` ainda têm comentários; o `baseline.yaml` ficou sem. Padronizar.
- [SUGESTÃO] `src/quantization/kv_hooks.py` — o módulo `typing` é importado apenas para `Any`. Desde Python 3.11, `Any` pode ser importado de `typing` normalmente, mas se o projeto já usa `from collections.abc import Callable`, seria consistente verificar se `Any` de `typing` é a fonte correta (é — `collections.abc` não tem `Any`). Sem problema funcional; apenas nota de consistência.
- [SUGESTÃO] `tests/` inexistente. Nenhuma cobertura dos 3 métodos de quantização, hooks, métricas ou avaliadores. Risco alto dado que a lógica de quantização/dequantização é matematicamente delicada (shape, dtype, padding, rotação inversa).

---

### Resumo

| Severidade | Quantidade |
|---|---|
| **Erros** | 1 |
| **Avisos** | 18 |
| **Sugestões** | 3 |

**Prioridade 1 — Erro crítico:**
`run_kv_quant` sem `try/finally` — hooks vazam em exceção, corrompem o modelo em memória.

**Prioridade 2 — Avisos de manutenção alto impacto:**
- Codebook lloyd-max recomputado a cada token (performance)
- `_load_prompts` / `_resolve_device` triplicados entre runners
- Importação de símbolos privados `_current_memory_mb` etc. entre módulos
- Arquivo tempfile sem cleanup garantido

**Prioridade 3 — Convenções:**
- 12 funções acima de 40 linhas
- 3 closures sem type hints
- `measure_memory_snapshot` com parâmetro morto

**Próximo passo sugerido:** corrigir o erro do `try/finally` em `run_kv_quant` e mover `_load_prompts`/`_resolve_device` para um módulo compartilhado — são as duas mudanças de maior impacto real.

---
_Relatório salvo em: `.pi/audit/2026-04-11-llm-quant-lab.md`_
