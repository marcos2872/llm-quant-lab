# Plano: Correção dos Problemas dos Resultados `results/long/`

**Data:** 2026-04-14
**Autor:** agente-plan
**Status:** concluído

---

## Objetivo

Corrigir os 3 problemas confirmados na validação dos resultados do servidor
(`results/long/`): (1) PPL inválida para todos os runs KV quant por falha
silenciosa dos hooks no Qwen2.5, (2) colapso total de qualidade nos métodos
Uniform e KIVI em contexto longo, (3) inconsistência na métrica `kv_mb`
entre baseline e runs KV quant.

---

## Escopo

**Dentro do escopo:**
- Corrigir `eval/perplexity.py` + `quantization/kv_hooks.py` para que a PPL
  seja medida com quantização de KV realmente ativa
- Corrigir `quantization/methods/uniform.py` (per-head ao invés de global min-max)
- Diagnosticar e corrigir `quantization/methods/kivi.py`
- Unificar a métrica de KV adicionando `kv_theoretical_mb` (via `compute_kv_mb`)
  para todos os runs (baseline, weight quant e kv quant)
- Atualizar `reporter/csv_writer.py` para incluir a nova coluna no CSV consolidado

**Fora do escopo:**
- Weight INT8 prefill lento — comportamento real do bitsandbytes LLM.int8()
  sem kernels fusionados; apenas documentar
- Truncamento precoce de weight_4bit em `lc_bio_01` — EOS legítimo emitido
  pelo modelo quantizado; comportamento esperado
- Re-execução automática dos experimentos do servidor (feita manualmente após
  as correções)
- Melhora de qualidade do TurboQuant (já performa bem)
- Suporte a arquiteturas além de Qwen2.5

---

## Contexto Técnico (diagnóstico da validação)

### P1 — PPL inválida para KV quant (🔴 crítico)

`eval_perplexity` usa `install_kv_hooks` de `kv_hooks.py`, que instala um
`register_forward_hook` nos módulos de atenção e tenta encontrar
`past_key_values` no output. Porém, `Qwen2.5Attention.forward` retorna apenas
`(attn_output, attn_weights)` — o cache é atualizado internamente via
`past_key_values.update(key_states, value_states, self.layer_idx, ...)` e
**não** é incluído no output tuple. O hook nunca encontra nenhum objeto cache
para quantizar → a PPL é avaliada sem quantização ativa → valor idêntico ao
baseline para todos os métodos KV.

Confirmado inspecionando:
```
/.venv/lib/.../transformers/models/qwen2/modeling_qwen2.py:L183
return attn_output, attn_weights   # ← sem past_key_values
```

### P2 — Colapso de qualidade em Uniform e KIVI (🔴 crítico)

Uniform 4-bit global: `t_min`/`t_max` calculados sobre **todos** os
`1 × 4 × 4000 × 128 = 2 048 000` valores do tensor KV. Outliers dominam o
range → scale alto → maioria dos valores mapeados para 2-3 níveis dos 16
disponíveis → atenção completamente degradada → gibberish.

KIVI 4-bit: grupos ao longo de `head_dim` (64 dims/grupo). Pode apresentar o
mesmo problema de range se o paper original propõe agrupamento ao longo de
`seq_len` (tokens consecutivos) para melhor localidade estatística. O diagnóstico
no Passo 1 confirma qual eixo é o correto.

### P3 — `kv_mb` incomparável entre baseline e KV quant (🟡 médio)

- **Baseline**: `kv_mb` = `kv_delta` = pico de memória GPU durante `generate()`
  menos memória antes = KV cache **+** ativações transitórias (~762 MB)
- **KV quant**: `kv_mb` = `sum(tracker)` = apenas bytes dos tensores comprimidos
  sem ativações (~54 MB para INT4)

A proporção 762/54 = 14× não representa redução real de 14×; representa
comparação inválida de grandezas diferentes. A métrica analítica `compute_kv_mb`
(já existente em `_utils.py`) fornece uma base de comparação justa para todos
os métodos.

---

## Arquivos Afetados

| Arquivo | Ação | Motivo |
|---|---|---|
| `src/quantization/kv_hooks.py` | modificar | adicionar `install_kv_proj_hooks` que intercepta `k_proj`/`v_proj` |
| `src/eval/perplexity.py` | modificar | trocar `install_kv_hooks` por `install_kv_proj_hooks` |
| `src/quantization/methods/uniform.py` | modificar | trocar global min-max por quantização per-head |
| `src/quantization/methods/kivi.py` | modificar | corrigir dimensão de agrupamento conforme diagnóstico |
| `src/runner/_utils.py` | modificar | adicionar `kv_theoretical_mb` em `measure_prompt` |
| `src/reporter/csv_writer.py` | modificar | incluir coluna `kv_theoretical_mb` ao agregar JSONs |

---

## Sequência de Execução

### 1. Diagnóstico de erro de reconstrução (Uniform vs KIVI)

**Arquivos:** `src/quantization/methods/uniform.py`,
`src/quantization/methods/kivi.py`  
**O que fazer:**  
Criar um script de diagnóstico temporário (não commitado) que instancia um
tensor sintético com a forma real dos KV do experimento
`(1, 4, 4000, 128)` — com valores amostrados de uma Normal(0,1) e com outliers
sintéticos adicionados em 5% dos canais (simulando a distribuição real de atenção)
— e mede o MSE de reconstrução para:
- uniform 4-bit global (atual)
- uniform 4-bit per-head (proposta)
- kivi 4-bit grupos ao longo de `head_dim` (atual, `group_size=64`)
- kivi 4-bit grupos ao longo de `seq_len` (proposta, grupos de 64 tokens)

Critério: o diagnóstico confirma qual variante tem MSE < 0.01 para 4-bit e qual
colapsa. Com isso, a causa raiz de Uniform e KIVI é confirmada antes de editar
código de produção.  
**Dependências:** nenhuma

### 2. Corrigir `uniform.py`: quantização per-head

**Arquivo:** `src/quantization/methods/uniform.py`  
**O que fazer:**  
Alterar `quantize_uniform` para calcular `t_min`/`t_max` **por head**:
- Receber tensor de forma `(batch, heads, seq, head_dim)`
- Computar `t_min = tensor.min(dim=-1).values.min(dim=-1).values` →
  shape `(batch, heads)` — mínimo por combinação batch×head
- Computar `scale` idem
- Armazenar `scale` e `zero_point` como tensores no `meta` (não mais scalars)

Adaptar `dequantize_uniform` para broadcast correto:
- `scale` shape `(batch, heads, 1, 1)` faz broadcast sobre `(batch, heads, seq, head_dim)`

O packing/unpacking de bits (`_pack_indices_flat`, `_unpack_indices_flat`)
permanece inalterado — opera sobre o tensor achatado independentemente.  
**Dependências:** resultado do Passo 1 (confirma que per-head resolve o problema)  
**Justificativa:** min-max global com 4000 tokens permite que poucos outliers
de magnitude extrema expandam o range, colocando a maioria dos valores úteis
em 2-3 bins dos 16 disponíveis. Per-head isola o range de cada cabeça,
reduzindo drásticamente o MSE de reconstrução.

### 3. Investigar e corrigir `kivi.py`

**Arquivo:** `src/quantization/methods/kivi.py`  
**O que fazer:**  
Com o resultado do Passo 1, aplicar a correção identificada:

- **Se causa for dimensão de agrupamento** (agrupamento ao longo de `seq_len`
  tem MSE << agrupamento ao longo de `head_dim`): refatorar `_pad_and_group`
  para agrupar tokens consecutivos ao longo da dimensão de sequência
  (dim=-2 no tensor KV). Grupos de `group_size` tokens consecutivos por head.
  A estatística dentro de cada grupo será mais homogênea do que ao longo dos
  128 dims do head.

- **Se ambas as abordagens tiverem MSE similar**: manter a implementação atual
  e investigar outro root cause (ex.: overflow em _unpack para sequências longas).

Independente da causa: adicionar assert de sanidade no início de
`quantize_kivi` que verifica `tensor.ndim == 4` e loga o shape para facilitar
depuração futura.  
**Dependências:** resultado do Passo 1

### 4. Corrigir `kv_hooks.py`: adicionar `install_kv_proj_hooks`

**Arquivo:** `src/quantization/kv_hooks.py`  
**O que fazer:**  
Adicionar função pública `install_kv_proj_hooks`:

```python
def install_kv_proj_hooks(
    model: torch.nn.Module,
    quantize_fn: Callable,
    dequantize_fn: Callable,
) -> tuple[list[Any], list[float]]:
    """
    Instala hooks após k_proj e v_proj de cada camada de atenção.

    Compatível com qualquer arquitetura onde o cache é atualizado
    internamente (ex: Qwen2.5) e o attention module não retorna
    past_key_values no output.
    """
```

Lógica interna:
1. Para cada camada de atenção encontrada por `_find_attention_layers`,
   buscar os sub-módulos `k_proj` e `v_proj` (Linear layers)
2. Instalar `register_forward_hook` em cada um que aplica
   `quantize_fn` seguido de `dequantize_fn` sobre o output
3. Retornar `(handles, kv_mem_tracker)` com mesma interface que
   `install_kv_hooks`

Manter `install_kv_hooks` existente sem alterações (usado por arquiteturas
que retornam `past_key_values` no output de atenção).  
**Dependências:** nenhuma  
**Justificativa:** `k_proj` e `v_proj` são as projeções que geram os tensores
K e V armazenados no cache. Quantizar e dequantizar seu output antes do uso em
atenção é semanticamente equivalente a quantizar o KV cache — e funciona
independentemente de como o cache é gerenciado internamente pelo modelo.

### 5. Corrigir `eval/perplexity.py`: usar `install_kv_proj_hooks`

**Arquivo:** `src/eval/perplexity.py`  
**O que fazer:**  
Na função `eval_perplexity`, dentro do bloco `if quantize_fn is not None`:

```python
# ANTES:
from src.quantization.kv_hooks import install_kv_hooks
handles, _ = install_kv_hooks(model, quantize_fn, dequantize_fn)

# DEPOIS:
from src.quantization.kv_hooks import install_kv_proj_hooks
handles, _ = install_kv_proj_hooks(model, quantize_fn, dequantize_fn)
```

O parâmetro `kv_mem_tracker` retornado é ignorado na PPL (não medimos bytes
aqui), então a interface permanece compatível.  
**Dependências:** Passo 4 (função `install_kv_proj_hooks` existir)  
**Obs:** o chamador em `main.py` (`eval-ppl`) não requer nenhuma mudança.

### 6. Adicionar `kv_theoretical_mb` em `_utils.py`

**Arquivo:** `src/runner/_utils.py`  
**O que fazer:**  
Em `measure_prompt`, após calcular `kv_mb`, adicionar:

```python
kv_theoretical_mb = compute_kv_mb(model, throughput.input_tokens + throughput.output_tokens)
```

Incluir no dict retornado por `m.to_dict()` via `RunMetrics` ou via merge
direto no dicionário — a opção mais simples é adicionar a chave
diretamente ao dict retornado após `m.to_dict()`:

```python
result = m.to_dict()
result["kv_theoretical_mb"] = round(kv_theoretical_mb, 2)
return result
```

Isso garante que **todos** os runs (baseline, weight_quant, kv_quant) tenham
a mesma métrica de KV comparável, calculada analiticamente a partir do
config do modelo.  
**Dependências:** nenhuma (`compute_kv_mb` já existe em `_utils.py`)

### 7. Atualizar `reporter/csv_writer.py`

**Arquivo:** `src/reporter/csv_writer.py`  
**O que fazer:**  
Adicionar `kv_theoretical_mb` na lista de colunas coletadas ao agregar os
JSONs de resultado. Campo é opcional nos JSONs antigos — usar `.get(..., None)`
para retrocompatibilidade com arquivos sem esse campo.  
**Dependências:** Passo 6

---

## Riscos e Mitigações

| Risco | Probabilidade | Mitigação |
|---|---|---|
| KIVI ter causa raiz diferente da hipótese (dimensão) | Média | Passo 1 confirma antes de qualquer mudança de código |
| `k_proj`/`v_proj` hook quantizar também tokens sem cache (prefill total) | Baixa | É o comportamento desejado para PPL: simula o impacto de quantizar todos os K/V |
| Corrigir Uniform/KIVI mas qualidade ainda insuficiente a 2-bit | Alta (esperado) | 2-bit é compressão extrema; documentar como limitação intrínseca |
| `kv_theoretical_mb` divergir do tracker (kv_mb atual) | Baixa | Tracker já foi validado vs teórico com erro < 1% para uniform e KIVI |
| `install_kv_proj_hooks` afetar outras camadas além de atenção | Baixa | Busca por `k_proj`/`v_proj` apenas dentro dos módulos de atenção identificados por `_find_attention_layers` |

---

## Fora do Escopo (deliberado)

- **weight_8bit prefill lento**: comportamento documentado do bitsandbytes
  LLM.int8() sem kernels fusionados para batch=1 com contexto 4k. Não é bug.
- **weight_4bit truncamento precoce** (`lc_bio_01`, 16 tokens): EOS legítimo
  emitido com pesos INT4. Não suprimir sem justificativa experimental.
- **Re-execução dos experimentos do servidor**: feita manualmente após merge
  das correções.
- **Melhorias de TurboQuant**: já é o método de melhor qualidade; artefatos
  léxicos residuais são aceitáveis a 2-bit.

---

## Critérios de Conclusão

- [ ] `eval-ppl --result-json kv_quant_uniform_4bit_*.json` produz PPL
  **diferente** do baseline (confirma que os hooks agora estão ativos)
- [ ] `eval-ppl --result-json kv_quant_kivi_4bit_*.json` idem
- [ ] Geração de texto com `kv-quant --method uniform --bits 4` em prompt
  de ~4k tokens produz texto coerente (sem repetições/gibberish/tokens aleatórios)
- [ ] Geração de texto com `kv-quant --method kivi --bits 4` idem
- [ ] Campo `kv_theoretical_mb` presente em todos os JSONs após novo run
- [ ] `summary.csv` contém coluna `kv_theoretical_mb`
- [ ] `kv_theoretical_mb` do baseline ≈ 219 MB para seq ~4000 tokens
  (Qwen2.5-7B: 28 camadas, 4 KV heads, 128 head_dim, 2 bytes FP16)
- [ ] `uv run pytest tests/` passa sem regressões (quando testes existirem)
