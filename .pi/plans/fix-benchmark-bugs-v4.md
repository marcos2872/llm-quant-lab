# Plano: Correção de Bugs v4 — Remover rot_scale; restaurar codebook teórico

**Data:** 2026-04-14
**Autor:** agente-plan
**Status:** concluído
**Referência:** `.pi/plans/fix-benchmark-bugs-v3.md`

---

## Contexto e histórico de falhas

| Rodada | Mudança em TurboQuant | PPL 4-bit | Needle 4-bit | Geração |
|---|---|---|---|---|
| v1 | outlier split no espaço rotacionado (bug original) | 5.312 | 0,90 | artefatos multilingues |
| v2 | outlier split removido + max-abs/dim + codebook N(0,0.3) | 4.005 | 0,90 | coerente ✅ |
| v3 | max-abs/dim → std/dim + codebook N(0,0.3) | — (PPL quebrada) | — | coerente ✅ |
| **v4** | std/dim + codebook N(0,1.0) clip(-5,5) | **1.709** | **0,50 ❌** | **colapso ❌** |

PPL melhorou mas geração piorou drasticamente em cada iteração. O padrão indica
que a causa-raiz do problema de PPL ainda não foi encontrada corretamente.

---

## Diagnóstico definitivo (confirmado numericamente)

### Por que cada abordagem falha ou funciona

Após normalização para S^{d-1} e rotação ortogonal R, as coordenadas de cada
vetor seguem N(0, 1/d) marginalmente. Este resultado é **exato e invariante a
qualquer transformação ortogonal anterior** (incluindo RoPE):

```
std global pré-RoPE:  0.088388  =  1/√128
std global pós-RoPE:  0.088388  =  1/√128
Diferença:            0.00000012   ← praticamente zero
```

| Escala | Distribuição após escala | Codebook | Problema |
|---|---|---|---|
| max-abs/dim (v2) | N(0, 0.27), range [-1,1] | N(0,0.3) | **Mismatch**: std=0.27 ≠ 0.3; sensível a RoPE por dim |
| std/dim (v4) | N(0, 1.0), range [-5,5] | N(0,1.0) | **Clipping**: 0.24% de valores além ±3σ mapeados ao centroide extremo → perde informação de chaves críticas → colapso de atenção |
| **Sem escala (original)** | N(0, 0.0884), range sem corte | **N(0, 1/√d)** | **Nenhum**: match perfeito, sem clipping, invariante a RoPE |

### Por que std/dim causa colapso mas max-abs/dim não

Com `std/dim`: valores > 3σ (~0,24% do total, mas potencialmente correspondentes
às chaves de atenção mais importantes) são mapeados ao centroide extremo do
codebook → perda de informação crítica → atenção torna-se quase uniforme →
modelo colapsa em tokens de alta frequência ("t", "E", "the").

Com `max-abs/dim`: range exatamente [-1, 1], **zero clipping** → todos os
valores, incluindo extremos, são preservados. O mismatch de codebook (std=0.27
vs N(0,0.3)) causa erro médio maior, mas os extremos críticos para atenção
são mantidos.

### A solução correta é remover toda a escala adaptativa

O codebook N(0, 1/√d) **sem qualquer rot_scale** é:
1. **Match perfeito**: std dos valores rotacionados = 1/√d = 0.0884 (exato)
2. **Zero clipping**: o Lloyd-Max cobre toda a distribuição real
3. **Invariante a RoPE**: std global = 1/√d independentemente de transformações
   ortogonais anteriores
4. **Sem overhead**: não armazena rot_scale no meta

A PPL catastrófica nas rodadas v1 era causada **exclusivamente** pelo outlier
split no espaço rotacionado (já removido em v2). O codebook original N(0, 1/√d)
sem escala já era correto; as mudanças de v2 e v3 introduziram problemas novos.

---

## Objetivo

Reverter `quantize_turboquant` e `dequantize_turboquant` ao design sem escala
adaptativa, restaurando o codebook teórico N(0, 1/√d) original.

---

## Escopo

**Dentro do escopo:**
- `src/quantization/methods/turboquant.py` — 4 remoções/substituições cirúrgicas

**Fora do escopo:**
- Qualquer outro arquivo

---

## Arquivos Afetados

| Arquivo | Ação | Motivo |
|---|---|---|
| `src/quantization/methods/turboquant.py` | modificar | Remover rot_scale; restaurar codebook N(0, 1/√d) |

---

## Sequência de Execução

### Passo 1 — Remover rot_scale de `quantize_turboquant`

**Localização:** bloco de escala + quantização (~L246-L265 do estado atual)

**Antes (v4 — errado):**
```python
# 3. Escala data-adaptive: std por dimensão → normaliza para N(0,1) por dim
rot_scale = rotated.std(dim=0, keepdim=True).clamp(min=1e-8)  # (1, head_dim)
rotated_scaled = rotated / rot_scale

# 4. Codebook único Lloyd-Max para N(0, 0.3) truncada em [-1,1] + quantização escalar
codebook = _get_theoretical_codebook(bits, head_dim, device)
q_all = _pack_indices(_scalar_quantize(rotated_scaled, codebook), bits)

meta = {
    "codebook": codebook,
    "rot_scale": rot_scale,   # ← remover
    ...
}
```

**Depois (correto — sem escala):**
```python
# 3. Codebook teórico N(0, 1/√d) + quantização escalar (sem escala adaptativa)
# Após S^{d-1} + rotação ortogonal: coordenadas ~ N(0, 1/d) exatamente.
# Codebook calibrado para esta distribuição, sem clipping, invariante ao RoPE.
codebook = _get_theoretical_codebook(bits, head_dim, device)
q_all = _pack_indices(_scalar_quantize(rotated, codebook), bits)

meta = {
    "codebook": codebook,
    # rot_scale removido
    ...
}
```

---

### Passo 2 — Remover rot_scale de `dequantize_turboquant`

**Localização:** bloco de reconstrução (~L282-L293 do estado atual)

**Antes (v4 — errado):**
```python
indices = _unpack_indices(quantized, bits, head_dim)
full_rotated_scaled = _scalar_dequantize(indices, meta["codebook"].to(device))

# 2. Desfaz a escala data-adaptive
full_rotated = full_rotated_scaled * meta["rot_scale"].to(device)

# 3. Rotação inversa Πᵀ
R = _get_rotation(head_dim, meta["rotation_seed"], device)
reconstructed = full_rotated @ R.T
```

**Depois (correto — sem escala):**
```python
indices = _unpack_indices(quantized, bits, head_dim)
full_rotated = _scalar_dequantize(indices, meta["codebook"].to(device))
# rot_scale removido — valores já estão na escala correta N(0, 1/√d)

R = _get_rotation(head_dim, meta["rotation_seed"], device)
reconstructed = full_rotated @ R.T
```

---

### Passo 3 — Restaurar codebook para N(0, 1/√d) em `_get_theoretical_codebook`

**Localização:** função `_get_theoretical_codebook`

**Antes (v4 — errado):**
```python
key = (bits, head_dim, "v3")  # v3: codebook N(0,1)
...
raw = rng.normal(0.0, 1.0, size=500_000).astype(np.float32)
samples = torch.from_numpy(raw.clip(-5.0, 5.0))
```

**Depois (correto — original com bump de versão):**
```python
key = (bits, head_dim, "v4")  # v4: codebook teórico N(0, 1/√d), sem rot_scale
...
# Distribuição teórica exata das coordenadas pós-rotação de vetores em S^{d-1}
# std = 1/√d; Lloyd-Max sem truncamento cobre toda a distribuição real
raw = rng.normal(0.0, 1.0 / (head_dim ** 0.5), size=200_000).astype(np.float32)
samples = torch.from_numpy(raw)   # sem clip — codebook se estende ao range real
```

O `clip` é removido pois o Lloyd-Max com amostras não-clipadas posiciona os
centroides extremos nos percentis corretos da cauda Gaussiana, cobrindo os
valores raros sem artificialmente comprimir o range.

---

### Passo 4 — Re-executar apenas TurboQuant e PPL

```bash
# Re-roda apenas os dois modos TurboQuant
make kv-quant METHOD=turboquant BITS=4 PROMPTS=benchmarks/prompts/long_context.jsonl
make kv-quant METHOD=turboquant BITS=2 PROMPTS=benchmarks/prompts/long_context.jsonl

# Re-avalia PPL dos dois modos (anota no JSON)
make eval-ppl

# Regenera relatório
make report
```

Baseline, weight_quant, kivi e uniform **não são rerodados** — sem mudanças.

---

## Riscos e Mitigações

| Risco | Prob. | Mitigação |
|---|---|---|
| Cache key "v3" ainda em memória → codebook N(0,1) reutilizado | Alta | Bump para "v4" na key invalida o cache |
| PPL TurboQuant ainda > 100 após fix | Baixa | Diagnóstico numérico confirma match perfeito; se persistir, investigar se install_kv_proj_hooks está ativo |
| "struggler" 2-bit persistir | Média | 2-bit (4 níveis) tem MSE teórico ≈ 35% da norma — pode ser limitação física, não bug |
| Geração 4-bit piorar novamente | Muito baixa | Sem clipping, sem escala → menos mecanismos de falha que v2/v3/v4 |

---

## Critérios de Conclusão

- [ ] PPL `kv_turboquant_4bit` < 20
- [ ] PPL `kv_turboquant_2bit` < 80
- [ ] `needle_recall` turboquant_4bit ≥ 0,90 (recuperar a regressão de v4)
- [ ] Outputs 4-bit: sem loops de caracteres únicos, sem "the the"
- [ ] `rot_scale` não presente em `meta` nem em `dequantize_turboquant`
- [ ] Apenas `src/quantization/methods/turboquant.py` modificado

---

## Resumo — 4 remoções, 3 linhas em 1 arquivo

```
_get_theoretical_codebook():
  key = (bits, head_dim, "v3")            →  key = (bits, head_dim, "v4")
  rng.normal(0.0, 1.0, 500_000).clip(-5)  →  rng.normal(0.0, 1/√d, 200_000)  [sem clip]

quantize_turboquant():
  REMOVER: rot_scale = rotated.std(...)
  REMOVER: rotated_scaled = rotated / rot_scale
  MUDAR:   _scalar_quantize(rotated_scaled, ...)  →  _scalar_quantize(rotated, ...)
  REMOVER: "rot_scale": rot_scale  do meta

dequantize_turboquant():
  REMOVER: full_rotated = full_rotated_scaled * meta["rot_scale"]
  RENOMEAR: full_rotated_scaled  →  full_rotated
```
