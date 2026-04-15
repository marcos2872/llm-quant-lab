# Plano: Corrigir TurboQuant — Algorithm 2, Outlier Handling e kv_mb

**Data:** 2026-04-15
**Autor:** agente-plan
**Status:** aprovado

---

## Objetivo

Corrigir a implementação do TurboQuant para reproduzir fielmente o paper
*"TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate"*
(arXiv:2504.19874). Os três problemas identificados são:

1. O `turboquant.py` implementa apenas o **Algorithm 1 (TurboQuant_mse)**, que
   é biasado para produto interno — causa do colapso de PPL nos resultados.
2. O tratamento de **outlier channels** está desativado (`outlier_bits` e
   `outlier_channels` marcados como `# ignorado`), contrariando o §4.3 do paper.
3. A medição de `kv_mb` em `kv_cache.py` **não contabiliza** os tensores extras
   armazenados no `meta` do TurboQuant (norms, r_norms, qjl bits).

A meta é comparação justa entre métodos: FP16 baseline, weight quant INT4/INT8,
KV uniform, KV KIVI e **KV TurboQuant fiel ao paper**. Nenhuma quantização é
otimizada além do que o paper descreve.

---

## Escopo

**Dentro do escopo:**
- Implementar Algorithm 2 (`TurboQuant_prod`) no `turboquant.py`
- Restaurar detecção de outlier channels no espaço original (pré-normalização)
- Aplicar dois TurboQuant_prod independentes: outlier channels em `outlier_bits`,
  canais normais em `bits`
- Adicionar suporte a 1-bit em `_pack_indices` / `_unpack_indices`
- Adicionar cache `_qjl_cache` e helper `_get_qjl_matrix`
- Corrigir `_handle_prefill` em `kv_cache.py` para incluir todos os tensores
  de meta no cálculo de `kv_mb`

**Fora do escopo:**
- Nenhuma mudança em `uniform.py`, `kivi.py`, `kv_hooks.py`
- Nenhuma mudança em runners, evals, reporter ou main.py
- Nenhuma mudança em configs YAML ou benchmarks
- Não implementar entropy encoding (paper §3.1 menciona mas não usa)
- Não implementar 3-bit packing nativo (ver Risco 1)
- Não re-rodar o benchmark no servidor (escopo de execução separado)

---

## Contexto do Paper — Algoritmos a Implementar

### Algorithm 1 — TurboQuant_mse (já implementado, serve como sub-rotina)

```
Quant_mse(x ∈ S^{d-1}), bit-width = b:
  y = Π @ x                          # rotação ortogonal
  idx_j = argmin_k |y_j - c_k|       # scalar quantize per coord (codebook b-bits)
  return idx

DeQuant_mse(idx):
  ỹ_j = c_{idx_j}                    # lookup centroide
  x̃ = Πᵀ @ ỹ                        # rotação inversa
  return x̃
```

### Algorithm 2 — TurboQuant_prod (a implementar)

```
Quant_prod(x ∈ S^{d-1}), bit-width = b:
  idx = Quant_mse(x)                 # Algorithm 1 com (b-1) bits
  r = x - DeQuant_mse(idx)           # resíduo no espaço original
  qjl = sign(S @ r)                  # S ∈ R^{d×d} i.i.d. N(0,1); qjl ∈ {-1,+1}^d
  return (idx, qjl, ‖r‖₂)

DeQuant_prod(idx, qjl, γ):
  x̃_mse = DeQuant_mse(idx)
  x̃_qjl = sqrt(π/2)/d · γ · Sᵀ @ qjl
  return x̃_mse + x̃_qjl
```

> **Por que o Algorithm 2 é necessário:** O Algorithm 1 introduz viés multiplicativo
> de `2/π ≈ 0.637` nos produtos internos q·k no mecanismo de atenção (provado na
> Seção 3.2 do paper). Este viés se acumula por 28 camadas e causa o colapso de PPL
> observado nos resultados. O Algorithm 2 elimina o viés via QJL no resíduo.

### Tratamento de Outliers (paper §4.3)

```
# Paper: 32 outlier channels @ 3-bit + 96 normal channels @ 2-bit
#        → effective 2.5 bits = (32×3 + 96×2) / 128
#
# Config atual: outlier_bits=4, outlier_channels=32, bits=2
#        → effective 2.5 bits = (32×4 + 96×2) / 128  ← mesmos bits efetivos
#
# Detecção: magnitude média no ESPAÇO ORIGINAL (antes da normalização para S^{d-1})
# Dois TurboQuant_prod independentes com suas próprias rotações e codebooks

Quant_outlier(flat: [n_tokens, head_dim]):
  outlier_idx, normal_idx = detect_outliers(flat, outlier_channels)  # espaço original!
  
  outlier_norm, out_norms = normalize_sphere(flat[:, outlier_idx])
  q_out = TurboQuant_prod(outlier_norm, bits=outlier_bits, seed=rotation_seed+1)
  
  normal_norm, norm_norms = normalize_sphere(flat[:, normal_idx])
  q_norm = TurboQuant_prod(normal_norm, bits=bits, seed=rotation_seed)
  
  return q_out, q_norm, outlier_idx, normal_idx, out_norms, norm_norms
```

---

## Arquivos Afetados

| Arquivo | Ação | Motivo |
|---|---|---|
| `src/quantization/methods/turboquant.py` | modificar | implementar Algorithm 2 + outlier handling |
| `src/quantization/kv_cache.py` | modificar (pequeno) | corrigir kv_mb para incluir todos os tensores de meta |

---

## Sequência de Execução

### Passo 1 — Adicionar suporte a 1-bit em `_pack_indices` / `_unpack_indices`

**Arquivo:** `src/quantization/methods/turboquant.py`

**Onde:** funções `_pack_indices` e `_unpack_indices` (já existentes no arquivo)

**O que fazer:** adicionar `bits=1` ao `if bits not in (2, 4)` de ambas as funções.
Para `bits=1`: 8 valores por byte, usando shift de 1 bit por posição.

```python
# Em _pack_indices — adicionar após a linha "if bits not in (2, 4):"
# Modificar para:
if bits not in (1, 2, 4):
    return indices

# O loop interno já funciona genericamente:
# ipb = 8 // bits → bits=1 → ipb=8; bits=2 → ipb=4; bits=4 → ipb=2
# A lógica de packing não precisa de outra mudança além de incluir bits=1 no guard
```

```python
# Em _unpack_indices — mesma mudança:
if bits not in (1, 2, 4):
    return packed
# mask = (1 << bits) - 1 → bits=1 → mask=1 ✓
```

**Dependências:** nenhuma — é uma extensão da lógica existente.

---

### Passo 2 — Adicionar `_qjl_cache` e `_get_qjl_matrix`

**Arquivo:** `src/quantization/methods/turboquant.py`

**Onde:** bloco de caches de sessão (logo após `_codebook_cache`)

**O que adicionar:**

```python
# Em caches de sessão — adicionar após _codebook_cache:
_qjl_cache: dict[tuple[int, int], torch.Tensor] = {}

# Em clear_caches — adicionar:
_qjl_cache.clear()

# Nova função helper — adicionar após _get_rotation:
def _get_qjl_matrix(dim: int, seed: int, device: torch.device) -> torch.Tensor:
    """
    Retorna (e cacheia) S ∈ R^{d×d} com entradas i.i.d. N(0,1).

    Matriz de projeção aleatória para QJL (Definition 1 do paper).
    Seed separado da rotação Π para independência estatística.
    """
    key = (dim, seed)
    if key not in _qjl_cache:
        rng = np.random.default_rng(seed)
        g = rng.standard_normal((dim, dim)).astype(np.float32)
        _qjl_cache[key] = torch.from_numpy(g)
    return _qjl_cache[key].to(device)
```

**Dependências:** nenhuma.

---

### Passo 3 — Adicionar `_turboquant_prod_encode` (Algorithm 2, puro, sem outliers)

**Arquivo:** `src/quantization/methods/turboquant.py`

**Onde:** nova função, após `_scalar_dequantize` e antes de `_split_channels`

**O que adicionar:**

```python
def _turboquant_prod_encode(
    normalized: torch.Tensor,   # (n_tokens, dim) normalizado para S^{d-1}
    bits: int,
    rotation_seed: int,
    qjl_seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Implementa Algorithm 2 (TurboQuant_prod) do paper.

    Retorna (q_mse, q_qjl, r_norms, codebook) onde:
      - q_mse:    índices MSE empacotados a (bits-1) bits por coordenada
      - q_qjl:    bits de sinal QJL empacotados a 1 bit por coordenada
      - r_norms:  ‖r‖₂ por token, shape (n_tokens, 1), float32
      - codebook: centroides Lloyd-Max para (bits-1) bits
    """
    n_tokens, dim = normalized.shape
    mse_bits = max(1, bits - 1)  # bits-1 para TurboQuant_mse interno

    # ── Algorithm 1 (TurboQuant_mse com bits-1) ──────────────────────────────
    R = _get_rotation(dim, rotation_seed, device)
    rotated = normalized @ R                               # (n_tokens, dim)
    codebook = _get_theoretical_codebook(mse_bits, dim, device)
    indices = _scalar_quantize(rotated, codebook)          # (n_tokens, dim)
    y_tilde = _scalar_dequantize(indices, codebook)        # (n_tokens, dim) — espaço rotacionado
    x_tilde_mse = y_tilde @ R.T                           # (n_tokens, dim) — espaço original

    # ── Resíduo e QJL (Algorithm 2, linhas 6-7) ──────────────────────────────
    r = normalized - x_tilde_mse                          # (n_tokens, dim)
    r_norms = r.norm(dim=-1, keepdim=True).float()        # (n_tokens, 1)

    S = _get_qjl_matrix(dim, qjl_seed, device)
    qjl_logits = r @ S.T                                  # (n_tokens, dim): equivale S @ r para batch
    qjl_signs = (qjl_logits >= 0).to(torch.uint8)         # {0=neg, 1=pos}; packing como uint8

    # ── Empacotamento ────────────────────────────────────────────────────────
    q_mse = _pack_indices(indices, mse_bits)
    q_qjl = _pack_indices(qjl_signs, 1)                   # 8 signs / byte

    return q_mse, q_qjl, r_norms, codebook


def _turboquant_prod_decode(
    q_mse: torch.Tensor,
    q_qjl: torch.Tensor,
    r_norms: torch.Tensor,        # (n_tokens, 1)
    codebook: torch.Tensor,
    bits: int,
    rotation_seed: int,
    qjl_seed: int,
    dim: int,
    n_tokens: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Reconstrói vetores normalizados a partir do formato Algorithm 2.

    Implementa DeQuant_prod (Algorithm 2, linhas 9-12).
    Retorna tensor (n_tokens, dim) ainda em escala unitária (S^{d-1}).
    Desnormalização pelas norms originais é responsabilidade do chamador.
    """
    import math
    mse_bits = max(1, bits - 1)
    R = _get_rotation(dim, rotation_seed, device)
    S = _get_qjl_matrix(dim, qjl_seed, device)

    # ── Reconstrução MSE ─────────────────────────────────────────────────────
    indices = _unpack_indices(q_mse, mse_bits, dim)               # (n_tokens, dim)
    y_tilde = _scalar_dequantize(indices, codebook.to(device))    # espaço rotacionado
    x_tilde_mse = y_tilde @ R.T                                   # espaço original

    # ── Reconstrução QJL ─────────────────────────────────────────────────────
    qjl_bits = _unpack_indices(q_qjl, 1, dim)                     # (n_tokens, dim) uint8
    qjl_signs = qjl_bits.float() * 2.0 - 1.0                     # {0,1} → {-1.0, +1.0}
    # x̃_qjl = sqrt(π/2)/d · ‖r‖ · Sᵀ @ qjl
    # Em batch (row vectors): (qjl_signs @ S) equivale a Sᵀ @ qjl por token
    factor = math.sqrt(math.pi / 2.0) / dim
    x_tilde_qjl = factor * r_norms.to(device) * (qjl_signs @ S)  # (n_tokens, dim)

    return x_tilde_mse + x_tilde_qjl
```

**Dependências:** Passos 1 e 2.

---

### Passo 4 — Mover detecção de outliers para espaço original

**Arquivo:** `src/quantization/methods/turboquant.py`

**Onde:** função `_split_channels` (já existe, mas opera no espaço `rotated`)

**O que fazer:** alterar a assinatura para receber `flat` (espaço original) em vez de
`rotated`. A lógica interna permanece idêntica — apenas o tensor de entrada muda.

```python
# Alterar assinatura de:
def _split_channels(
    rotated: torch.Tensor,    # ← ERRADO: espaço rotacionado não tem outliers
    ...
# Para:
def _split_channels(
    flat: torch.Tensor,       # ← CORRETO: espaço original, antes da normalização
    ...
```

> **Justificativa:** A rotação ortogonal Π redistribui a energia uniformemente —
> detectar outliers no espaço rotacionado não tem sentido pois todos os canais
> terão magnitude similar. O paper detecta outliers nas embedding keys originais
> (before rotation), que exibem canais fixos de magnitude elevada (§2 do QJL paper).

**Dependências:** nenhuma além do Passo 3.

---

### Passo 5 — Reescrever `quantize_turboquant` para Algorithm 2 + outliers

**Arquivo:** `src/quantization/methods/turboquant.py`

**Onde:** função `quantize_turboquant` (substituição completa do corpo)

**O que fazer:**

```python
def quantize_turboquant(
    tensor: torch.Tensor,
    bits: int = 4,
    outlier_bits: int = 0,       # 0 → usa bits+1; agora ATIVO (não mais ignorado)
    outlier_channels: int = 32,  # agora ATIVO
    rotation_seed: int = 42,
    layer_idx: int = 0,          # mantido para compatibilidade
) -> tuple[torch.Tensor, dict]:
    """
    Quantiza KV com TurboQuant_prod (Algorithm 2, arXiv:2504.19874).

    Pipeline fiel ao paper:
      1. Reshape para (n_tokens, head_dim)
      2. [Se outlier_channels > 0] Detectar outliers no espaço ORIGINAL
         e aplicar dois TurboQuant_prod independentes por grupo de canais
      3. [Se outlier_channels == 0] Aplicar TurboQuant_prod em todo o vetor
      Em ambos os casos:
      4. TurboQuant_prod = (b-1)-bit MSE + 1-bit QJL no resíduo (Algorithm 2)
      5. Normalização para S^{d-1} (Lema 1) por grupo antes da quantização
    """
    original_shape = tensor.shape
    original_dtype = tensor.dtype
    device = tensor.device
    head_dim = tensor.shape[-1]

    flat = tensor.float().reshape(-1, head_dim)
    n_tokens = flat.shape[0]

    # resolve outlier_bits: 0 → bits+1 (1 bit acima do bits principal)
    eff_outlier_bits = outlier_bits if outlier_bits > 0 else (bits + 1)
    # seed da QJL: seed diferente da rotação para independência estatística
    qjl_seed = rotation_seed + 1000

    if outlier_channels > 0 and outlier_channels < head_dim:
        # ── Modo com outlier handling (paper §4.3) ────────────────────────
        # Detecção no espaço original (antes da normalização)
        outlier_idx, normal_idx, outlier_vals, normal_vals = _split_channels(
            flat, outlier_channels, head_dim, device
        )
        d_out = outlier_idx.shape[0]
        d_norm = normal_idx.shape[0]

        # Normaliza cada grupo para S^{d-1} (Lema 1)
        out_norm, out_norms = _normalize_to_sphere(outlier_vals)   # (n, d_out)
        norm_norm, norm_norms = _normalize_to_sphere(normal_vals)  # (n, d_norm)

        # TurboQuant_prod nos outlier channels (eff_outlier_bits)
        q_out_mse, q_out_qjl, r_out_norms, cb_out = _turboquant_prod_encode(
            out_norm, eff_outlier_bits, rotation_seed + 1, qjl_seed + 1, device
        )
        # TurboQuant_prod nos canais normais (bits)
        q_norm_mse, q_norm_qjl, r_norm_norms, cb_norm = _turboquant_prod_encode(
            norm_norm, bits, rotation_seed, qjl_seed, device
        )

        meta = {
            "mode": "outlier",
            # canais normais
            "q_norm_qjl": q_norm_qjl,
            "r_norm_norms": r_norm_norms,
            "codebook_norm": cb_norm,
            "norms_norm": norm_norms,
            "bits_norm": bits,
            # canais outlier
            "q_out_qjl": q_out_qjl,
            "r_out_norms": r_out_norms,
            "codebook_out": cb_out,
            "norms_out": out_norms,
            "bits_out": eff_outlier_bits,
            # índices de canal
            "outlier_idx": outlier_idx,
            "normal_idx": normal_idx,
            "d_out": d_out,
            "d_norm": d_norm,
            # shape e metadados gerais
            "original_shape": original_shape,
            "original_dtype": str(original_dtype),
            "head_dim": head_dim,
            "n_tokens": n_tokens,
            "rotation_seed": rotation_seed,
            "qjl_seed": qjl_seed,
        }
        # q_norm_mse é o tensor principal (maior componente) — usado para kv_mb
        return q_norm_mse, meta

    else:
        # ── Modo sem outlier handling — TurboQuant_prod no vetor completo ──
        normalized, norms = _normalize_to_sphere(flat)
        q_mse, q_qjl, r_norms, codebook = _turboquant_prod_encode(
            normalized, bits, rotation_seed, qjl_seed, device
        )
        meta = {
            "mode": "full",
            "q_qjl": q_qjl,
            "r_norms": r_norms,
            "codebook": codebook,
            "norms": norms,
            "original_shape": original_shape,
            "original_dtype": str(original_dtype),
            "head_dim": head_dim,
            "n_tokens": n_tokens,
            "bits": bits,
            "rotation_seed": rotation_seed,
            "qjl_seed": qjl_seed,
        }
        return q_mse, meta
```

**Dependências:** Passos 3 e 4.

---

### Passo 6 — Reescrever `dequantize_turboquant` para Algorithm 2 + outliers

**Arquivo:** `src/quantization/methods/turboquant.py`

**Onde:** função `dequantize_turboquant` (substituição completa do corpo)

**O que fazer:**

```python
def dequantize_turboquant(quantized: torch.Tensor, meta: dict) -> torch.Tensor:
    """
    Reconstrói tensor KV a partir do formato Algorithm 2.

    Inverte quantize_turboquant; trata ambos os modos ("full" e "outlier").
    """
    device = quantized.device
    dtype = getattr(torch, meta["original_dtype"].replace("torch.", ""))
    head_dim = meta["head_dim"]
    n_tokens = meta["n_tokens"]
    rotation_seed = meta["rotation_seed"]
    qjl_seed = meta["qjl_seed"]

    if meta.get("mode") == "outlier":
        # ── Reconstituição por grupo de canais ───────────────────────────
        # Canais normais
        norm_recon = _turboquant_prod_decode(
            quantized,               # q_norm_mse
            meta["q_norm_qjl"].to(device),
            meta["r_norm_norms"].to(device),
            meta["codebook_norm"].to(device),
            meta["bits_norm"],
            rotation_seed,
            qjl_seed,
            meta["d_norm"],
            n_tokens,
            device,
        )
        norm_recon = _denormalize(norm_recon, meta["norms_norm"].to(device))

        # Canais outlier
        out_recon = _turboquant_prod_decode(
            meta["q_out_mse"].to(device),   # NOTE: armazenado diretamente no meta
            meta["q_out_qjl"].to(device),
            meta["r_out_norms"].to(device),
            meta["codebook_out"].to(device),
            meta["bits_out"],
            rotation_seed + 1,
            qjl_seed + 1,
            meta["d_out"],
            n_tokens,
            device,
        )
        out_recon = _denormalize(out_recon, meta["norms_out"].to(device))

        # Monta tensor completo na ordem original dos canais
        result = torch.empty(n_tokens, head_dim, dtype=torch.float32, device=device)
        result[:, meta["normal_idx"].to(device)] = norm_recon
        result[:, meta["outlier_idx"].to(device)] = out_recon
        return result.reshape(meta["original_shape"]).to(dtype)

    else:
        # ── Modo sem outlier handling ─────────────────────────────────────
        bits = meta["bits"]
        normalized_recon = _turboquant_prod_decode(
            quantized,
            meta["q_qjl"].to(device),
            meta["r_norms"].to(device),
            meta["codebook"].to(device),
            bits,
            rotation_seed,
            qjl_seed,
            head_dim,
            n_tokens,
            device,
        )
        reconstructed = _denormalize(normalized_recon, meta["norms"].to(device))
        return reconstructed.reshape(meta["original_shape"]).to(dtype)
```

> **Atenção (detalhe de implementação):** No modo `"outlier"`, o tensor `q_out_mse`
> dos canais outlier precisa ser armazenado no `meta` pelo `quantize_turboquant`
> (não é retornado como tensor principal). O Passo 5 deve incluir
> `"q_out_mse": q_out_mse` no dicionário `meta` do modo outlier.

**Dependências:** Passos 3, 4, 5.

---

### Passo 7 — Corrigir `_handle_prefill` em `kv_cache.py`

**Arquivo:** `src/quantization/kv_cache.py`

**Onde:** método `_handle_prefill` da classe `QuantizedDynamicCache`

**O que fazer:** substituir o cálculo de `extra_mb` por uma função auxiliar que
contabiliza todos os tensores relevantes armazenados no `meta` do TurboQuant.

```python
def _meta_extra_mb(meta: dict) -> float:
    """
    Calcula MB de todos os tensores extras em meta (além do tensor quantizado
    principal). Inclui norms, r_norms, q_qjl e equivalentes para outliers.
    Agnóstico ao método — soma tudo o que for torch.Tensor no meta, exceto
    chaves que começam com 'codebook' (cacheados globalmente, não por run).
    """
    total = 0.0
    if not isinstance(meta, dict):
        return total
    _SKIP = {"codebook", "codebook_norm", "codebook_out"}
    for k, v in meta.items():
        if k in _SKIP:
            continue
        if isinstance(v, torch.Tensor):
            total += _mb(v)
    return total
```

Depois, em `_handle_prefill`:

```python
# Substituir o bloco atual de extra_mb por:
extra_mb = _meta_extra_mb(mk) + _meta_extra_mb(mv)
self.tracker.append(_mb(qk) + _mb(qv) + extra_mb)
```

> **Por que não contar codebooks:** o `_codebook_cache` é global e compartilhado
> entre todos os prompts e layers do mesmo run — seria double-counting severo.

**Dependências:** Passo 5.

---

## Detalhe: Armazenamento de `q_out_mse` no meta (correção do Passo 5)

No modo `"outlier"` do `quantize_turboquant`, o tensor retornado é `q_norm_mse`
(canais normais). O `q_out_mse` (canais outlier) deve ser salvo no `meta`:

```python
# No dicionário meta do modo "outlier", adicionar:
"q_out_mse": q_out_mse,
```

Isso é necessário para que `dequantize_turboquant` consiga reconstruir os
canais outlier sem precisar do tensor retornado.

---

## Conferência de Consistência — Bits Efetivos vs Paper

| Config atual (configs/kv_quant.yaml) | Bits efetivos | Referência no paper |
|---|---|---|
| `bits=2, outlier_bits=4, outlier_channels=32` | (32×4 + 96×2)/128 = **2.5 bits** | paper §4.3: "2.5-bit setup" ✅ |
| `bits=4, outlier_bits=5, outlier_channels=32` | (32×5 + 96×4)/128 = **4.25 bits** | paper §4.3: "3.5-bit" — não idêntico |

O config de bits=2 já corresponde ao setup do paper. O config de bits=4 fica em
4.25 bits efetivos (ligeiramente acima do 3.5-bit do paper). Isso é aceitável para
o objetivo do benchmark (comparação de algoritmos, não reprodução exata de números
do paper). O valor é documentado nos resultados para transparência.

Para reproduzir exatamente o 3.5-bit do paper seria necessário um config adicional
com `bits=3, outlier_bits=4, outlier_channels=64` ou similar — fora do escopo atual.

---

## Riscos e Mitigações

| Risco | Probabilidade | Mitigação |
|---|---|---|
| 3-bit packing ausente: `bits=4` usa (b-1)=3-bit MSE mas `_pack_indices` suporta apenas 1, 2, 4-bit | alta | Usar 4-bit storage para índices 3-bit (codebook tem 8 centroides que cabem em 4 bits); kv_mb ligeiramente sobestimado → documentado em comentário no código |
| `kv_mb` do modo outlier contabiliza `q_out_mse` no meta: dois tensores somados | baixa | `_meta_extra_mb` itera sobre todo o meta; `q_out_mse` é um tensor e será contabilizado automaticamente |
| Memória GPU na dequantização: `_turboquant_prod_decode` materializa `r_norms * (qjl_signs @ S)` em float32 full-dim | média | operação já feita no baseline (attention score computation); overhead existente no pipeline original. Aceitar como limitação do benchmark Python puro (sem kernel CUDA) |
| Seed do QJL igual ao da rotação: `_get_qjl_matrix(dim, rotation_seed+1000)` usa seed diferente; se `rotation_seed` for muito alto pode colidir | baixíssima | seed QJL = `rotation_seed + 1000`; para rotation_seed=42 → qjl_seed=1042, sem colisão prática |
| Compatibilidade do `QuantizedDynamicCache` com Qwen2.5 após mudança de formato do meta | baixa | a interface pública `quantize_fn / dequantize_fn` não muda; apenas o conteúdo interno do meta muda; o cache delega tudo às funções |

---

## Critérios de Conclusão

- [ ] `python3 -c "from src.quantization.methods.turboquant import quantize_turboquant, dequantize_turboquant; import torch; t = torch.randn(1,8,16,64); q,m = quantize_turboquant(t, bits=2, outlier_channels=32); r = dequantize_turboquant(q, m); print('OK', r.shape)"` executa sem erro
- [ ] `python3 -c "from src.quantization.methods.turboquant import quantize_turboquant, dequantize_turboquant; import torch; t = torch.randn(1,8,16,64); q,m = quantize_turboquant(t, bits=4, outlier_channels=0); r = dequantize_turboquant(q, m); print('OK', r.shape)"` executa sem erro
- [ ] `python3 -c "from src.quantization.methods.turboquant import _pack_indices, _unpack_indices; import torch; idx = torch.zeros(4,8,dtype=torch.int8); p = _pack_indices(idx,1); u = _unpack_indices(p,1,8); print('1-bit OK', p.shape, u.shape)"` executa sem erro
- [ ] Erro de reconstrução MSE-only vs prod-only diminui ao rodar um mini-teste de produto interno (x · q_recon ≈ x · y para vetores aleatórios)
- [ ] `kv_mb` do TurboQuant após re-run inclui overhead de norms + r_norms + qjl bits (verificável comparando os novos JSONs com os antigos)
- [ ] PPL do TurboQuant 4-bit se aproxima do baseline (< 2× baseline PPL) após o fix — confirmado em re-run no servidor
- [ ] Nenhum teste de uniform, kivi ou weight_quant é alterado

---

## Fora do Escopo (explícito)

- Implementar kernel CUDA para a multiplicação `qjl_signs @ S` (overhead de 5–15% vs baseline é aceitável para benchmark)
- Implementar entropy encoding dos índices (§3.1 do paper, ganho de 5% na memória)
- Adicionar modo TurboQuant 3.5-bit exato do paper (requer config extra, possível extensão futura)
- Modificar qualquer arquivo fora de `turboquant.py` e `kv_cache.py`
- Re-rodar o benchmark (tarefa separada, executada após a correção)
