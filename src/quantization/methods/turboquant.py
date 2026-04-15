"""
src/quantization/methods/turboquant.py
----------------------------------------
TurboQuant (MSE-optimal + produto-interno-unbiased) conforme:
  "TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate"
  (Zandieh, Daliri, Hadian, Mirrokni — arXiv:2504.19874, 2025)

Pipeline (quantize_turboquant):
  Modo "full" (outlier_channels == 0):
    1. Reshape KV → (n_tokens, head_dim)
    2. Normaliza cada vetor para S^{d-1}  (Lema 1: ||x||=1)
    3. TurboQuant_prod (Algorithm 2):
         a. TurboQuant_mse com (bits-1) bits → q_mse  (Algorithm 1)
         b. Resíduo r = x - DeQuant_mse(q_mse)
         c. QJL = sign(S @ r), ‖r‖  → q_qjl, r_norms
    4. Armazena (q_mse, q_qjl, r_norms, codebook, norms)

  Modo "outlier" (outlier_channels > 0)  — paper §4.3:
    1–2. Idênticos ao modo full
    3. Detecta outlier channels no ESPAÇO ORIGINAL (antes da normalização)
    4. Aplica TurboQuant_prod independente por grupo de canais:
         - outliers: eff_outlier_bits = outlier_bits (ou bits+1 se 0)
         - normais:  bits
    5. Armazena os dois grupos + índices de canal

Codebook: Lloyd-Max sobre N(0, 1/√d) — distribuição teórica das coordenadas
pós-rotação. Calculado UMA VEZ por (bits, head_dim, "v4") e cacheado.
"""

from __future__ import annotations

import math

import numpy as np
import torch

# ── caches de sessão ──────────────────────────────────────────────────────────
# _rotation_cache:  (dim, seed)          → Π ortogonal
# _codebook_cache:  (bits, head_dim, v)  → centroids Lloyd-Max N(0, 1/√d)
# _qjl_cache:       (dim, seed)          → S ∈ R^{d×d} i.i.d. N(0,1)
_rotation_cache: dict[tuple[int, int], torch.Tensor] = {}
_codebook_cache: dict[tuple[int, int, str], torch.Tensor] = {}
_qjl_cache: dict[tuple[int, int], torch.Tensor] = {}


def clear_caches() -> None:
    """Limpa todos os caches de sessão. Útil em sessões longas ou multirun."""
    _rotation_cache.clear()
    _codebook_cache.clear()
    _qjl_cache.clear()


# ── helpers de rotação e projeção ─────────────────────────────────────────────

def _get_rotation(dim: int, seed: int, device: torch.device) -> torch.Tensor:
    """Retorna (e cacheia) Π ∈ R^{d×d} ortogonal via QR de matriz Gaussiana."""
    key = (dim, seed)
    if key not in _rotation_cache:
        rng = np.random.default_rng(seed)
        g = rng.standard_normal((dim, dim)).astype(np.float32)
        q, _ = np.linalg.qr(g)
        _rotation_cache[key] = torch.from_numpy(q)
    return _rotation_cache[key].to(device)


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


# ── normalização para esfera unitária ────────────────────────────────────────

def _normalize_to_sphere(
    flat: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Normaliza cada linha para ||x||₂ = 1.

    Obrigatório: Lema 1 e Teorema 1 assumem x ∈ S^{d-1}.
    Retorna (normalizado, normas) para posterior desnormalização.
    """
    norms = flat.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    return flat / norms, norms


def _denormalize(flat: torch.Tensor, norms: torch.Tensor) -> torch.Tensor:
    """Restaura magnitude original multiplicando pelas normas salvas."""
    return flat * norms


# ── codebook teórico data-independent ───────────────────────────────────────

def _lloyd_max_1d(samples: torch.Tensor, n_levels: int) -> torch.Tensor:
    """Lloyd-Max 1D: iterações EM sobre amostras até convergência (máx 50)."""
    percentiles = torch.linspace(0, 100, n_levels + 2)[1:-1]
    sorted_vals = samples.sort().values
    idx = (percentiles / 100 * (len(sorted_vals) - 1)).long().clamp(0, len(sorted_vals) - 1)
    centroids = sorted_vals[idx].clone()
    for _ in range(50):
        dists = (samples.unsqueeze(1) - centroids.unsqueeze(0)).abs()
        assigns = dists.argmin(dim=1)
        new_centroids = torch.stack([
            samples[assigns == k].mean() if (assigns == k).any() else centroids[k]
            for k in range(n_levels)
        ])
        if (new_centroids - centroids).abs().max() < 1e-7:
            break
        centroids = new_centroids
    return centroids


def _get_theoretical_codebook(
    bits: int,
    head_dim: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Codebook Lloyd-Max para distribuição teórica N(0, 1/√d).

    Calculado UMA VEZ por (bits, head_dim), seed=0 fixo.
    Após normalização para S^{d-1} e rotação ortogonal Π, cada coordenada
    segue N(0, 1/d) exatamente — independente do RoPE. Sem truncamento.
    """
    key = (bits, head_dim, "v4")  # v4: codebook teórico N(0, 1/√d), sem rot_scale
    if key in _codebook_cache:
        return _codebook_cache[key].to(device)
    rng = np.random.default_rng(0)
    # std = 1/√d; Lloyd-Max sem clip cobre as caudas reais da distribuição
    raw = rng.normal(0.0, 1.0 / (head_dim ** 0.5), size=200_000).astype(np.float32)
    samples = torch.from_numpy(raw)
    centroids = _lloyd_max_1d(samples, 2 ** bits)
    _codebook_cache[key] = centroids
    return centroids.to(device)


# ── bit packing ──────────────────────────────────────────────────────────────

def _pack_indices(indices: torch.Tensor, bits: int) -> torch.Tensor:
    """
    Empacota índices int8/uint8 em uint8:
      bits=1 → 8 valores/byte
      bits=2 → 4 valores/byte
      bits=4 → 2 valores/byte
      outros → sem pack (retorna como está)

    Nota: bits=3 (mse_bits para bits=4) não tem packing nativo — retorna int8
    com 8 bits/valor. kv_mb ligeiramente superestimado neste caso (documentado).
    """
    if bits not in (1, 2, 4):
        return indices
    ipb = 8 // bits
    n, d = indices.shape
    pad = (-d) % ipb
    u = torch.nn.functional.pad(indices.int(), (0, pad)).to(torch.uint8)
    packed = torch.zeros(n, (d + pad) // ipb, dtype=torch.uint8, device=indices.device)
    for i in range(ipb):
        packed |= u[:, i::ipb] << ((ipb - 1 - i) * bits)
    return packed


def _unpack_indices(packed: torch.Tensor, bits: int, n_cols: int) -> torch.Tensor:
    """
    Inverte _pack_indices.
      bits=1 → 8 valores/byte
      bits=2 → 4 valores/byte
      bits=4 → 2 valores/byte
      outros → retorna como está (sem unpack)
    n_cols: dimensão original antes do padding.
    """
    if bits not in (1, 2, 4):
        return packed
    ipb, mask = 8 // bits, (1 << bits) - 1
    n = packed.shape[0]
    out = torch.zeros(n, packed.shape[1] * ipb, dtype=torch.int8, device=packed.device)
    for i in range(ipb):
        out[:, i::ipb] = ((packed >> ((ipb - 1 - i) * bits)) & mask).to(torch.int8)
    return out[:, :n_cols]


# ── quantização escalar por coordenada ───────────────────────────────────────

def _scalar_quantize(
    values: torch.Tensor,
    codebook: torch.Tensor,
) -> torch.Tensor:
    """
    Quantização escalar: centroide mais próximo por coordenada.

    Implementa Algoritmo 1 linha 6: idx_j = argmin_k |y_j − c_k|
    values:   (n_tokens, dim)
    codebook: (n_levels,)
    retorna:  (n_tokens, dim) int8 se n_levels <= 128, senão uint8/int16
    """
    dists = (values.unsqueeze(-1) - codebook).abs()
    dtype = (torch.int8  if len(codebook) <= 128 else
             torch.uint8 if len(codebook) <= 256 else torch.int16)
    return dists.argmin(dim=-1).to(dtype)


def _scalar_dequantize(
    indices: torch.Tensor,
    codebook: torch.Tensor,
) -> torch.Tensor:
    """Reconstrói valores float a partir dos índices e codebook."""
    return codebook[indices.long()]


# ── Algorithm 2: TurboQuant_prod (encode + decode) ───────────────────────────

def _turboquant_prod_encode(
    normalized: torch.Tensor,   # (n_tokens, dim) normalizado para S^{d-1}
    bits: int,
    rotation_seed: int,
    qjl_seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Implementa Algorithm 2 (TurboQuant_prod) do paper arXiv:2504.19874.

    Retorna (q_mse, q_qjl, r_norms, codebook) onde:
      q_mse:    índices MSE empacotados a (bits-1) bits por coordenada
      q_qjl:    bits de sinal QJL empacotados a 1 bit por coordenada
      r_norms:  ‖r‖₂ por token, shape (n_tokens, 1), float32
      codebook: centroides Lloyd-Max para (bits-1) bits

    O Algorithm 1 interno usa (bits-1) bits para deixar 1 bit para o QJL,
    eliminando o viés multiplicativo 2/π no produto interno (Seção 3.2).
    """
    n_tokens, dim = normalized.shape
    mse_bits = max(1, bits - 1)

    # ── Algorithm 1 (TurboQuant_mse com mse_bits) ────────────────────────────
    R = _get_rotation(dim, rotation_seed, device)
    rotated = normalized @ R
    codebook = _get_theoretical_codebook(mse_bits, dim, device)
    indices = _scalar_quantize(rotated, codebook)
    y_tilde = _scalar_dequantize(indices, codebook)        # espaço rotacionado
    x_tilde_mse = y_tilde @ R.T                           # espaço original

    # ── Resíduo e QJL (Algorithm 2, linhas 6-7) ──────────────────────────────
    r = normalized - x_tilde_mse
    r_norms = r.norm(dim=-1, keepdim=True).float()         # (n_tokens, 1)

    S = _get_qjl_matrix(dim, qjl_seed, device)
    # S @ r por token: em batch de row-vectors → r @ S.T
    qjl_logits = r @ S.T                                   # (n_tokens, dim)
    qjl_signs = (qjl_logits >= 0).to(torch.uint8)         # {0=neg, 1=pos}

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

    Implementa DeQuant_prod (Algorithm 2, linhas 9-12):
      x̃ = x̃_mse + sqrt(π/2)/d · ‖r‖ · Sᵀ @ qjl

    Retorna (n_tokens, dim) em escala S^{d-1} — desnormalização é responsabilidade
    do chamador.
    """
    mse_bits = max(1, bits - 1)
    R = _get_rotation(dim, rotation_seed, device)
    S = _get_qjl_matrix(dim, qjl_seed, device)

    # ── Reconstrução MSE ─────────────────────────────────────────────────────
    indices = _unpack_indices(q_mse, mse_bits, dim)
    y_tilde = _scalar_dequantize(indices, codebook.to(device))
    x_tilde_mse = y_tilde @ R.T

    # ── Reconstrução QJL ─────────────────────────────────────────────────────
    qjl_bits = _unpack_indices(q_qjl, 1, dim)             # int8 com valores 0 ou 1
    qjl_signs = qjl_bits.float() * 2.0 - 1.0             # {0,1} → {-1.0, +1.0}
    # Sᵀ @ qjl em batch de row-vectors: qjl_signs @ S
    factor = math.sqrt(math.pi / 2.0) / dim
    x_tilde_qjl = factor * r_norms.to(device) * (qjl_signs @ S)

    return x_tilde_mse + x_tilde_qjl


# ── detecção de canais outlier no espaço original ────────────────────────────

def _split_channels(
    flat: torch.Tensor,          # espaço original (antes da normalização para S^{d-1})
    outlier_channels: int,
    head_dim: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Separa os `outlier_channels` de maior magnitude média no espaço original.

    Detectar no espaço original (não no rotacionado) é correto porque a rotação
    ortogonal redistribui energia uniformemente — canais outlier só existem no
    espaço original (§2 do paper QJL). Retorna (outlier_idx, normal_idx,
    outlier_vals, normal_vals).
    """
    n_outliers = min(outlier_channels, head_dim)
    channel_score = flat.abs().mean(dim=0)
    _, outlier_idx = channel_score.topk(n_outliers)
    outlier_idx, _ = outlier_idx.sort()
    outlier_set = set(outlier_idx.tolist())
    normal_idx = torch.tensor(
        [i for i in range(head_dim) if i not in outlier_set],
        dtype=torch.long,
        device=device,
    )
    return outlier_idx, normal_idx, flat[:, outlier_idx], flat[:, normal_idx]


# ── API pública ───────────────────────────────────────────────────────────────

def quantize_turboquant(
    tensor: torch.Tensor,
    bits: int = 4,
    outlier_bits: int = 0,       # 0 → usa bits+1; agora ATIVO
    outlier_channels: int = 32,  # agora ATIVO (paper §4.3)
    rotation_seed: int = 42,
    layer_idx: int = 0,          # mantido para compatibilidade
) -> tuple[torch.Tensor, dict]:
    """
    Quantiza KV com TurboQuant_prod (Algorithm 2, arXiv:2504.19874).

    Pipeline:
      Modo "full"    (outlier_channels == 0): TurboQuant_prod no vetor completo
      Modo "outlier" (outlier_channels  > 0): dois TurboQuant_prod independentes
        - outlier channels a eff_outlier_bits (bits+1 se outlier_bits==0)
        - canais normais a bits
        Detecção de outliers no espaço ORIGINAL (antes da normalização).

    Em ambos os modos: TurboQuant_prod = (b-1)-bit MSE + 1-bit QJL no resíduo.
    O QJL elimina o viés multiplicativo 2/π do Algorithm 1 (Seção 3.2).
    """
    original_shape = tensor.shape
    original_dtype = tensor.dtype
    device = tensor.device
    head_dim = tensor.shape[-1]

    flat = tensor.float().reshape(-1, head_dim)
    n_tokens = flat.shape[0]

    # outlier_bits=0 → bits+1 (1 bit acima do principal, conforme paper §4.3)
    eff_outlier_bits = outlier_bits if outlier_bits > 0 else (bits + 1)
    # seed QJL separado da rotação para independência estatística
    qjl_seed = rotation_seed + 1000

    if outlier_channels > 0 and outlier_channels < head_dim:
        # ── Modo outlier (paper §4.3) ─────────────────────────────────────
        outlier_idx, normal_idx, outlier_vals, normal_vals = _split_channels(
            flat, outlier_channels, head_dim, device
        )
        d_out = outlier_idx.shape[0]
        d_norm = normal_idx.shape[0]

        # Normaliza cada grupo para S^{d-1} (Lema 1)
        out_norm, out_norms = _normalize_to_sphere(outlier_vals)
        norm_norm, norm_norms = _normalize_to_sphere(normal_vals)

        # TurboQuant_prod: outlier channels a eff_outlier_bits
        q_out_mse, q_out_qjl, r_out_norms, cb_out = _turboquant_prod_encode(
            out_norm, eff_outlier_bits, rotation_seed + 1, qjl_seed + 1, device
        )
        # TurboQuant_prod: canais normais a bits
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
            # canais outlier (q_out_mse armazenado no meta; não é retornado)
            "q_out_mse": q_out_mse,
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
            # metadados gerais
            "original_shape": original_shape,
            "original_dtype": str(original_dtype),
            "head_dim": head_dim,
            "n_tokens": n_tokens,
            "rotation_seed": rotation_seed,
            "qjl_seed": qjl_seed,
        }
        # q_norm_mse é o tensor principal (maior componente em número de bytes)
        return q_norm_mse, meta

    else:
        # ── Modo full: TurboQuant_prod no vetor completo ──────────────────
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


def dequantize_turboquant(quantized: torch.Tensor, meta: dict) -> torch.Tensor:
    """
    Reconstrói tensor KV a partir do formato Algorithm 2.

    Trata ambos os modos ("full" e "outlier") de forma transparente.
    Inverte quantize_turboquant: DeQuant_prod → desnormalização → reshape.
    """
    device = quantized.device
    dtype = getattr(torch, meta["original_dtype"].replace("torch.", ""))
    head_dim = meta["head_dim"]
    n_tokens = meta["n_tokens"]
    rotation_seed = meta["rotation_seed"]
    qjl_seed = meta["qjl_seed"]

    if meta.get("mode") == "outlier":
        # ── Reconstrói canais normais ─────────────────────────────────────
        norm_recon = _turboquant_prod_decode(
            quantized,
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

        # ── Reconstrói canais outlier ─────────────────────────────────────
        out_recon = _turboquant_prod_decode(
            meta["q_out_mse"].to(device),
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

        # ── Monta tensor completo na ordem original dos canais ────────────
        result = torch.empty(n_tokens, head_dim, dtype=torch.float32, device=device)
        result[:, meta["normal_idx"].to(device)] = norm_recon
        result[:, meta["outlier_idx"].to(device)] = out_recon
        return result.reshape(meta["original_shape"]).to(dtype)

    else:
        # ── Modo full ─────────────────────────────────────────────────────
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
