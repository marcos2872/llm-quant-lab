"""
src/quantization/methods/turboquant.py
----------------------------------------
TurboQuant (MSE-optimal) conforme Algorithm 1 de:
  "TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate"
  (Zandieh, Daliri, Hadian, Mirrokni — arXiv:2504.19874, 2025)

Pipeline (quantize_turboquant):
  1. Reshape KV → (n_tokens, head_dim)
  2. Normaliza cada vetor para ||x|| = 1  (Lema 1 exige x ∈ S^{d-1})
  3. Rotaciona com Π ∈ R^{d×d} ortogonal (Π = QR de matriz Gaussiana)
  4. Detecta outlier_channels de maior magnitude no espaço rotacionado
  5. Quantização escalar por coordenada — Algoritmo 1 linha 6:
       idx_j = argmin_k |y_j − c_k|  para todo j ∈ [d]
     Canais normais quantizados a `bits` bits.
     Canais outlier quantizados a `outlier_bits` bits (>= bits).
  6. Dequantize: reconstrói coordenadas, aplica Πᵀ, desnormaliza.

Codebook: Lloyd-Max sobre N(0, 1/√d) — distribuição teórica das coordenadas
pós-rotação (aproximação de alta dimensão da Beta do Lema 1). Calculado UMA VEZ
por (bits, head_dim) e cacheado — independente dos dados (implementa Eq. 4).
"""

from __future__ import annotations

import numpy as np
import torch

# ── caches de sessão ──────────────────────────────────────────────────────────
# _rotation_cache:  (head_dim, seed)  → Π
# _codebook_cache:  (bits, head_dim, version)  → centroids  (data-independent,
#                   calibrado para N(0, 1/√d) — distribuição teórica pós-rotação)
_rotation_cache: dict[tuple[int, int], torch.Tensor] = {}
_codebook_cache: dict[tuple[int, int], torch.Tensor] = {}


def clear_caches() -> None:
    """Limpa os caches de rotação e codebook. Útil em sessões longas ou multirun."""
    _rotation_cache.clear()
    _codebook_cache.clear()


# ── helpers de rotação ────────────────────────────────────────────────────────

def _get_rotation(dim: int, seed: int, device: torch.device) -> torch.Tensor:
    """Retorna (e cacheia) Π ∈ R^{d×d} ortogonal via QR de matriz Gaussiana."""
    key = (dim, seed)
    if key not in _rotation_cache:
        rng = np.random.default_rng(seed)
        g = rng.standard_normal((dim, dim)).astype(np.float32)
        q, _ = np.linalg.qr(g)
        _rotation_cache[key] = torch.from_numpy(q)
    return _rotation_cache[key].to(device)


# ── normalização para esfera unitária (D1) ────────────────────────────────────

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


# ── codebook teórico data-independent (D2) ───────────────────────────────────

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
    head_dim: int,  # mantido na assinatura para compatibilidade de cache key
    device: torch.device,
) -> torch.Tensor:
    """
    Codebook Lloyd-Max para distribuição teórica N(0, 1/√d).

    Calculado UMA VEZ por (bits, head_dim), seed=0 fixo.
    Após normalização para S^{d-1} e rotação ortogonal Π, cada coordenada
    segue N(0, 1/d) exatamente — independente do RoPE. Codebook sem truncamento.
    """
    key = (bits, head_dim, "v4")  # v4: codebook teórico N(0, 1/√d), sem rot_scale
    if key in _codebook_cache:
        return _codebook_cache[key].to(device)
    rng = np.random.default_rng(0)
    # Distribuição teórica exata das coordenadas pós-rotação de vetores em S^{d-1}:
    # std = 1/√d. Lloyd-Max sem truncamento cobre toda a distribuição real.
    raw = rng.normal(0.0, 1.0 / (head_dim ** 0.5), size=200_000).astype(np.float32)
    samples = torch.from_numpy(raw)  # sem clip — centroides cobrem as caudas reais
    centroids = _lloyd_max_1d(samples, 2 ** bits)
    _codebook_cache[key] = centroids
    return centroids.to(device)


# ── bit packing (compressão real de storage) ────────────────────────────────

def _pack_indices(indices: torch.Tensor, bits: int) -> torch.Tensor:
    """Empacota índices int8 em uint8: bits=2 → 4/byte; bits=4 → 2/byte; outros → sem pack."""
    if bits not in (2, 4):
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
    """Inverte _pack_indices. n_cols: dimensão original antes do padding."""
    if bits not in (2, 4):
        return packed
    ipb, mask = 8 // bits, (1 << bits) - 1
    n = packed.shape[0]
    out = torch.zeros(n, packed.shape[1] * ipb, dtype=torch.int8, device=packed.device)
    for i in range(ipb):
        out[:, i::ipb] = ((packed >> ((ipb - 1 - i) * bits)) & mask).to(torch.int8)
    return out[:, :n_cols]


# ── quantização escalar por coordenada (D4) ──────────────────────────────────

def _scalar_quantize(
    values: torch.Tensor,
    codebook: torch.Tensor,
) -> torch.Tensor:
    """
    Quantização escalar: centroide mais próximo por coordenada.

    Implementa Algoritmo 1 linha 6: idx_j = argmin_k |y_j − c_k|
    values:   (n_tokens, dim)
    codebook: (n_levels,)
    retorna:  (n_tokens, dim) int8 se n_levels <= 128, senão int16
    """
    dists = (values.unsqueeze(-1) - codebook).abs()
    dtype = (torch.int8  if len(codebook) <= 128 else
             torch.uint8 if len(codebook) <= 256 else torch.int16)
    return dists.argmin(dim=-1).to(dtype)


def _scalar_dequantize(
    indices: torch.Tensor,
    codebook: torch.Tensor,
) -> torch.Tensor:
    """Reconstrói valores float a partir dos índices e codebook (aceita int8 e int16)."""
    return codebook[indices.long()]


# ── detecção de canais outlier ────────────────────────────────────────────────

def _split_channels(
    rotated: torch.Tensor,
    outlier_channels: int,
    head_dim: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Separa os `outlier_channels` de maior magnitude média dos demais.

    Operado no espaço rotacionado (distribuição mais uniforme → seleção precisa).
    Retorna (outlier_idx, normal_idx, outlier_vals, normal_vals).
    """
    n_outliers = min(outlier_channels, head_dim)
    channel_score = rotated.abs().mean(dim=0)
    _, outlier_idx = channel_score.topk(n_outliers)
    outlier_idx, _ = outlier_idx.sort()
    outlier_set = set(outlier_idx.tolist())
    normal_idx = torch.tensor(
        [i for i in range(head_dim) if i not in outlier_set],
        dtype=torch.long,
        device=device,
    )
    return outlier_idx, normal_idx, rotated[:, outlier_idx], rotated[:, normal_idx]


# ── API pública ───────────────────────────────────────────────────────────────


def quantize_turboquant(
    tensor: torch.Tensor,
    bits: int = 4,
    outlier_bits: int = 0,      # mantido para compatibilidade; ignorado
    outlier_channels: int = 32, # mantido para compatibilidade; ignorado
    rotation_seed: int = 42,
    layer_idx: int = 0,         # mantido para compatibilidade; não utilizado
) -> tuple[torch.Tensor, dict]:
    """
    Quantiza KV com TurboQuant (arXiv:2504.19874 Algoritmo 1).

    Pipeline fiel ao paper:
      1. Reshape para (n_tokens, head_dim)
      2. Normaliza cada vetor para S^{d-1} (Lema 1: ||x||=1)
      3. Rotaciona com Π ortogonal → após rotação cada coordenada segue
         N(0, 1/d) marginalmente; não há outliers neste espaço
      4. Quantização escalar com codebook único Lloyd-Max para N(0, 1/√d)
         (Algoritmo 1 linha 6; codebook data-independent, calculado uma vez)
      5. Armazena (q_all, codebook, norms) para reconstrução

    Nota: outlier_bits e outlier_channels são aceitos por compatibilidade de
    interface com _get_quant_fns mas não utilizados. A rotação ortogonal já
    redistribui a energia uniformemente, eliminando outliers no espaço rotacionado.
    """
    original_shape, original_dtype = tensor.shape, tensor.dtype
    device, head_dim = tensor.device, tensor.shape[-1]

    # 1. Normalização para esfera unitária S^{d-1} (Lema 1)
    flat = tensor.float().reshape(-1, head_dim)
    normalized, norms = _normalize_to_sphere(flat)
    n_tokens = flat.shape[0]

    # 2. Rotação Π ∈ R^{d×d} (Algoritmo 1, linha 5)
    R = _get_rotation(head_dim, rotation_seed, device)
    rotated = normalized @ R

    # 3. Codebook teórico N(0, 1/√d) + quantização escalar (sem escala adaptativa).
    # Após S^{d-1} + rotação ortogonal: coordenadas ~ N(0, 1/d) exatamente.
    # Codebook calibrado para esta distribuição, sem clipping, invariante ao RoPE.
    codebook = _get_theoretical_codebook(bits, head_dim, device)
    q_all = _pack_indices(_scalar_quantize(rotated, codebook), bits)

    meta = {
        "codebook": codebook,
        "norms": norms,
        "rotation_seed": rotation_seed,
        "head_dim": head_dim,
        "n_tokens": n_tokens,
        "original_shape": original_shape,
        "original_dtype": str(original_dtype),
        "bits": bits,
    }
    return q_all, meta


def dequantize_turboquant(quantized: torch.Tensor, meta: dict) -> torch.Tensor:
    """
    Reconstrói tensor KV a partir do formato comprimido.

    Reverso de quantize_turboquant (Algoritmo 1 DeQuant_mse):
      1. Desempacota índices → lookup no codebook → espaço rotacionado
      2. Aplica Πᵀ = Π⁻¹ (rotação inversa; ortogonal → Πᵀ = Π⁻¹)
      3. Desnormaliza pelas normas originais (desfaz normalização para S^{d-1})
    """
    device = quantized.device
    dtype = getattr(torch, meta["original_dtype"].replace("torch.", ""))
    head_dim = meta["head_dim"]
    n_tokens = meta["n_tokens"]
    bits = meta["bits"]

    # 1. Desempacota índices e reconstrói no espaço rotacionado
    indices = _unpack_indices(quantized, bits, head_dim)  # (n_tokens, head_dim)
    full_rotated = _scalar_dequantize(indices, meta["codebook"].to(device))
    # rot_scale removido — valores já estão na escala correta N(0, 1/√d)

    # 2. Rotação inversa Πᵀ
    R = _get_rotation(head_dim, meta["rotation_seed"], device)
    reconstructed = full_rotated @ R.T

    # 3. Desnormaliza pelas normas originais
    reconstructed = _denormalize(reconstructed, meta["norms"].to(device))

    return reconstructed.reshape(meta["original_shape"]).to(dtype)
