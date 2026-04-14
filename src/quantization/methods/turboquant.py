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
# _codebook_cache:  (bits, head_dim)  → centroids  (data-independent)
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
    head_dim: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Codebook Lloyd-Max para N(0, 1/√d) — distribuição teórica das coordenadas
    pós-rotação (Lema 1, Eq. 4). Calculado UMA VEZ por (bits, head_dim),
    seed=0 fixo, independente dos dados.
    """
    key = (bits, head_dim)
    if key in _codebook_cache:
        return _codebook_cache[key].to(device)
    rng = np.random.default_rng(0)
    samples = torch.from_numpy(
        rng.normal(0.0, 1.0 / (head_dim ** 0.5), size=200_000).astype(np.float32)
    )
    centroids = _lloyd_max_1d(samples, 2 ** bits)
    _codebook_cache[key] = centroids
    return centroids.to(device)


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
    dtype = torch.int8 if len(codebook) <= 128 else torch.int16
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

def _build_meta(
    q_outlier: torch.Tensor, cb_normal: torch.Tensor, cb_outlier: torch.Tensor,
    outlier_idx: torch.Tensor, normal_idx: torch.Tensor, norms: torch.Tensor,
    rotation_seed: int, head_dim: int, n_tokens: int,
    original_shape: tuple, original_dtype: torch.dtype,
) -> dict:
    """Constrói dicionário de metadados para dequantize_turboquant."""
    return {
        "q_outlier": q_outlier, "cb_normal": cb_normal, "cb_outlier": cb_outlier,
        "outlier_idx": outlier_idx, "normal_idx": normal_idx, "norms": norms,
        "rotation_seed": rotation_seed, "head_dim": head_dim, "n_tokens": n_tokens,
        "original_shape": original_shape, "original_dtype": str(original_dtype),
    }


def quantize_turboquant(
    tensor: torch.Tensor,
    bits: int = 4,
    outlier_bits: int = 0,
    outlier_channels: int = 32,
    rotation_seed: int = 42,
    layer_idx: int = 0,  # mantido para compatibilidade de interface; não utilizado
) -> tuple[torch.Tensor, dict]:
    """Quantiza KV com TurboQuantmse (arXiv:2504.19874 Algoritmo 1).
    outlier_bits=0 → bits+1 automático. Retorna (q_normal, meta)."""
    original_shape, original_dtype = tensor.shape, tensor.dtype
    device, head_dim = tensor.device, tensor.shape[-1]
    # bits*2 garante SNR dos canais outlier próximo ao FP16 (ver diagnóstico)
    eff_outlier_bits = outlier_bits if outlier_bits > 0 else bits * 2

    # 1. Normalização para esfera unitária S^{d-1} (Lema 1)
    flat = tensor.float().reshape(-1, head_dim)
    normalized, norms = _normalize_to_sphere(flat)

    # 2. Rotação Π ∈ R^{d×d} (Algoritmo 1, linha 5)
    R = _get_rotation(head_dim, rotation_seed, device)
    rotated = normalized @ R

    # 3. Detecta canais outlier no espaço rotacionado
    outlier_idx, normal_idx, outlier_vals, normal_vals = _split_channels(
        rotated, outlier_channels, head_dim, device
    )

    # 4. Dois codebooks independentes + quantização escalar (Seção 4.3)
    cb_normal  = _get_theoretical_codebook(bits, head_dim, device)
    cb_outlier = _get_theoretical_codebook(eff_outlier_bits, head_dim, device)
    q_normal  = _scalar_quantize(normal_vals,  cb_normal)
    q_outlier = _scalar_quantize(outlier_vals, cb_outlier)

    meta = _build_meta(
        q_outlier, cb_normal, cb_outlier, outlier_idx, normal_idx, norms,
        rotation_seed, head_dim, flat.shape[0], original_shape, original_dtype,
    )
    return q_normal, meta


def dequantize_turboquant(quantized: torch.Tensor, meta: dict) -> torch.Tensor:
    """
    Reconstrói tensor KV a partir do formato comprimido.

    Reverso de quantize_turboquant (Algoritmo 1 DeQuant_mse):
      1. Reconstrói canais normais e outlier a partir dos codebooks
      2. Monta tensor completo no espaço rotacionado
      3. Aplica Πᵀ = Π⁻¹ (rotação inversa)
      4. Desnormaliza pelas normas originais
    """
    device = quantized.device
    dtype = getattr(torch, meta["original_dtype"].replace("torch.", ""))
    head_dim = meta["head_dim"]
    n_tokens = meta["n_tokens"]

    # 1. Reconstrói as duas partições
    normal_recon  = _scalar_dequantize(quantized,          meta["cb_normal"].to(device))
    outlier_recon = _scalar_dequantize(meta["q_outlier"].to(device), meta["cb_outlier"].to(device))

    # 2. Monta tensor rotacionado completo
    full_rotated = torch.zeros(n_tokens, head_dim, device=device, dtype=torch.float32)
    full_rotated[:, meta["normal_idx"].to(device)]  = normal_recon
    full_rotated[:, meta["outlier_idx"].to(device)] = outlier_recon

    # 3. Rotação inversa Πᵀ (Πᵀ = Π⁻¹ para matriz ortogonal)
    R = _get_rotation(head_dim, meta["rotation_seed"], device)
    reconstructed = full_rotated @ R.T

    # 4. Desnormaliza pelas normas originais (desfaz normalização para S^{d-1})
    reconstructed = _denormalize(reconstructed, meta["norms"].to(device))

    return reconstructed.reshape(meta["original_shape"]).to(dtype)
