"""
src/runner/loader.py
--------------------
Carrega modelo e tokenizer a partir de um dicionário de configuração.

Suporta:
  - dtype FP16 / FP32 / BF16
  - quantização de pesos via bitsandbytes (INT8 / INT4)
  - device auto / cpu / cuda / mps
"""

from __future__ import annotations

import logging

import torch
from rich.console import Console
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

console = Console()
logger = logging.getLogger(__name__)

# mapeamento de string → torch.dtype
_DTYPE_MAP: dict[str, torch.dtype] = {
    "fp16": torch.float16,
    "fp32": torch.float32,
    "bf16": torch.bfloat16,
}


def _resolve_device(device: str) -> str:
    """Resolve 'auto' para o melhor device disponível."""
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _build_bnb_config(wq: dict) -> BitsAndBytesConfig:
    """Constrói BitsAndBytesConfig a partir do sub-dict weight_quantization."""
    bits = wq.get("bits", 4)
    if bits == 8:
        return BitsAndBytesConfig(load_in_8bit=True)
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=wq.get("double_quant", True),
        bnb_4bit_quant_type=wq.get("quant_type", "nf4"),
        bnb_4bit_compute_dtype=torch.float16,
    )


def _log_model_info(model: AutoModelForCausalLM, name: str) -> None:
    """Loga número de parâmetros e estimativa de memória."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    mem_fp16_gb = total * 2 / 1024 ** 3
    console.print(
        f"[cyan]Modelo:[/cyan] {name}\n"
        f"  Parâmetros totais : {total / 1e6:.1f}M\n"
        f"  Treináveis        : {trainable / 1e6:.1f}M\n"
        f"  Memória FP16 est. : {mem_fp16_gb:.2f} GB"
    )


def load_model(config: dict) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Carrega modelo e tokenizer conforme config dict.

    Parâmetros relevantes no config:
      model, dtype, device, trust_remote_code, weight_quantization

    Retorna (model, tokenizer) com model em eval mode.
    """
    model_name: str = config["model"]
    dtype_str: str = config.get("dtype", "fp16")
    device_str: str = config.get("device", "auto")
    trust: bool = config.get("trust_remote_code", False)

    device = _resolve_device(device_str)
    torch_dtype = _DTYPE_MAP.get(dtype_str, torch.float16)

    wq = config.get("weight_quantization", {})
    use_bnb = wq.get("enabled", False)

    console.print(f"[bold]Carregando modelo[/bold] {model_name}  |  device={device}  |  dtype={dtype_str}  |  bnb={use_bnb}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=trust,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs: dict = {
        "trust_remote_code": trust,
        "torch_dtype": torch_dtype,
    }

    if use_bnb:
        load_kwargs["quantization_config"] = _build_bnb_config(wq)
        load_kwargs["device_map"] = "auto"
    else:
        load_kwargs["device_map"] = device if device != "cpu" else None

    model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)

    if not use_bnb and device == "cpu":
        model = model.to(device)

    model.eval()
    _log_model_info(model, model_name)
    return model, tokenizer
