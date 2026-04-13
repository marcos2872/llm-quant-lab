"""
src/runner/loader.py
--------------------
Carrega modelo e tokenizer a partir de um dicionário de configuração.

Suporta:
  - dtype FP16 / FP32 / BF16
  - quantização de pesos via bitsandbytes (INT8 / INT4)
  - device cuda (NVIDIA obrigatório; auto resolve para cuda)
"""

from __future__ import annotations

import logging
import warnings

import torch
from rich.console import Console
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

console = Console()
logger = logging.getLogger(__name__)

# suprime FutureWarning e UserWarning do transformers
# usa message= porque stacklevel alto faz o warning aparecer no código do chamador
warnings.filterwarnings("ignore", message=".*past_key_values.*", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*do_sample.*top_k.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*Cache.*", category=FutureWarning)

_DTYPE_MAP: dict[str, torch.dtype] = {
    "fp16": torch.float16,
    "fp32": torch.float32,
    "bf16": torch.bfloat16,
}


def _require_cuda() -> None:
    """Garante que CUDA está disponível; aborta com mensagem clara caso contrário."""
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA não encontrado. Este projeto requer uma GPU NVIDIA com drivers CUDA.\n"
            "Verifique: nvidia-smi e torch.cuda.is_available()."
        )


def _resolve_auto_device(device: str) -> str:
    """Resolve 'auto' para cuda; exige CUDA disponível."""
    _require_cuda()
    if device != "auto":
        return device
    return "cuda"


def _bnb_available() -> bool:
    """Verifica se bitsandbytes pode ser usado (requer CUDA)."""
    return torch.cuda.is_available()


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


def _build_load_kwargs(
    use_bnb: bool,
    wq: dict,
    torch_dtype: torch.dtype,
    device: str,
) -> tuple[dict, torch.dtype]:
    """Constrói kwargs para from_pretrained e dtype final efetivo."""
    kwargs: dict = {"torch_dtype": torch_dtype}
    if use_bnb:
        kwargs["quantization_config"] = _build_bnb_config(wq)
        kwargs["device_map"] = {"" : torch.cuda.current_device()}
    else:
        kwargs["device_map"] = {"": device}
    return kwargs, torch_dtype


def _log_model_info(model: AutoModelForCausalLM, name: str) -> None:
    """Loga número de parâmetros e estimativa de memória."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    console.print(
        f"[cyan]Modelo:[/cyan] {name}\n"
        f"  Parâmetros totais : {total / 1e6:.1f}M\n"
        f"  Treináveis        : {trainable / 1e6:.1f}M\n"
        f"  Memória FP16 est. : {total * 2 / 1024 ** 3:.2f} GB"
    )


def load_model(config: dict) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Carrega modelo e tokenizer conforme config dict.

    Parâmetros relevantes: model, dtype, device, trust_remote_code, weight_quantization.
    Retorna (model, tokenizer) com model em eval mode.
    """
    model_name: str = config["model"]
    dtype_str: str = config.get("dtype", "fp16")
    device = _resolve_auto_device(config.get("device", "auto"))
    trust: bool = config.get("trust_remote_code", False)
    torch_dtype = _DTYPE_MAP.get(dtype_str, torch.float16)

    wq = config.get("weight_quantization", {})
    use_bnb = wq.get("enabled", False)

    if use_bnb and not _bnb_available():
        bits = wq.get("bits", 4)
        console.print(
            f"[yellow]⚠ bitsandbytes INT{bits} requer CUDA — não disponível.[/yellow]\n"
            f"[yellow]  Carregando em {dtype_str} sem quantização de pesos.[/yellow]"
        )
        use_bnb = False

    console.print(
        f"[bold]Carregando modelo[/bold] {model_name}  |  "
        f"device={device}  |  dtype={dtype_str}  |  bnb={use_bnb}"
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs, torch_dtype = _build_load_kwargs(use_bnb, wq, torch_dtype, device)
    load_kwargs["trust_remote_code"] = trust

    model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)

    model.eval()
    _log_model_info(model, model_name)
    return model, tokenizer
