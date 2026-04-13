# Plano: Fix NVIDIA-only — device_map bitsandbytes + remover fallbacks CPU/MPS

**Data:** 2026-04-13
**Autor:** agente-plan
**Status:** aprovado

---

## Objetivo

Corrigir o `ValueError: '.to' is not supported for '4-bit' or '8-bit' bitsandbytes models`
causado pelo uso de `device_map="auto"` junto com `BitsAndBytesConfig`. Ao mesmo tempo,
tornar o projeto exclusivamente NVIDIA/CUDA, removendo fallbacks para CPU e MPS.

## Escopo

**Dentro do escopo:**
- Fix do `device_map` quando bitsandbytes está ativo (`device_map={"": torch.cuda.current_device()}`)
- Guarda antecipada `_require_cuda()` que aborta com `RuntimeError` claro se CUDA não disponível
- Remoção do ramo MPS em `_resolve_auto_device`
- Remoção da conversão fp16 → fp32 para CPU em `_build_load_kwargs`
- Remoção do `model.to(device)` condicional para CPU em `load_model`
- Atualização de `device: auto` → `device: cuda` nos três YAMLs de config

**Fora do escopo:**
- Suporte a múltiplas GPUs com sharding / tensor parallelism
- Mudanças nos runners de eval, reporter ou hooks KV
- Migração de versões de bitsandbytes / accelerate / transformers

---

## Causa Raiz

Em `src/runner/loader.py`, `_build_load_kwargs` passa `device_map="auto"` junto com
`BitsAndBytesConfig`. Com essa combinação, `accelerate` invoca `dispatch_model`, que itera
os módulos chamando `.to(device)` em cada um. O bitsandbytes bloqueia essa operação em
modelos 4-bit/8-bit porque os pesos já estão em formato quantizado fixo.

**Fix:** usar `device_map={"": torch.cuda.current_device()}` — instrui o accelerate a
mapear todo o modelo para a GPU ativa de uma vez, sem iterar módulos individualmente.

---

## Arquivos Afetados

| Arquivo | Ação | Motivo |
|---|---|---|
| `src/runner/loader.py` | modificar | fix `device_map` + guarda CUDA + remover CPU/MPS |
| `configs/baseline.yaml` | modificar | `device: auto` → `device: cuda` |
| `configs/weight_quant.yaml` | modificar | `device: auto` → `device: cuda` |
| `configs/kv_quant.yaml` | modificar | `device: auto` → `device: cuda` |

---

## Sequência de Execução

### 1. Adicionar `_require_cuda()` em `loader.py`
**Arquivo:** `src/runner/loader.py`
**O que fazer:** Adicionar função antes de `_resolve_auto_device`:
```python
def _require_cuda() -> None:
    """Garante que CUDA está disponível; aborta com mensagem clara caso contrário."""
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA não encontrado. Este projeto requer uma GPU NVIDIA com drivers CUDA.\n"
            "Verifique: nvidia-smi e torch.cuda.is_available()."
        )
```
**Dependências:** nenhuma

### 2. Modificar `_resolve_auto_device()` em `loader.py`
**Arquivo:** `src/runner/loader.py`
**O que fazer:** Remover ramo MPS; chamar `_require_cuda()` antes de retornar `"cuda"`:
```python
def _resolve_auto_device(device: str) -> str:
    """Resolve 'auto' para cuda; exige CUDA disponível."""
    if device != "auto":
        _require_cuda()
        return device
    _require_cuda()
    return "cuda"
```
**Dependências:** passo 1

### 3. Modificar `_build_load_kwargs()` em `loader.py`
**Arquivo:** `src/runner/loader.py`
**O que fazer:**
- Quando `use_bnb=True`: trocar `device_map="auto"` por `device_map={"": torch.cuda.current_device()}`
- Quando `use_bnb=False`: usar `device_map={"": device}` (mapeia todo o modelo para o device explicitamente)
- Remover o ramo `elif device == "cpu"` com a conversão fp16 → fp32
- Remover o parâmetro `device` do retorno desnecessário (já está no dict)

```python
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
        kwargs["device_map"] = {"": torch.cuda.current_device()}
    else:
        kwargs["device_map"] = {"": device}
    return kwargs, torch_dtype
```
**Dependências:** passo 1

### 4. Modificar `load_model()` em `loader.py`
**Arquivo:** `src/runner/loader.py`
**O que fazer:**
- Remover `if not use_bnb and device == "cpu": model = model.to(device)` — device_map já
  cuida do placement em todos os casos
**Dependências:** passo 3

### 5. Atualizar configs YAML
**Arquivos:** `configs/baseline.yaml`, `configs/weight_quant.yaml`, `configs/kv_quant.yaml`
**O que fazer:** Trocar `device: auto` por `device: cuda` nos três arquivos
**Dependências:** nenhuma (paralelo ao passo 1)

---

## Riscos e Mitigações

| Risco | Probabilidade | Mitigação |
|---|---|---|
| `device_map={"": "cuda"}` vs `{"": 0}` — aceitação pelo accelerate | baixa | usar `torch.cuda.current_device()` que retorna int; respeita `CUDA_VISIBLE_DEVICES` |
| `_require_cuda()` chamado em `_resolve_auto_device` quando `device` já é `"cuda"` hardcoded | nenhuma | chamada explícita cobre ambos os casos (auto e explícito) |
| Outros runners (kv_quant, baseline, evals) quebrarem | baixa | `load_model` é o único ponto de carga; mudança propaga automaticamente |

---

## Critérios de Conclusão

- [ ] `make weight-quant BITS=4` executa sem `ValueError`
- [ ] `make weight-quant BITS=8` executa sem `ValueError`
- [ ] `make baseline` executa sem erro
- [ ] Em máquina sem CUDA, qualquer comando exibe `RuntimeError` com mensagem clara e encerra imediatamente
