# LLM Quant Lab: 2,5× Menos VRAM em LLMs de 7B — Weight INT4 Ainda Roda 38% Mais Rápido

Rodar um modelo de 7B em produção custa caro. Um Qwen2.5-7B-Instruct em FP16 ocupa **15,3 GB de VRAM** — acima do limite de GPUs de consumo como RTX 3090 (24 GB, mas dividida com KV cache e ativações) e impossível em GPUs de 8 GB.

Construí um benchmark local reproduzível comparando **5 estratégias de quantização** — 2 para pesos, 3 para KV cache — usando PyTorch, bitsandbytes e hooks de atenção customizados. Todos os experimentos rodam sobre o mesmo conjunto de prompts de contexto longo (~4 000 tokens de entrada), no mesmo hardware, com métricas coletadas automaticamente.

> Testado em **HP Z4 G5 Workstation** · Intel Xeon w3-2535 (10 cores / 20 threads, até 4,6 GHz) · **NVIDIA RTX 4000 20GB VRAM Ada Generation** · 128 GB RAM · Ubuntu 24.04.4 LTS (Kernel 6.17) · CUDA 12.x (driver 580.126.09)

**[Repositório completo com benchmarks e código-fonte](https://github.com/marcos2872/llm-quant-lab)**

---

## Resultados Principais

**Weight INT4 (NF4 double quant)** é o campeão absoluto: **2,52× menos VRAM** (6,1 GB vs 15,3 GB) e **+38% de throughput no decode** (18,9 tok/s vs 13,7 tok/s). O modelo de 7B passa a caber em GPUs de 8 GB.

**KV Uniform / KIVI 2-bit** comprimem o KV cache em **7× sem overhead de decode** — de 764 MB para 108 MB, liberando headroom para contextos mais longos sem mover nada para CPU.

**TurboQuant 4-bit** comprime o KV cache em **3,5×** (217 MB) preservando canais outlier em FP16, trocando um pouco de compressão por maior robustez numérica.

---

## Comparação Detalhada

| Método | Pesos (MB) | KV Cache (MB) | Pico GPU (MB) | Compressão de Pico | Decode (tok/s) | Prefill (tok/s) |
|---|---|---|---|---|---|---|
| **Baseline FP16** | 14 537 | 764 | 15 302 | 1,0× | 13,7 | 26 018 |
| **Weight INT8** | 11 542 | 932 | 12 475 | 1,2× | 12,5 | 3 878 |
| **Weight INT4 NF4** | 5 308 | 763 | 6 070 | **2,5×** | **18,9 (+38%)** | 9 221 |
| **KV Uniform 4-bit** | 14 537 | 109 | 15 193 | 1,0× | 12,9 | 25 478 |
| **KV Uniform 2-bit** | 14 537 | 109 | 15 193 | 1,0× | 12,5 | 25 231 |
| **KV KIVI 4-bit** | 14 537 | 109 | 15 206 | 1,0× | 13,1 | 25 300 |
| **KV KIVI 2-bit** | 14 537 | 109 | 15 206 | 1,0× | 13,1 | 25 374 |
| **KV TurboQuant 4-bit** | 14 537 | 218 | 15 357 | 1,0× | 11,7 | 25 304 |
| **KV TurboQuant 2-bit** | 14 537 | 218 | 15 357 | 1,0× | 11,9 | 25 174 |

> Médias sobre 4 prompts de contexto longo (~4 000 tokens de entrada, 64 tokens gerados). Modelo: **Qwen/Qwen2.5-7B-Instruct**.

---

## Por que Weight INT4 Acelera o Decode?

O decode de LLMs é **memory-bandwidth bound** — a GPU passa mais tempo movendo os pesos da VRAM para os registradores do que computando. Ao reduzir o tamanho dos pesos de FP16 (2 bytes) para INT4 (0,5 byte), **cada token gerado requer 4× menos leitura de memória**.

Com Weight INT4 NF4, o prefill fica **mais lento** (26 018 → 9 221 tok/s) porque cada operação de multiplicação exige dequantização on-the-fly dos pesos. Mas o decode — que processa 1 token por vez — acelera 38%, exatamente onde o usuário sente a latência.

```
Baseline FP16  →  pesos: 14,5 GB  |  decode: 13,7 tok/s
Weight INT4    →  pesos:  5,3 GB  |  decode: 18,9 tok/s  (+38%)
```

Weight INT8 **não acelerou** nada nesta GPU: o overhead de dequantização cancelou o ganho de bandwidth — um resultado comum em GPUs Ada/Ampere com tensor cores otimizados para FP16.

---

## Por que KV Uniform e KIVI Empatam?

Ambos alcançam **exatamente 108,8 MB** de KV cache — compressão de 7× — porque os dois operam com 2-bit por valor e mesma granularidade por grupo. A diferença está em como calculam os extremos:

- **Uniform**: min-max global por tensor → rápido, sem estado
- **KIVI**: estatísticas por grupo de canais → mais adaptativo, mesma compressão prática aqui

Para distribuições de atenção do Qwen2.5, a covariância entre canais não foi suficientemente heterogênea para que o KIVI se destacasse. Em modelos com atenção multi-query (MQA) ou com outliers de ativação mais extremos (ex.: Llama-3), o KIVI tende a ser superior.

---

## Por que TurboQuant Comprime Menos?

O método TurboQuant desta implementação preserva **32 canais outlier em FP16** (configurável via `outlier_channels: 32`). Esses canais têm variância muito alta e degradariam a qualidade se quantizados agressivamente.

O resultado é **217 MB de KV** em vez de 109 MB — 3,5× em vez de 7× de compressão. A troca é intencional: menor compressão, maior fidelidade numérica em sequências longas onde outliers de atenção são críticos para o raciocínio multi-hop.

Pipeline matemático do TurboQuant:

```
1. R  ← matriz ortogonal Haar aleatória  (d × d, seed fixo)
2. y  =  K / V × Rᵀ                      (rotação → uniformiza variância)
3. outliers ← canais com ||y_i||₂ > limiar  (preservados em FP16)
4. ŷ  ← Lloyd-Max(y_normal, b bits)       (codebook ótimo por grupo)
5. armazenado: (ŷ, outliers_fp16, codebook)
6. dequantização: ỹ = expand(ŷ, outliers) × R  (rotação inversa)
```

---

## Estratégia Híbrida: Prefill Quantizado, Decode em FP16

Um detalhe de implementação importante: o KV cache **não re-quantiza a cada passo de decode**. A estratégia usada é:

```
┌─────────────────────────────────────┐
│  Prefill (N tokens, quantizado 1×)  │  →  armazenado comprimido em GPU
└─────────────────────────────────────┘
              ↓ decode ↓
┌─────────────────────────────────────┐
│  Buffer FP16 de tokens novos        │  →  cresce lentamente (< 256 tokens)
└─────────────────────────────────────┘
```

Na atenção, o contexto comprimido é dequantizado e concatenado com o buffer FP16 a cada passo. Isso elimina o custo de re-quantizar tokens gerados e preserva a qualidade de raciocínio incremental.

---

## Como Rodar no Seu Ambiente

```bash
# Instalar dependências (requer GPU NVIDIA + CUDA)
make setup

# Baseline FP16
make baseline MODEL=Qwen/Qwen2.5-7B-Instruct

# Weight quantization INT4
make weight-quant BITS=4

# KV cache quantization — todos os métodos
make kv-quant METHOD=uniform    BITS=2
make kv-quant METHOD=kivi       BITS=2
make kv-quant METHOD=turboquant BITS=4

# Pipeline completo + relatório CSV + gráficos
make all
```

Resultados em `results/raw/*.json`, relatório em `results/reports/summary.csv`.

---

## Hardware Utilizado

```
Sistema:   HP Z4 G5 Workstation Desktop PC
OS:        Ubuntu 24.04.4 LTS (Noble Numbat) · Kernel 6.17.0-19-generic
CPU:       Intel Xeon w3-2535 · 10 cores / 20 threads · até 4,6 GHz · 20 MB L2
GPU:       NVIDIA RTX 4000 Ada Generation · driver 580.126.09
RAM:       128 GiB DDR5
Armazena.: NVMe Samsung 2 TB (sistema) + 3× HDD HGST 2 TB
```

Todos os números foram coletados com o sistema em estado estável (7d 15h de uptime), sem outros processos GPU ativos.

---

## Referências

- **TurboQuant**: Xu et al. (2025) — [arXiv:2504.19874](https://arxiv.org/abs/2504.19874)
- **KIVI**: Liu et al. (2024) — [arXiv:2402.02750](https://arxiv.org/abs/2402.02750)
- **bitsandbytes**: Dettmers et al. — [github.com/bitsandbytes-foundation/bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes)
- **Qwen2.5**: Qwen Team, Alibaba Cloud (2024)

---

*Teste no seu modelo e compartilhe os resultados — especialmente se usar MQA ou modelos com outliers extremos.*
