# DeepSeek-V3-Inspired Financial LLM From Scratch

A **101.7M-parameter DeepSeek-V3-inspired educational/research implementation**, built
from scratch in PyTorch and specialized toward financial language using selected,
license-verified financial datasets.

> This is **not** the full-scale DeepSeek-V3 model, and it does not use DeepSeek's
> pretrained weights. It is a small educational/research implementation of the same
> architectural ideas. All text generation comes from the PyTorch model in this
> repository — no OpenAI, Claude, Gemini, DeepSeek API, Ollama, Llama, Mistral, or
> other pretrained model is used for generation. Hugging Face is used **only as a
> dataset source**.

## Architecture

Implemented from scratch:

- **Multi-Head Latent Attention (MLA)** — compresses K/V into a shared latent space
  (`kv_lora_rank`), with a separate RoPE component concatenated onto the content heads
- **Mixture of Experts (MoE)** — 8 experts, top-2 routing, a shared always-on expert,
  and auxiliary-loss-free load balancing via a learned per-expert bias
- **Multi-Token Prediction (MTP)** — an extra head predicting the token after next
- **RoPE**, **RMSNorm**, **SwiGLU**

### Model configuration

There is exactly **one** authoritative configuration: [`configs/model_config.yaml`](configs/model_config.yaml),
loaded via `DeepSeekConfig.default()`. Training, validation, inference, evaluation, and
inspection all read it, and every checkpoint stores a full copy of the config it was
trained with.

| Parameter | Value |
|---|---|
| Total parameters | **101,723,264** (measured) |
| Vocabulary | 50,257 (GPT-2 / tiktoken) |
| Context length | 1,024 |
| Embedding dim | 512 |
| Layers | 8 |
| Attention heads | 8 |
| KV LoRA rank | 128 |
| Q LoRA rank | 192 |
| RoPE dim | 32 |
| Experts | 8 (top-2 per token) |
| Expert hidden / shared expert hidden | 512 / 768 |
| MTP heads | 1 |
| Weight tying | `wte.weight is lm_head.weight` |

<details>
<summary><b>Note on the previously documented 109,032,032 figure</b></summary>

An earlier version of this README stated 109,032,032 parameters. The model as committed
instantiates **101,723,264** — a difference of 7,308,768 (6.70%).

The README documents 9 architecture parameters, and the code matches all 9 exactly. The
only undocumented free knobs are `expert_intermediate_size`, `shared_expert_intermediate_size`,
and `rope_dim`. The 109M target is provably unreachable through them: achievable totals
form a lattice of spacing `gcd(110736, 13842) = 13842`, and `7,308,768 mod 13,842 = 192 ≠ 0`
(including `rope_dim` still leaves a residual of 12). The nearest reachable value is
109,031,840 — 192 short.

The older figure therefore came from a model instance whose configuration is not in this
repository. The architecture has **not** been altered to chase the number; the measured
count is reported instead.
</details>

## Quick start

```bash
pip install -r requirements.txt
```

```bash
python main.py dataset list
```

```bash
python main.py dataset prepare --name fineweb_edu --max-tokens 1000000
```

```bash
python main.py train --preset tiny_debug
```

```bash
python main.py evaluate --checkpoint checkpoints/base/checkpoint_100.pt
```

```bash
python main.py chat --checkpoint checkpoints/base/checkpoint_100.pt
```

## CLI

| Command | Purpose |
|---|---|
| `dataset list` | Registered datasets with HF ids, licenses, verification status |
| `dataset download --name X` | Probe a source without materializing it |
| `dataset prepare --name X --max-tokens N` | Stream → clean → dedup → split → tokenize → shard |
| `dataset stats [--name X]` | Real tokenizer statistics from prepared shards |
| `train --preset tiny_debug\|small\|financial_poc` | Stage A pretraining |
| `train --resume <ckpt.pt>` | Restore model/optimizer/step/best-val/token count |
| `train --stage instruction --checkpoint <ckpt.pt>` | Stage B instruction tuning |
| `evaluate --checkpoint <ckpt.pt>` | Financial LLM POC Evaluation |
| `chat --checkpoint <ckpt.pt> [--document F]` | Financial chat (router + calculator + RAG) |
| `inspect tokens\|architecture\|moe\|forward` | Real model internals |

## Training presets

Budgets are configurable; **`financial_poc` is never the default** — it must be selected
explicitly.

| Preset | Train tokens | Val tokens | Steps | Batch × seq |
|---|---|---|---|---|
| `tiny_debug` | 1M | 100K | 100 | 2 × 256 |
| `small` | 10M | 500K | 2,000 | 16 × 512 |
| `financial_poc` | 50M | 2M | 8,000 | 16 × 1024 |

`seq_len` may be shorter than the model's 1024 `block_size` (the architecture supports any
`t <= block_size`). `tiny_debug` uses 2 × 256 because batch 8 × seq 1024 exhausts memory
on CPU.

## Dataset sources

All entries are license-verified against the Hugging Face dataset card. Anything that
cannot be verified is marked `REQUIRES DATASET VERIFICATION` and excluded from default mixes.

| Name | Hugging Face ID | Category | License |
|---|---|---|---|
| `fineweb_edu` | `HuggingFaceFW/fineweb-edu` (CC-MAIN-2024-51) | general | ODC-BY |
| `tinystories` | `roneneldan/TinyStories` | general | CDLA-Sharing-1.0 |
| `financial_text_investopedia` | `FinLang/investopedia-instruction-tuning-dataset` | financial_text | CC-BY-NC-4.0 |
| `financial_qa_sujet` | `sujet-ai/Sujet-Finance-Instruct-177k` | financial_qa | Apache-2.0 |
| `financial_reports_sec` | `JanosAudran/financial-reports-sec` | financial_reports | Apache-2.0 |
| `financial_reasoning_finqa` | `ibm-research/finqa` | financial_reasoning | CC-BY-4.0 |
| `financial_instruction_alpaca` | `gbharti/finance-alpaca` | financial_instruction | MIT |
| `financial_sentiment_phrasebank` | `takala/financial_phrasebank` | news_sentiment | CC-BY-NC-SA-3.0 |

⚠️ Two sources are **non-commercial** (CC-BY-NC-4.0, CC-BY-NC-SA-3.0). Research/education use only.

Provenance for every prepare run — records seen/used/removed, tokens, revision, license,
date — is appended to [`data/dataset_manifest.json`](data/dataset_manifest.json) and embedded
in every checkpoint.

### Default mixture

| Bucket | Weight |
|---|---|
| FineWeb-Edu | 40% |
| Financial text | 20% |
| Financial QA | 15% |
| Financial reports | 10% |
| Financial reasoning | 10% |
| Financial instruction | 5% |

Weights are validated to sum to 1.0. Buckets without prepared shards are dropped and the
remainder renormalized.

## Data pipeline

```
HF stream (bounded by max_tokens/max_records)
  → clean (HTML, Unicode, boilerplate, whitespace)
  → filter (empty, malformed, min length)
  → dedup (exact SHA-256 + shingle near-dup)
  → deterministic hash split (seeded; duplicates land in the same split)
  → tokenize (tiktoken GPT-2)
  → shards (data/shards/<name>/{train,validation}/shard_NNN.bin + index.json)
```

Financial notation (`₹ $ € % EPS P/E EBITDA ROE ROIC FCF YoY QoQ FY2025 Q4 10.5%`) is
explicitly preserved by the cleaner.

**Leakage prevention:** evaluation sets live in `data/evaluation/` and are *never* written
into `data/shards/`, so they cannot enter the training mixture. `evaluate` runs an explicit
substring check of every eval question against every training shard before scoring.

## Financial chat architecture

```
USER → QUERY ROUTER → { LLM | CALCULATOR | RAG } → ANSWER + SOURCE
```

The router classifies into `GENERAL`, `FINANCIAL_KNOWLEDGE`, `NUMERICAL`, `DOCUMENT`,
`LIVE_DATA`, `UNKNOWN`.

- **Arithmetic is never left to the model.** [`tools/financial_calculator.py`](tools/financial_calculator.py)
  computes 15 metrics deterministically (margins, CAGR, ROE/ROA/ROIC, D/E, current ratio,
  FCF, EPS, P/E, EV/EBITDA); the model only explains the result.
- **Live market data is never fabricated.** With no provider configured, `LIVE_DATA`
  returns `"Current market data is not available."`
- **RAG citations are never invented.** Page numbers are recorded only when the parser
  actually reports them (PDFs); a `.txt` source cites the filename only.

## Repository layout

```
configs/            model_config.yaml (single source of truth) + training presets
models/             config, model, attention (MLA), layers, moe, mtp
data_sources/       registry, HF loader, cleaning, splitter, shards, mixer, manifest
training/           trainer, data_loader, instruction_dataset, instruction_trainer
evaluation/         evaluator, financial_metrics
tools/              financial_calculator
rag/                document_loader, chunker, embeddings, retriever
app/backend/services/  financial_router, chat_service, inspector
data/               raw/ processed/ shards/ evaluation/ instruction/
checkpoints/        base/ financial/ instruction/
```

## Checkpoints

Every checkpoint saves weights, optimizer state, step, train/val loss, best val loss,
tokens processed, the full model config, the dataset config and manifest snapshot, the
seed, runtime info (Python/PyTorch/CUDA/GPU), and a timestamp — as
`checkpoint_<step>.pt` plus a readable `checkpoint_<step>.json`.

## Limitations

- Runs verified here are **CPU-only**; no GPU was available in this environment.
- `tiny_debug` (100 steps, ~51K tokens) is a **pipeline test, not a trained model**.
  Generated text at that scale is incoherent, and evaluation accuracy reflects that.
- Not a licensed financial advisor. Educational/research use only; outputs must not be
  treated as personalized financial advice.
- Two datasets carry non-commercial licenses.
