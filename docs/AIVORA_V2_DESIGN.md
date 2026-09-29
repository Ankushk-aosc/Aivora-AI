# Aivora V2: candidate designs

Step 11. **Nothing here is implemented or trained.** Four candidates, with the
numbers that decide between them and the risks that could sink each.

Compute estimates use this project's **measured** throughput on Kaggle
(~10,500 tokens/second for the 101.7M configuration, from the training logs),
scaled by parameter count, not a theoretical FLOP count. They are estimates for
planning, not promises, and they assume the free Kaggle quota (~30 GPU
hours/week, 12-hour session cap, ~16 GB VRAM).

**No accuracy prediction appears in this document.** The current evidence cannot
support one: the only run this project has completed was at 6% of
compute-optimal data with a training format that could not teach question
answering, so it establishes no scaling point to extrapolate from.

## The constraint that shapes every option

From `docs/AIVORA_TRAINING_DATA_AUDIT.md`: the completed run used **120.9M
unique tokens for 101.7M parameters - 1.19 tokens per parameter**, against a
compute-optimal ratio of roughly 20. Data, not parameters, is the axis that has
never been tested. Every design below therefore specifies its corpus first.

Also required in all designs, from
`docs/AIVORA_TRAINING_PIPELINE_AUDIT.md`, and cheap:

* `<|endoftext|>` written between documents when shards are built
* one consistent prompt template, used in training and at inference
* prompt masking wherever instruction or QA data is used
* capability evaluation during the run, not loss alone
* supervision for the behaviours the benchmark tests: ratio interpretation,
  valuation reasoning, risk analysis, abstention

---

## V2-A: same architecture, corpus that matches it

The control experiment. It answers the question the project has never asked:
what does *this* architecture do when fed properly?

| field | value |
| --- | --- |
| parameters | 101.7M (unchanged) |
| layers / hidden / heads | 8 / 512 / 8 (unchanged) |
| attention | MLA, kv_lora_rank 128, q_lora_rank 192, rope_dim 32 (unchanged) |
| MoE | 8 experts, top-2, shared expert (unchanged - measured healthy) |
| context | 1024 (unchanged) |
| tokenizer | GPT-2 BPE, 50,257 (unchanged) |
| **target unique tokens** | **2.0B (≈20/parameter)** |
| corpus requirement | ~16x today's 121M: more FineWeb-Edu, more EDGAR, more Investopedia-class explanatory text, plus the missing categories |
| epochs | 1-2, not 16 |
| estimated GPU time | 2.0B / 10,500 tok/s ≈ **53 hours** |
| calendar on free quota | ~2 weeks |
| VRAM | ~4-6 GB (fp16 params 0.2 GB, AdamW states ~0.8 GB, activations at batch 8x1024) - comfortable |
| storage | ~4 GB of uint16 shards |
| risks | corpus assembly is the real work, and licence-clean financial text at this volume is not guaranteed to exist; if the result is still weak, that is genuine evidence about capacity |

**Why it is worth doing first:** it is the only design that isolates one
variable. Every other option changes the model and the data at once, and then
nothing is learned about either.

## V2-B: compute-optimal at a smaller size

If 2B tokens cannot be assembled, shrink the model to match the corpus that can.

| field | value |
| --- | --- |
| parameters | ~60M |
| layers / hidden / heads | 6 / 384 / 6 |
| attention | MLA, kv_lora_rank 96, q_lora_rank 128, rope_dim 32 |
| MoE | 4 experts, top-2, shared expert |
| context | 1024 |
| tokenizer | unchanged |
| target unique tokens | ~1.2B (≈20/parameter) |
| estimated GPU time | ~1.2B / ~17,000 tok/s ≈ **20 hours** |
| calendar on free quota | under one week |
| VRAM | ~3-4 GB |
| risks | a smaller model may be below the size where financial reasoning appears at all; it tests the recipe rather than producing a better product |

**Why it is worth considering:** it fits the quota with room to iterate, and a
correctly-fed 60M model is a more informative baseline than an underfed 101M one.

## V2-C: scale up

| field | value |
| --- | --- |
| parameters | ~350M |
| layers / hidden / heads | 16 / 1024 / 16 |
| attention | MLA, kv_lora_rank 256, q_lora_rank 384 |
| MoE | 8 experts, top-2, expert intermediate 1024 |
| context | 2048 |
| tokenizer | unchanged |
| target unique tokens | ~7B (≈20/parameter) |
| estimated GPU time | 7B / ~3,500 tok/s ≈ **555 hours** |
| calendar on free quota | **~4-5 months** - not feasible |
| VRAM | ~10-12 GB with activation checkpointing; tight but possible on 16 GB |
| risks | infeasible on the current budget; a 12-hour session cap means ~46 resumes, each a failure opportunity; 7B licence-clean financial-leaning tokens is a substantial data programme |

Included because the brief asked for the comparison, and because it makes the
budget explicit: **this is the design most likely to produce a genuinely useful
from-scratch financial model, and it is out of reach without paid compute.**

## V2-D: no new base model - adapter on an open model

| field | value |
| --- | --- |
| base | Qwen2.5-1.5B-Instruct (frozen) |
| trained parameters | LoRA adapters, ~10-20M |
| context | 32,768 (the base's) |
| tokenizer | the base's |
| training data | the instruction/QA/interpretation set, ~10-50M tokens |
| estimated GPU time | **2-6 hours** |
| VRAM | ~8-10 GB for LoRA fine-tuning at short context |
| local feasibility | **inference does not fit this machine**: 8 GB RAM total, 0.4 GB free, and the 1.5B weights are ~3.1 GB; Kaggle or another host is required |
| risks | it is no longer a from-scratch model, which may defeat the purpose of the project; the adapter path is already partly built and tested here |

Evidence already in this repository: Qwen2.5-1.5B-Instruct scored **38/45**
untuned and **40/45** with this project's finance LoRA, against Aivora's
**1-5/45**, on the same questions through the same scorer.

---

## Side by side

| | V2-A | V2-B | V2-C | V2-D |
| --- | --- | --- | --- | --- |
| parameters | 101.7M | ~60M | ~350M | 1.5B frozen + ~15M |
| unique tokens | 2.0B | 1.2B | 7B | 10-50M (SFT only) |
| tokens/parameter | ~20 | ~20 | ~20 | n/a |
| GPU hours (estimated) | ~53 | ~20 | ~555 | 2-6 |
| feasible on free quota | yes, ~2 weeks | yes, <1 week | **no** | yes |
| isolates one variable | **yes** | no | no | no |
| still a from-scratch model | yes | yes | yes | **no** |
| main risk | corpus assembly | may be too small to reason | compute budget | not the project's stated goal |

## What I would need before recommending one

* **Corpus feasibility.** Can 2B licence-clean tokens with a meaningful financial
  share actually be assembled? `data_sources/dataset_registry.py` already lists
  19 sources; the audit shows the run used ~121M of what they can supply, so the
  headroom exists but has not been measured.
* **A measurement that survives.** Both existing benchmarks are spent. The
  external evaluator (`evaluation/external.py`) exists for this, but a fresh
  benchmark from outside this repository is a prerequisite for judging any V2.
* **The decision on whether "from scratch" is the goal or the means.** V2-D is
  the cheapest route to a working product and the least interesting answer to the
  project's original question. That is a judgement about intent, not evidence.
