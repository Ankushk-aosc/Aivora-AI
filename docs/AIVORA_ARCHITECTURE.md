# Aivora — current-state architecture and training pipeline

Written from inspection of the repository at commit `ff67b75`, not from the
project's own documentation. Every number here was read out of code, a
checkpoint's metadata, or a measurement run against the checkpoint; where
something is unverified it says so.

**Baseline artifact:** `checkpoint_247850` — step 247,850, 1,964,851,200 tokens,
val loss 4.4258, sha256 `6e9e50ca…b004`. Published at
[Ankush0845/financial-llm-from-scratch](https://huggingface.co/Ankush0845/financial-llm-from-scratch).

---

## 1. Model

`models/` (~600 lines, torch only). Config resolved from
`DeepSeekConfig.default()`:

| field | value | field | value |
| --- | --- | --- | --- |
| vocab_size | 50257 (GPT-2 BPE via `tiktoken`) | n_experts | 8 |
| block_size | 1024 | n_experts_per_token | 2 |
| n_layer | 8 | expert_intermediate_size | 512 |
| n_embd | 512 | shared_expert_intermediate_size | 768 |
| n_head | 8 | use_shared_expert | true |
| kv_lora_rank | 128 | mtp_num_heads | 1 |
| q_lora_rank | 192 | mtp_loss_weight | 0.3 |
| rope_dim | 32 | aux_loss_weight | **0.0** |
| dropout | 0.1 | bias | true |

Parameter count: **101,723,264** trainable; 136,892,248 tensors in the
state_dict (the difference is buffers, e.g. each layer's causal mask).

Components: `attention.py` (multi-head latent attention — compressed KV with
separate RoPE projections), `moe.py`, `mtp.py` (multi-token prediction head),
`layers.py` (SwiGLU), `model.py`.

### MoE routing — measured, not assumed

`MoELayer` uses **auxiliary-loss-free** balancing: `aux_loss_weight` is 0 and a
per-expert bias buffer is nudged ±0.001 per training step toward balance. The
layer computes `expert_usage` every forward pass and discards it.

`scripts/diagnose_moe.py` hooks the routers and replays real validation tokens.
On the baseline checkpoint, 6,144 tokens:

| | |
| --- | --- |
| perfect balance | 12.5% of assignments per expert |
| observed | 3.6% – 20.6%, normalised entropy **0.95 – 1.00** |
| collapsed layers | **0 of 8** measured |

There is no expert collapse and therefore no dead capacity to recover. A 9th
MoE layer lives in the MTP head and only runs during training.

---

## 2. Tokenizer and data pipeline

`data_sources/` — GPT-2 BPE (`tiktoken`), no custom tokenizer.

* `dataset_registry.py` — 19 registered sources with `hf_id`, licence,
  category (FineWeb-Edu, Investopedia, EDGAR/SEC, Sujet-Finance-177k,
  finance-alpaca, FinQA, Dolly, OASST1, SQuAD, GSM8K, Wikipedia, TinyStories).
* `prepare.py` / `shard_writer.py` — streams, tokenizes and writes `uint16`
  shards under `data/shards/<dataset>/{train,validation}/` with an
  `index.json`; `manifest.py` records provenance into
  `data/dataset_manifest.json`, which is embedded in every checkpoint.
* `build_instruction_dataset.py` — builds the instruction JSONL (38,477 rows
  from finance-alpaca, Investopedia, Dolly) with dedup, length filters, and a
  leakage check against `data/evaluation/*.jsonl`. **It found and excluded one
  real leak.**

Training mix (`configs/financial_poc.yaml`): 55% financial, 45% general.

---

## 3. Training

`training/trainer.py` (620 lines). Hard-won details that must not be
regressed:

| mechanism | detail |
| --- | --- |
| optimizer | AdamW, betas (0.9, 0.95), weight_decay 0.1, eps 1e-9 |
| schedule | warmup → cosine to `min_lr`; **no torch LR scheduler** — a plain function sets `param_group["lr"]` *before* each step |
| resume LR | a resume reads `preset["continuation"]` (peak 3e-5), **not** the from-scratch 3e-4. Resuming at 3e-4 diverged val 6.07 → 53.9 |
| resume warmup | 500 steps ramping from `min_lr`, because `optimizer.load_state_dict` restores the *old* LR |
| precision | fp16 + `GradScaler` gated strictly on dtype; bf16/fp32 untouched |
| stability | `clip_grad_norm_(1.0)` after `unscale_`; 25-consecutive-skip abort |
| budgets | `max_train_seconds` (Kaggle kills sessions at 12 h and may drop outputs), `keep_last_checkpoints` (20 GB output cap; keeps newest **and** best-val) |
| multi-GPU | `DataParallel` when >1 GPU; returns the *unwrapped* model |
| horizon | `lr_horizon_override` so a short probe sees the real schedule's LR |

`training/instruction_trainer.py` (Stage B) now mirrors those safeguards and
adds early stopping plus best-weight reload. `training/hf_lora_sft.py` does
LoRA on a Hugging Face model, with version-tolerant `TrainingArguments`.

### Checkpoint format

`torch.save` of `{model_state_dict, optimizer_state_dict, step, best_val_loss,
tokens_processed}` (~1.2 GB) plus a sidecar `.json` carrying model config,
dataset config, dataset manifest, seed, runtime, and `effective_hparams` — the
values actually used, because `dataset_config.learning_rate` records the
preset's from-scratch LR even on a continuation run.

---

## 4. Evaluation (existing)

`evaluation/`:

* `evaluator.py` — category sets in `data/evaluation/` (financial_qa 10,
  terminology 12, numerical 15, reasoning 8 = **45 questions**), plus
  `check_leakage()` which decodes shards and substring-matches questions.
* `financial_metrics.py` — exact/normalized match, keyword coverage, numeric
  tolerance, `concept_overlap` (documented as unreliable) and **`rubric_match`**
  (per-question concept groups).
* `generic.py` — `evaluate_generator(fn)` so *any* model scores on the same
  questions, which is how Qwen and Aivora were compared.
* `data/evaluation/holdout.jsonl` — **50** later questions, 30 rubric-graded,
  20 numeric.

Known scorer weakness, measured: string matching fails correct paraphrases
("Depreciation spreads the cost of a tangible asset … across its useful life"
scored 0 against "spreading the cost of a tangible asset over its useful
life"); rubric grading agreed with human judgement on 5/5 such cases.

---

## 5. Serving pipeline

`app/backend/` — stdlib `http.server`, ~60 endpoints, static frontend.

Answer path (`chat_service.py` + `financial_router.py`):

1. **figures present** → `tools/financial_calculator.py` (20+ formulas) and
   `tools/derived_calculations.py` (multi-step identities: ROE from assets and
   liabilities, P/E from net income and shares, EBIT from EBITDA, growth
   between periods, margins from components);
2. **glossary term present** → 92-entry curated glossary, longest key wins;
3. **paraphrase** → `knowledge_retrieval.py`: MiniLM embeddings with a TF-IDF
   fallback, accept ≥0.60 or ≥0.45 with a shared stem;
4. **otherwise** → the model, shown only if `quality.py` does not flag it.

`services/generation.py` abstracts the backend: `DeepSeekBackend` (this
project's checkpoints) or `HFBackend` (any HF causal LM + optional LoRA).

Security (all covered by `tests/test_api_security.py`, 25 checks): static
files confined to the frontend directory (a traversal served `data/auth.db`
before), client paths confined, self-registration fixed to `viewer`,
code-execution admin-only, CORS allow-list, body-size cap.

---

## 6. Experiment tracking and reproducibility — already present

`experiments/tracker.py` records name, model config, dataset config, metrics,
seed, checkpoint, notes, timestamp and an `environment()` block (python, torch,
platform, CUDA, GPU, **git revision**). Four experiment directories exist.
`ai_platform/model_registry.py` registers checkpoints with sha256 and verifies
integrity.

**This substantially covers the plan's Phase 1 and Phase 15. It should be
extended, not rebuilt.**

---

## 7. What the measurements say about capability

| system | score | grading |
| --- | --- | --- |
| Aivora model alone | **1–5 / 45** (2–11%) | sampled generation, hence the spread |
| Aivora + full pipeline | **50 / 50** held-out | rubric + numeric |
| Qwen2.5-1.5B-Instruct, untuned | **38 / 45** | greedy |
| Qwen + the project's finance LoRA | **40 / 45** | greedy |
| Aivora after distillation from that teacher | 4 / 45 | greedy |

Training history, all measured: four pretraining sessions to the planned
247,850 steps (the final 10,488-step LR anneal moved val loss by nothing);
instruction tuning at 2e-5 and 5e-6 (both overfitted inside 500 steps);
distillation (style improved, facts did not); MoE routing healthy.

**Conclusion for planning:** the model's ceiling is capacity — 101M parameters
on ~2B tokens — not schedule, routing or data volume. Pipeline components
(calculator, glossary, retrieval, guard) carry essentially all current
accuracy. Any plan that expects a large gain from more pretraining of this
checkpoint contradicts the evidence above.

---

## 8. Gaps against the proposed plan

| plan phase | status |
| --- | --- |
| 1 baseline preservation | tracker exists; needs per-experiment dirs for the new work |
| 2 benchmark (~1,000 q) | **built: 999 items** (795 dev / 204 hidden test) in `data/benchmark/`, 896 generated with computed answers + 103 authored with rubrics, 46 abstention items; `tests/test_benchmark.py` recomputes 204 answers and verifies split hygiene |
| 3 baseline evaluation | **done**: `reports/baseline_report.{json,md}` (795 dev items, 4 ablations) + `reports/baseline_analysis.md`. model alone 0.88%, full pipeline 58.36%; the calculator is worth +53 pts, retrieval +4 |
| 4–5 dataset + reasoning levels | builder exists (38k); no difficulty levels, no statement-analysis data |
| 6 instruction tuning | 2 runs done; 1e-6…5e-6 untested |
| 7 distillation v2 | v1 done; verifier stage **not** built |
| 8 hard-example mining | not built |
| 9–10 tools + RAG | built (calculator, glossary, retrieval); no SEC/live data |
| 11 guard | degeneracy guard only; no abstention/claim checking |
| 12–13 matrix + ablations | not built; `evaluate_generator` makes it straightforward |
| 14 performance/memory | throughput measured (~10.5k tok/s, 8.83 GiB); no leak audit |
| 16 production architecture | matches today's pipeline, minus SEC data and verification |

---

## 9. Honest notes on cost

Phases 2, 4, 5 and 7 are the expensive ones, and mostly *authoring* rather than
compute: a 1,000-question benchmark with hidden split, difficulty-tiered
reasoning data, and a verified teacher→student pipeline. On a free Kaggle
quota (~30 GPU h/week) and this machine (8 GB RAM, no local GPU), the practical
order is: benchmark → baseline report with ablations → targeted data → one
controlled SFT experiment at a time.
