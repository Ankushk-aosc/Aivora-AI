# Aivora training pipeline audit

Step 10. Read from `training/trainer.py`, `models/`, `configs/financial_poc.yaml`
and the checkpoint's own `effective_hparams`. Nothing was changed.

## 1. Architecture

| field | value |
| --- | --- |
| parameters | 101,723,264 trainable |
| layers | 8 |
| hidden size | 512 |
| attention heads | 8 |
| attention | Multi-head Latent Attention: `kv_lora_rank` 128, `q_lora_rank` 192, `rope_dim` 32 |
| MoE | 8 experts, top-2 routing, expert intermediate 512, shared expert 768 |
| MoE balancing | auxiliary-loss-free; `aux_loss_weight` 0.0, per-expert bias nudged ±0.001 per step |
| extra head | Multi-Token Prediction, 1 head, loss weight 0.3 |
| dropout | 0.1 |
| context length | 1024 |
| tokenizer | GPT-2 BPE via `tiktoken`, vocab 50,257 |

MoE health was measured, not assumed: `scripts/diagnose_moe.py` over real
validation tokens gives normalised routing entropy 0.95-1.00 and **0 of 8 layers
collapsed**. There is no dead capacity to recover, and no evidence of an
architectural defect.

## 2. Objective and loss

| item | value |
| --- | --- |
| objective | next-token cross-entropy over the whole sequence |
| auxiliary objective | multi-token prediction, weighted 0.3 |
| MoE auxiliary loss | none (0.0) |
| **prompt masking** | **none** |
| **document separators** | **none** - `encode_ordinary`, ids concatenated |
| **packing** | contiguous blocks across document boundaries |
| label construction | shift-by-one over the packed stream |

This is the finding that matters most in this document. The objective was
"predict the next token of a stream of concatenated financial and general text".
No part of it distinguished a question from an answer, marked where an answer
ended, or rewarded declining to answer. Instruction and QA datasets were present
(6.8% of tokens) but consumed as undifferentiated text.

`training/instruction_trainer.py` **does** implement prompt-masked labels and a
template - but that is Stage B, run separately, and the published checkpoint is
the pretraining output. Two Stage B runs were attempted (learning rates 2e-5 and
5e-6) and both overfitted within 500 steps on 800-38k instruction rows.

## 3. Optimisation

| item | value |
| --- | --- |
| optimizer | AdamW, betas (0.9, 0.95), weight_decay 0.1, eps 1e-9 |
| precision | float16 with `GradScaler`, gated strictly on dtype |
| gradient clipping | 1.0, applied after `unscale_` |
| schedule | warmup then cosine to `min_lr`, set manually per step (no torch scheduler) |
| from-scratch peak LR | 3.0e-4 |
| continuation peak LR | 3.0e-5, min 1.0e-5, warmup 500 steps |
| micro-batch | 8 sequences x 1024 tokens = 8,192 tokens |
| gradient accumulation | 16 |
| effective batch | 128 sequences = 131,072 tokens per optimizer update |
| recorded steps | 247,850 - these are **micro-steps** (`tokens_processed += X.numel()` per micro-step) |
| **optimizer updates** | **~15,491** (247,850 / 16) |
| divergence guard | abort on non-finite loss; 25-consecutive-skip abort; val-loss threshold |

So the run performed roughly **15.5 thousand weight updates**. For comparison,
small language models of this size are typically trained for hundreds of
thousands of updates. The LR schedule was consumed over micro-steps.

The continuation LR of 3e-5 exists because resuming at the from-scratch 3e-4
diverged (val 6.07 to 53.9). That safeguard is correct and should be kept.

## 4. Data flow

| item | value |
| --- | --- |
| unique training tokens | 120,857,129 |
| validation tokens | ~4.9M budget |
| tokens processed | 1,964,851,200 = **16.3 epochs** |
| shard format | `uint16` token ids with an `index.json` per dataset split |
| mixture | sampled per batch from the configured weights |
| validation procedure | periodic loss over held-out shards; best-val checkpoint kept alongside the newest |
| provenance | every dataset's licence and record counts recorded in the manifest, embedded in each checkpoint |

Validation is loss-only. **No capability evaluation ran during training**, so
nothing in the loop would have revealed that the model could not answer a
question - val loss fell while question-answering ability stayed at ~1%.

## 5. Prompt and template format

Pretraining used **no template**. Serving uses:

```
Question: {query}\nAnswer:
```

This format appears nowhere in the training data as a consistent pattern, so at
inference the model is being asked to follow a convention it never learned. The
instruction trainer's own format (`prompt_style`) was introduced later, for
Stage B.

## 6. What was Aivora actually trained to do?

**To continue financial-flavoured text, for 15.5 thousand updates, over a
121M-token corpus seen 16 times, with no notion of a question, an answer
boundary, or an admissible refusal.**

Everything the measurements show follows from that sentence:

| measured behaviour | explanation in the pipeline |
| --- | --- |
| never emits `<|endoftext|>` (0.0%) | the token was never in the training data |
| runs on into a new "Question:" | no answer-boundary signal existed |
| never abstains (abstention accuracy 0%) | no data or objective taught refusal |
| fluent but wrong prose | next-token objective on a small corpus rewards plausibility |
| 1.64% on arithmetic-shaped questions | no arithmetic supervision; MTP does not supply it |
| validation loss plateaued at 4.37 | 16 epochs on 121M tokens exhausts what the corpus can teach |

## 7. What is sound and should be preserved

* MoE routing is healthy and the auxiliary-loss-free balancing works.
* MLA, the MTP head and the fp16 + GradScaler + clipping recipe are correct.
* The resume-LR safeguard, checkpoint pruning that keeps the best-val checkpoint,
  the time budget for Kaggle sessions, and the manifest embedded in every
  checkpoint are all good engineering and were hard-won.
* Provenance and licence tracking is better than most projects of this size.

## 8. What must change before another run

In order of expected effect, all of them cheap relative to GPU time:

1. **Insert `<|endoftext|>` between documents** when writing shards. One line in
   `shard_writer.py`. Without it no model trained here can learn to stop.
2. **Train on more unique tokens.** 1.19 tokens per parameter is ~6% of
   compute-optimal; the corpus, not the parameter count, is the binding
   constraint that has been tested.
3. **Mask prompts and use one consistent template** wherever instruction or QA
   data is used, and use the same template at inference.
4. **Evaluate capability during training**, not just loss - the existing
   benchmark harness can run on a checkpoint mid-run.
5. **Add supervision for the target behaviours**: ratio interpretation, valuation
   reasoning, risk analysis, and explicit abstention examples. None exist today.
