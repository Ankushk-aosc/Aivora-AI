# SFT_EXPERIMENT_001 - the run that broke the thing it was meant to fix

Phase 14, and the Phase 15 gate applied to it. **Verdict: does not pass. Do not
promote.** The diagnosis is below, and it changed what I believe about the model.

## What was run

| | |
| --- | --- |
| base checkpoint | `checkpoint_247850`, sha256 `6e9e50ca59e5…` (frozen baseline) |
| dataset | `financial_sft_v1`, sha256 `eefba579…`, 7,849 rows / 522,736 tokens |
| train / validation | 7,470 / 379, split deterministically by row-id hash |
| format | one template, prompt masked, EOS trained (arm 2's format) |
| optimizer | AdamW, betas (0.9, 0.95), weight_decay 0.01, eps 1e-9 |
| schedule | peak lr 5e-5, 45-step warmup, cosine to 10% |
| batch / block | 4 / 192 |
| steps | 900 (2.6 hours, CPU, no GPU quota) |
| checkpoint | `checkpoints/sft_001/sft_001_best.pt`, sha256 `4e76a3f8892f27a0…` |

Validation loss fell monotonically and never plateaued: 1.9345 → 1.7784 → 1.5226
→ 1.3822 → 1.2135 → **1.1182** at step 900. Early stopping never fired.

(That loss is not comparable to the pretraining figure of 4.37 - only answer
tokens contribute and the distribution is instruction data.)

## Model-only results

| | baseline | E1 arm 1 (control) | **SFT_001** |
| --- | --- | --- | --- |
| dev (200 subset) | 1.64%* | 3.5% | **5.0%** |
| blind prose (144) | 7.64% | 10.42% | **5.56%** |
| dev hallucination | ~97.6%* | 95.83% | **89.58%** |
| blind hallucination | 92.36% | 89.58% | **79.86%** |
| **abstention accuracy** | **0%** | 0% | **62.5%** |
| **extraction** | **0/25** | 0% | **0%** |

\* baseline figures are over the full 795-item dev split.

Two real gains: **abstention went from 0% to 62.5%** - the first time this model
has ever declined to answer - and hallucination fell by 8 to 12 points on both
sets. Blind prose accuracy *fell*, and extraction stayed at zero.

## The diagnosis: SFT destroyed a capability the base model had

Extraction being 0% after 2,373 targeted examples is the anomaly, so I looked at
what the model actually emits. It produces the right FORM and the wrong VALUE:

```
What is Total liabilities?   expected 2,965.00   got '3,743.00'
What is Revenue?             expected 7,170.00   got '150.00'
What is Cash?                expected 2,700.00   got '795.00'
```

None of those numbers appear anywhere in their contexts. So it learned "answer
with a formatted number and stop" without learning to read the value.

Held-out items **in its own training format** scored 0/10 as well, so this is not
brittleness to the benchmark's layout.

Then the decisive test - can it copy anything at all, with no finance involved?

| copy-from-context probe (6 items) | result |
| --- | --- |
| frozen baseline `checkpoint_247850` | **5/6** |
| SFT_001 | **0/6** |

```
"The access code is XK42. What is the access code?"
  baseline -> "XK42.Question: What is the purpose of the access code?..."   (copies, cannot stop)
  SFT_001  -> "The provided information is insufficient to determine this."  (stops, cannot copy)
```

**The base model could already copy from context. The SFT run destroyed that and
replaced it with canned templates.** This is catastrophic forgetting, and it was
caused by how I built the dataset, not by the model's capacity.

## Why it happened - measured, not guessed

| task | rows | answer tokens | token share | mean answer length | distinct answers |
| --- | --- | --- | --- | --- | --- |
| calculation | 3,083 | 63,395 | 35.7% | 20.6 | 2,353 |
| reasoning | 407 | 30,267 | 17.0% | 74.4 | 407 |
| definition | 610 | 24,959 | 14.0% | 40.9 | **123** |
| interpretation | 483 | 19,928 | 11.2% | 41.3 | **6** |
| **extraction** | **2,373** | **17,858** | **10.0%** | **7.5** | 2,369 |
| abstention | 473 | 10,590 | 6.0% | 22.4 | **7** |
| grounded | 400 | 9,509 | 5.4% | 23.8 | 400 |
| explanation | 20 | 1,224 | 0.7% | 61.2 | 20 |

Three compounding design errors, all mine:

1. **Extraction is 30% of the rows but 10% of the loss.** Its answers are 7.5
   tokens; a definition's are 41. Loss is averaged over answer tokens, so the
   task I most wanted to teach contributed least to the gradient.
2. **956 rows share 13 distinct answer strings.** The interpretation and
   abstention sets repeat the same 6 and 7 sentences 81-84 times each. Emitting
   those strings unconditionally is the cheapest available loss reduction, and
   that is what the model learned to do - including on questions about access
   codes.
3. **Early stopping watched validation loss, which kept improving while
   capability was being destroyed.** This is the brief's warning about confusing
   lower loss with better capability, and it happened literally: loss 1.93 → 1.12
   while copy-from-context went 5/6 → 0/6.

## Phase 15 gate

**Failed.** The checkpoint is not promoted. It improves two metrics and removes a
capability the base model had, which is not a trade I can justify - and the brief
is explicit that the response to a failed gate is diagnosis, not more training.

## What changes for SFT_EXPERIMENT_002

Each fix addresses one measured cause:

1. **Balance by answer tokens, not rows.** Extraction needs roughly 4x its
   current share, either by more rows or by weighting short answers.
2. **Raise answer entropy.** Interpretation and abstention need many distinct
   phrasings; 6 sentences repeated 84 times teaches a reflex, not a behaviour.
3. **Keep a replay slice.** A fraction of plain pretraining-style text, so
   general behaviour - including copying - has gradient support.
4. **Gate on capability, not loss.** The copy probe and extraction accuracy run
   at every evaluation, and the run stops when they degrade, whatever the loss is
   doing.
5. **Lower peak learning rate and fewer steps**, since 900 steps at 5e-5 was
   enough to overwrite a pretrained behaviour.

## What this does to the training decision

`docs/AIVORA_TRAINING_DECISION.md` listed D (model capacity) as unproven. This
experiment is evidence **against** capacity being the binding constraint for
extraction: a 101.7M model that copies 5 of 6 context tokens already has the
mechanism. What it lacks is a way to stop, which arm 2 tests, and an SFT recipe
that does not trade one capability for another.

The order of blame, updated by evidence:

* **format and objective (B/F)** - evidenced, and now partly fixed
* **SFT recipe** - newly evidenced, and mine to fix
* **data coverage (A)** - still evidenced for prose breadth
* **capacity (D)** - weaker than I thought, given the copy probe
