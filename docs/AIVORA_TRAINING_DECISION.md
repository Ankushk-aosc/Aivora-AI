# Why Aivora is weak, and the smallest experiment that would prove how to fix it

Step 13. Each candidate cause is marked by what the evidence actually supports.
Frozen state: tag `AIVORA_PRE_TRAINING_BASELINE`.

## The evidence base

| measurement | value | source |
| --- | --- | --- |
| model-only, hidden split | 3.43% (7/204), hallucination 96.39% | post-hoc diagnostic |
| model-only, dev split | 1.64% (13/795) | after the inference fix |
| model-only, blind prose | 7.64% (11/144) | blind set |
| system correct / model wrong | **189 of 204 (92.65%)** | four-group comparison |
| system wrong / model correct | **0 of 204** | four-group comparison |
| abstention rate | **0.00%**, abstention accuracy 0/10 | diagnostic |
| EOS stops | **0 of 204**; `<|endoftext|>` never emitted | generation traces |
| extraction, model-only | **0 of 25** | by-intent breakdown |
| accuracy by difficulty (1-5) | 3.57 / 4.84 / 1.49 / 5.26 / 3.57 - flat | by-difficulty breakdown |
| unique training tokens | 120,857,129 | checkpoint manifest |
| tokens processed | 1,964,851,200 = **16.3 epochs** | checkpoint manifest |
| unique tokens per parameter | **1.19** (optimal ≈ 20) | derived |
| optimizer updates | ~15,491 | trainer accounting |
| MoE routing entropy | 0.95-1.00, 0 of 8 layers collapsed | `scripts/diagnose_moe.py` |
| sampling matrix | 0% premature stops, accuracy unmoved | 12 configs x 3 lengths |

## A. Data limitation - **EVIDENCED, and primary**

1.19 unique tokens per parameter is about **6% of compute-optimal**. The corpus
was 121M tokens shown 16.3 times, which is a recipe for memorising a small
sample rather than learning a domain. Validation loss plateaued at 4.37 while
capability stayed near zero - consistent with a corpus exhausted rather than a
model saturated.

And the specific behaviours the benchmark tests have **no data behind them at
all**: no dataset in the mixture teaches ratio interpretation, valuation
reasoning, risk analysis, or abstention. Those are exactly the categories at 0%.

## B. Instruction-following limitation - **EVIDENCED, and primary**

Pretraining used no template, no prompt masking, and consumed instruction and QA
data (6.8% of tokens) as undifferentiated text. Serving then prompts with
`"Question: …\nAnswer:"`, a convention the model never saw as a consistent
pattern.

The sharpest evidence is **extraction: 0 of 25**. Reading a labelled figure back
out of a block of text requires no reasoning and no world knowledge - only
copying the number next to the asked label. A model that fails all 25 is not
failing to reason; it is failing to *respond to a request at all*. It continues
text, which is what it was trained to do.

## C. Reasoning limitation - **NOT ISOLABLE from the evidence**

Reasoning scores 0 of 25 and calculation 2.78%, which looks like a reasoning
failure and cannot be attributed as one. Any model that does not answer
questions will score zero on reasoning questions, so A, B and F fully account
for these numbers without invoking C. **Reasoning capability has not been
measured, because nothing in the pipeline ever asked the model to reason in a
format it could recognise.**

The flat accuracy across difficulty levels 1-5 supports this reading: a model
with partial reasoning degrades as difficulty rises. This one does not vary,
because it is not on the curve at any level.

## D. Model-capacity limitation - **UNPROVEN; my earlier claim retracted**

`docs/AIVORA_ARCHITECTURE.md` previously concluded "the model's ceiling is
capacity - 101M parameters on ~2B tokens". That was wrong, and the error was
mine: I read `tokens_processed` as unique tokens. The run used 121M unique
tokens.

**No capacity conclusion is available from a run at 6% of compute-optimal data
with a format that could not teach the task.** 101.7M parameters may well be
insufficient for financial reasoning - that is plausible and untested. It cannot
be asserted from this evidence.

## E. Architecture limitation - **NO EVIDENCE OF A DEFECT**

MoE routing entropy is 0.95-1.00 with zero collapsed layers, measured on real
validation tokens. MLA, the MTP head, the fp16 + GradScaler + clipping recipe and
the resume-LR safeguard all work as intended. Nothing measured points at the
architecture.

## F. Training-objective limitation - **EVIDENCED**

`encode_ordinary` plus no separator means **`<|endoftext|>` never appeared in the
training data**, and the model emitted it 0 times in 204 generations. It has no
representation of a completed answer, which is why 201 of 204 answers ran to the
token cap and why answers spill into a fresh `"Question:"`.

Likewise nothing in the objective rewarded declining to answer, and the measured
abstention rate is **0.00%** - not low, zero. The model has no mechanism for "I
do not know".

## Verdict

**Primary: A (data) and B/F (format and objective), jointly.** They are
separable from each other but both are clearly evidenced, and each is sufficient
on its own to produce what was measured.

**C and D are live hypotheses that the current evidence cannot test.** Anyone
claiming Aivora is too small, or cannot reason, is asserting beyond the data -
including my own earlier report.

**E is not supported.**

---

## The smallest experiment that would settle it

**E1: a format-controlled comparison at fixed corpus and fixed token budget.**

Two arms, identical in every respect except the three format changes:

| | arm 1 (control) | arm 2 (format fixed) |
| --- | --- | --- |
| corpus | the existing 121M tokens | the same 121M tokens |
| token budget | 484M processed (4 epochs) | 484M processed (4 epochs) |
| architecture | unchanged | unchanged |
| `<|endoftext|>` between documents | no | **yes** |
| prompt masking on instruction/QA rows | no | **yes** |
| single consistent template, also used at inference | no | **yes** |
| estimated GPU time | ~13 hours | ~13 hours |

Total ~26 GPU hours, inside one week of the free quota, and it does not require
assembling a new corpus.

**What each outcome proves:**

* **Arm 2 learns to emit `<|endoftext|>` and accuracy moves materially** (say
  model-only above 15% on an external benchmark): the model was *mis-trained*,
  not merely under-trained. B and F dominate, and the expensive data programme
  can be scoped afterwards with a working recipe.
* **Arm 2 learns `<|endoftext|>` but accuracy does not move**: format was
  necessary and insufficient. A becomes the live hypothesis, and the ~53-hour
  2B-token run (Strategy A in `AIVORA_EXPERIMENT_PLAN.md`) is justified with a
  clean prior.
* **Arm 2 does not even learn to stop**: something is wrong in the data pipeline
  beyond what this audit found, and no scaling run should be started until it is
  understood.

Arm 1 exists so that the comparison is not against a differently-trained
historical checkpoint. It is worth 13 hours precisely because without it, any
improvement could be attributed to the format change or to the four-epoch
schedule, and nothing would be learned.

**Prerequisite, not optional:** an external benchmark. The dev split is saturated
at 100%, the hidden split has been evaluated three times, and the blind set was
written by the same author as the glossary it tests. `evaluation/external.py`
accepts an outside benchmark with no code changes; supplying one is the gate on
every number above.

## What I would not do next

* Retrain the current checkpoint with the same recipe. It reproduces 1.64%.
* Run SFT on this base. It was tried twice and overfitted within 500 steps, and
  the base has not changed since.
* Scale to 350M parameters. It is ~555 GPU hours, out of reach on the current
  budget, and would confound capacity with data and format all at once.
* Add more glossary entries or patterns to raise the blind-prose score. That
  treats the measurement, not the model, and the space of phrasings is unbounded.
