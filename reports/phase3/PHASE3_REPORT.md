# Phase 3 — zero-shot and few-shot base models

**Status: PROVISIONAL.** Run before the scorer was validated against your hand
labels (EXPERIMENT_RULES_v2 §4) and before you supplied the threshold numbers
(§1 still reads `<your numbers>`). Every figure below is **selection split,
n=61** — the split that exists to choose a configuration and is never reported
as a result. The frozen set has not been touched.

Date: 2026-10-05 · Hardware: Kaggle T4 · Repo commit `490708b`
Raw: [`phase3_base_models.json`](phase3_base_models.json) (sha256 `509a905c…`)
Scorer: `scripts/eval_heldout.py`, identical for every system in this project.

---

## 1. Results — all ten configurations

Greedy decoding, one system prompt, one scorer. The only variables are the model
and the number of shots.

| configuration | overall | copy | extraction | wording | calculation | abstention | invented values | s/item |
|---|---|---|---|---|---|---|---|---|
| **Qwen2.5-1.5B (3-shot)** | **90.0%** | 100.0% | **100.0%** | 88.2% | 33.3% | 100.0%* | **0.0%** | 0.44 |
| Qwen2.5-1.5B (0-shot) | 85.0% | 100.0% | 86.7% | 88.2% | 33.3% | 85.7% | 0.0% | 0.34 |
| SmolLM2-1.7B (0-shot) | 73.3% | 100.0% | 86.7% | 94.1% | 0.0% | 0.0% | 0.0% | 0.35 |
| SmolLM2-1.7B (3-shot) | 71.7% | 100.0% | 60.0% | 70.6% | 0.0% | 100.0%† | 0.0% | 0.30 |
| Qwen2.5-0.5B (3-shot) | 68.3% | 100.0% | 73.3% | 82.3% | 16.7% | 0.0% | 1.9% | 0.45 |
| SmolLM2-360M (0-shot) | 68.3% | 86.7% | 100.0% | 70.6% | 16.7% | 0.0% | 0.0% | 0.89 |
| Qwen2.5-0.5B (0-shot) | 65.0% | 93.3% | 73.3% | 82.3% | 0.0% | 0.0% | 5.7% | 0.40 |
| SmolLM2-135M (0-shot) | 36.7% | 93.3% | 20.0% | 29.4% | 0.0% | 0.0% | 1.9% | 2.43 |
| SmolLM2-135M (3-shot) | 36.7% | 80.0% | 13.3% | 5.9% | 0.0% | 100.0%† | 0.0% | 0.47 |
| SmolLM2-360M (3-shot) | 15.0% | 6.7% | 0.0% | 5.9% | 0.0% | 100.0%† | 0.0% | 0.54 |

No configuration failed to run. `*` genuine — see §3. `†` **artifact, not a
capability** — see §3.

Leader, with intervals (Wilson, 95%):

| gate | Qwen2.5-1.5B 3-shot | 95% CI |
|---|---|---|
| copy | 15/15 — 100% | [79.6, 100] |
| extraction | 15/15 — 100% | [79.6, 100] |
| wording | 15/17 — 88.2% | [65.7, 96.7] |
| calculation | 2/6 — 33.3% | [9.7, 70.0] |
| abstention | 7/7 — 100% | [64.6, 100] |
| over-refusal | 0/1 | — |
| invented values | 0/53 — 0% | — |

The per-gate n is small (6–17). These intervals are wide and they are the reason
this is a selection decision, not a result.

## 2. The hardware control passed

SmolLM2-135M 0-shot was re-run on Kaggle purely as a control. It reproduced the
local run **exactly** — overall 36.7%, copy 93.3%, extraction 20.0%. The T4 run
and the 8 GB laptop run agree digit for digit, so nothing in the comparison is a
hardware effect. *(measured)*

## 3. The few-shot abstention artifact — a defect in my prompt, not a finding

Four configurations score 100% on the abstention gate. Three of those scores are
worthless, and the cause is my own few-shot block:

| configuration | answerable items wrongly refused | abstention gate | over-refusal |
|---|---|---|---|
| SmolLM2-360M (3-shot) | **50/53** | 100% | 100% |
| SmolLM2-135M (3-shot) | 31/53 | 100% | 100% |
| SmolLM2-1.7B (3-shot) | 16/53 | 100% | 100% |
| Qwen2.5-0.5B (3-shot) | 2/53 | 0% | 0% |
| **Qwen2.5-1.5B (3-shot)** | **0/53** | 100% | **0%** |

One of my three few-shot examples (`scripts/eval_base_models.py`, `FEW_SHOT[2]`)
is an abstention. At 1-in-3 it is far above the evaluation's true base rate, and
every SmolLM2 model copied the refusal rather than the behaviour: 360M 3-shot
refused 50 of 53 answerable items, which is the whole of its collapse to 15.0%
and its copy score of 6.7%. Those models score 100% on abstention because they
refuse *everything* — the gate cannot distinguish them from a model that always
says "insufficient information". **A 100% abstention score is only meaningful
read next to over-refusal**, which is exactly why the rules require the two to be
reported separately. Qwen2.5-1.5B 3-shot is the one configuration that earns it:
7/7 refusals where refusal is right, 0/53 where it is wrong. *(measured)*

This is a prompt defect I introduced, and it means **the 3-shot numbers for the
SmolLM2 family measure my prompt, not the models.** If few-shot is used in
Phase 4 the block needs rebalancing (no abstention example, or abstention at the
evaluation's real base rate) and the affected configurations re-run. The 0-shot
numbers are unaffected.

The second thing that looked odd — SmolLM2-360M 0-shot at extraction 100% but
overall 68.3% — is not an anomaly. Extraction is 15 of 60 items; that row fails
abstention 0/7 (it hallucinates rather than refusing), calculation 1/6 and
wording 12/17. The overall figure is just the weighted mix. *(measured)*

## 4. Calculation is weak everywhere, exactly as pre-registered

Best calculation score in the table is 33.3% (2/6, CI [9.7, 70.0]). Six
configurations score 0%. This was pre-registered as the expected outcome: the
base model is not supposed to do arithmetic. Phase 4's pipeline extracts the
operands and computes in Python, and `tests/test_tool_pipeline.py` already proves
the arithmetic cannot come from the model (33/33 passing). Calculation is the one
gate where these numbers predict nothing about the shipped system. *(inferred —
the pipeline has not yet been scored on this split)*

## 5. Against Aivora

Aivora's figures are **frozen-split** and therefore *not* item-comparable with
the table above — different items, so no paired test is valid and none is run
here. Treated only as a scale reference:

| | copy | extraction | invented values |
|---|---|---|---|
| Aivora baseline (frozen) | 13.0% | 1.4% | 42.9% |
| Aivora SFT_002 (frozen) | 34.8% | 13.0% | 36.8% |
| Qwen2.5-1.5B 3-shot (selection) | 100.0% | 100.0% | 0.0% |

The pre-registered expectation in EXPERIMENT_RULES_v2 §1 was that the 101.7M
from-scratch model would score below 15% on extraction and invent values on more
than 25% of answerable items, while a pre-trained instruct model would exceed 60%
extraction. Both halves hold: Aivora 1.4% and 13.0% extraction with 42.9% and
36.8% invented values; the 1.5B model 100% extraction with 0% invented. The gap
is roughly two orders of magnitude on extraction, and it is not a tuning gap.
*(measured, with the split caveat above)*

To make this a defensible comparison, Aivora must be scored on this same
selection split so a paired exact McNemar test applies. That has not been done —
the two local attempts were killed at the 30-minute background limit.

## 6. Eval-set integrity — a false alarm, and the fix

The Kaggle run printed `frozen sha256 f55cdda1…`; the manifest recorded
`e02c3dc3…`. **Cause: line endings, not tampering.** `core.autocrlf=true`, so the
Windows checkout holds 313 CR bytes that Git strips on commit; Linux checks the
same file out with LF. Proven: stripping CR from the local file gives exactly
`f55cdda1…`, and canonicalising both to sorted parsed items gives `0e50dbfd…` on
each side — all 313 items identical, and identical to the freeze commit
`2c8673b`. *(measured)*

The byte hash was a bad integrity check: it was guaranteed to disagree on any
cross-platform run, so it both raised a false alarm and could never have caught a
real change. Fixed:

- `content_sha256()` in `scripts/eval_heldout.py` hashes the **parsed items in
  canonical order**, so it is line-ending independent and still changes if any
  item changes.
- `verify_split()` now runs on **every scored run** and raises `SystemExit`
  rather than scoring a set that does not match the manifest.
- The manifest records `content_sha256` for both splits, **computed from the
  freeze commit and verified equal to the working tree** — not back-filled from
  whatever happened to be on disk. The byte hash is kept, labelled platform
  dependent.
- Verified three ways: passes on the real set; refuses a copy with one answer
  changed; gives the same hash for LF and CRLF versions of the same set.

No earlier result is invalidated. The set scored on Kaggle was the frozen set.

## 7. What this does and does not establish

**Establishes** *(measured)*: a like-for-like ranking of five open instruct
models on identical items, prompts, decoding and scorer, with the hardware
control passing. Qwen2.5-1.5B-Instruct leads at 90.0% overall with 100% copy,
100% extraction and 0% invented values. Non-arithmetic extraction is already well
above the 60% interim bar, 0-shot and 3-shot.

**Does not establish**: a result. Selection-split numbers under an unvalidated
scorer, per-gate n of 6–17, wide intervals. Nothing here may be quoted as
Aivora-vs-base-model on matched items, and nothing here says what the Phase 4
pipeline will score.

## 8. Stopping here for your review, as instructed

Blocking on you:

1. **The threshold numbers.** EXPERIMENT_RULES_v2 §1 still contains the literal
   placeholder `<your numbers>`. I have deliberately not filled it in — those
   thresholds are yours to set, and ship/no-ship in Phase 4 depends on them.
2. **The 30 hand labels.** `reports/scorer_validation/sample_for_labelling.jsonl`
   holds 29 sampled outputs with the scorer's verdict hidden. Until κ ≥ 0.8 is
   demonstrated, every number in this report stays provisional.

My recommendation when you return: take **Qwen2.5-1.5B-Instruct 0-shot** into
Phase 4, not the 3-shot variant. They are within one wording item of each other
(85.0% vs 90.0%, overlapping intervals), 0-shot avoids the artifact in §3
entirely, and the pipeline supplies its own structured prompt, so the few-shot
block would be redundant. Running it 0-shot also removes my prompt from the
critical path.

Not started, pending your review: Phase 4 scoring, the paired Aivora comparison
on this split, and any frozen-set run.

### Still outstanding from earlier sessions

- **Three Hugging Face tokens were pasted into an earlier session and should be
  revoked** at https://huggingface.co/settings/tokens. Nothing here uses them.
- `reports/heldout/aivora_sft_003_frozen.json` and the two `*_rescored_v2.json`
  files were created outside my runs. I have not relied on them and cannot
  account for how they were produced.
