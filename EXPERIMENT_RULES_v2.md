# Experiment rules v2 - 2026-09-30

Supersedes `EXPERIMENT_RULES.md`, which is **left untouched** as the record of
what was pre-registered on 2026-09-30 before any system was scored.

**Reason for v2:** the thresholds in v1 (commit `95fa38d`) were chosen by Claude
without the project owner. Thresholds decide what ships, so they are the owner's
to set. v2 records the owner's instructions and marks clearly which numbers are
still theirs to give.

No system has been scored on the frozen set under v1 thresholds, so nothing is
invalidated by this change.

---

## 1. Threshold status

### 1a. Set by the owner - authoritative

| rule | value | status |
| --- | --- | --- |
| **source span validation** | **100%** of extracted values must carry a span that is verified present in the context | owner-set, binding |
| **hallucination gate** | a system that fails it is **not shipped**, whatever else it scores | owner-set, binding |
| **calculation, full pipeline** | **>= 95%** | owner-set, binding |

### 1b. AWAITING THE OWNER'S NUMBERS

The instruction read "I am setting them: `<your numbers>`" and the numbers did
not come through. Rather than invent values and attribute them to the owner, the
v1 numbers are carried below **as interim placeholders with no authority**. They
are used for nothing except keeping the pipeline runnable, and **no ship / no-ship
decision may be made against them**.

| gate | interim placeholder (from v1, NOT owner-set) |
| --- | --- |
| extraction | 60% |
| wording generalisation | 50% |
| copy | 80% |
| abstention | 70% correct refusals |
| over-refusal | <= 15% |
| hallucination (invented values) | <= 10% |
| calculation, model alone (no tools) | 80% |

Phase 3 is measurement, not a ship decision, so it proceeds under this
uncertainty. **Phase 4 onward requires the owner's numbers**, since that is where
"does the pipeline meet the bar" is asked.

## 2. Statistical tests

Changed on the owner's instruction:

* **Comparing two systems on the same items**: exact **McNemar test** on the
  discordant pairs (b, c), two-sided, reported with the counts. This replaces the
  two-proportion and Fisher tests used in v1, which treat paired observations as
  independent and therefore overstate the uncertainty when both systems answer
  the same questions.
* **A single system's score**: 95% **Wilson interval**, unchanged from v1.
* A difference is called real only at **p < 0.05**. Otherwise it is reported as
  "not distinguishable at this sample size".
* Pooling across gates must be declared, with per-gate numbers still shown.
* No capability claim from training loss. Unchanged.

### Note on power at these sample sizes

With 69 extraction items, McNemar needs roughly 10 or more discordant pairs to
detect a moderate difference at p < 0.05. Where the test is underpowered, the
report says so with the discordant counts, rather than reporting a null result
as evidence of equivalence.

## 3. Pre-registered Aivora expectations - with numbers

v1 said "near zero" and "materially higher", which are not falsifiable. Restated
as explicit intervals, derived from each system's measured gate results
(`reports/gates_baseline.json`, `reports/gates_sft_002.json`) and the frozen
set's per-gate sizes.

**Aivora frozen baseline (`checkpoint_247850`)**, measured gates: copy 5/6,
extraction 0/20, generalisation 0/15, calculation 0/20.

| gate | n (frozen) | predicted k | predicted accuracy |
| --- | --- | --- | --- |
| copy | 69 | 28 - 62 | 40% - 90% |
| extraction | 69 | 0 - 7 | 0% - 10% |
| wording | 75 | 0 - 8 | 0% - 10% |
| calculation | 48 | 0 - 5 | 0% - 10% |
| abstention | 52 | 0 - 10 | 0% - 20% |

**Aivora SFT_002**, measured gates: copy 4/6, extraction 2/20, generalisation
3/15, calculation 1/20, abstention 1/5.

| gate | n (frozen) | predicted k | predicted accuracy |
| --- | --- | --- | --- |
| copy | 69 | 21 - 55 | 30% - 80% |
| extraction | 69 | 0 - 17 | 0% - 25% |
| wording | 75 | 0 - 26 | 0% - 35% |
| calculation | 48 | 0 - 10 | 0% - 20% |
| abstention | 52 | 0 - 21 | 0% - 40% |

**Falsification rule:** if either system scores **above the upper bound** of any
interval above, the scorer is treated as suspect. Work stops, the scorer is
audited against hand labels, and no other result is trusted until the
discrepancy is explained. A score **below** the lower bound on copy is equally a
signal - it would mean the frozen set's copy items are harder than the probes,
which changes how every later copy number is read.

## 4. Scorer validation - owner hand-labels 30 outputs

Before Phase 3 results are accepted:

* 30 outputs are sampled across gates and written to
  `reports/scorer_validation/sample_for_labelling.jsonl`, with the scorer's own
  verdict **hidden**.
* The owner labels each correct or incorrect.
* Agreement is reported as raw agreement and **Cohen's kappa**, with every
  disagreement listed in full.
* **Acceptance: kappa >= 0.8 and no systematic disagreement in one category.**
  Below that, the scorer is fixed and the Phase 3 numbers are recomputed before
  anything is concluded from them.

## 5. Prompt finalisation and single-shot frozen scoring

* Each model's prompt is finalised on the **selection split only**. Any number
  produced during prompt iteration is selection-split, labelled as such, and
  never quoted as a result.
* Once finalised, each configuration is scored on the frozen set **exactly
  once**. If a configuration is scored twice, both runs are reported with the
  reason for the second.
* Raw outputs are saved for every run, passing or failing, under
  `reports/heldout/`.

## 6. Unchanged from v1

Sections 1 (hypothesis), 2 (one variable per experiment), 3 (decoding and
prompting), 5 (stop rules), 7 (what counts as a correct answer) and 8 (integrity
rules) of `EXPERIMENT_RULES.md` carry over unchanged, except where section 1a
above makes a rule stricter.

Source-span validation at 100% is new and stricter than anything in v1: a value
whose span cannot be located in the context is rejected and becomes an
abstention, and a pipeline that cannot produce spans cannot pass Phase 4.
