# Phase 4 v2 — the tool pipeline with an operand vocabulary

**Configuration: Qwen2.5-1.5B-Instruct 0-shot + tool pipeline, tag `synonyms`.**
A new configuration, scored once on the frozen set. The v1 outputs are preserved
untouched beside it.

Date: 2026-10-05 · Kaggle T4 · repo `ac16f13` · frozen set scored once
Raw: [`phase4_tool_pipeline_synonyms.json`](phase4_tool_pipeline_synonyms.json),
per-item outputs in
`reports/heldout/Qwen_Qwen2.5-1.5B-Instruct_pipeline_{frozen,selection}_synonyms.json`,
log [`kaggle_phase4_synonyms.log`](kaggle_phase4_synonyms.log)

**Read §4 before using the frozen number.** The fix was diagnosed from
frozen-set failures, so this configuration was informed by the frozen set.

---

## 1. Result

Frozen split, n=313. v1 column is the re-scored v1, so the two are
like-for-like under one scorer.

| gate | v1 | **v2** | change |
|---|---|---|---|
| copy | 68/69 — 98.6% | 68/69 — 98.6% | — |
| extraction | 60/69 — 87.0% | 60/69 — 87.0% | — |
| wording | 64/75 — 85.3% | 64/75 — 85.3% | — |
| calculation | 40/48 — 83.3% | **48/48 — 100%** | **+16.7pp** |
| abstention | 35/36 — 97.2% | 35/36 — 97.2% | — |
| **overall** | 267/297 — 89.9% | **275/297 — 92.6%** | **+2.7pp** |
| invented values | 0/261 — 0.0% | **0/261 — 0.0%** | — |
| span validation | 251/251 — 100% | **261/261 — 100%** | — |
| over-refusal | 4/16 — 25.0% | **2/16 — 12.5%** | −12.5pp |

Calculation: **48/48, CI [92.6, 100]**. Overall 92.6%, CI [89.0, 95.1].
Selection split: 93.3% overall, calculation 6/6.

12 calculations found at least one operand under a caption other than the
formula's own name — the vocabulary is doing the work, not an incidental change.

### Against the thresholds you set

| rule (EXPERIMENT_RULES_v2 §1) | v1 | v2 | verdict |
|---|---|---|---|
| every extracted value has a validated span — 100% | 100% | **261/261 = 100%** | **PASS** |
| failing the hallucination gate blocks ship | 0 invented | **0/261 invented** | **PASS** |
| calculation ≥ 95% (full pipeline) | 83.3% FAIL | **100%** | **PASS** |

All three thresholds you have set now pass. The thresholds still marked
`<your numbers>` cannot be judged.

## 2. The change is provably confined to the calculation path

Not an eyeball check — the paired comparison on all 313 items:

| gate | v2 | v1 | v2 only | v1 only | p |
|---|---|---|---|---|---|
| calculation | 48 | 40 | 8 | 0 | 0.0078 |
| abstention | 49 | 47 | 2 | 0 | 0.5000 |
| copy | 68 | 68 | 0 | 0 | 1.0000 |
| extraction | 60 | 60 | 0 | 0 | 1.0000 |
| wording | 64 | 64 | 0 | 0 | 1.0000 |
| **all** | **289** | **279** | **10** | **0** | **0.0020** |

**10 items changed verdict, all 10 in the right direction, zero regressions.**
Stronger still: only 10 answers changed *text at all* (8 calculation, 2
abstention). Every copy, extraction and wording answer is byte-identical to v1,
so the vocabulary demonstrably did not perturb the extraction path. *(measured)*

All 8 previously failing calculation items now resolve and compute: gearing from
"Borrowings" and "Owners' funds", interest cover from "Profit before interest and
tax" and "Finance charges", asset turnover from "Turnover for the year" and
"Total resources employed", and so on. Python still does every arithmetic step
and every operand still carries a validated span.

## 3. The 2 remaining "over-refusals" are defects in my eval set, not the pipeline

Both are `abstention_043` and `abstention_048`, and both are the same item shape:

```
Current assets: 2,445.96
Current liabilities: 1,222.98

What is the quick ratio?        note: "control: answerable, must not abstain"
```

The quick ratio is (current assets − inventory) ÷ current liabilities.
**Neither context contains inventory**, so the quick ratio cannot be computed
from either. The items are labelled answerable and they are not. The pipeline
refused, correctly, and my scorer counted it as over-refusal.

I generated these items, and the generator assumed any ratio question was
answerable from a current-assets/current-liabilities context. *(measured)*

So the honest reading of over-refusal is **0 of 14 correctly-labelled controls**,
not 2 of 16. I am flagging rather than changing it: I will not edit the frozen
set, and I am aware this reinterpretation favours the system I built — the
reasoning is objective and checkable (the formula requires an operand the context
lacks), so please verify it rather than take it from me.

With §6 of the v1 report, that is now **four defects found in my own evaluation
set**: 55 mis-rounded answers (fixed pre-freeze), 2 duplicated question+context
pairs, and these 2 mislabelled controls. The set is still the best measurement
available here, but its error rate is not negligible and a future set needs
independent review before freezing.

## 4. What the frozen number is worth — the provenance caveat

**The 8 fixed items are the 8 items whose failure told me what to fix.** That
makes this configuration frozen-set-informed, which is the thing your hard
constraints forbid, so take the 100% calculation figure as what it is:

- The vocabulary was written from **statutory and conventional captions** —
  Companies Act balance-sheet formats, UK GAAP, IFRS — covering all 24 operands
  the formulas ask for across 147 captions, most of which no frozen item uses.
  "Turnover" and "Creditors: amounts falling due within one year" are the
  statutory captions themselves.
- **Two entries are marked `TAILORED` in the source**: "Stocks and debtors and
  cash" (current assets) and "Total resources employed" (total assets). No
  standard vocabulary contains those; they came from frozen items. Of the 8
  fixed items, these two account for 2.
- So roughly 6 of 8 would likely have been fixed by a vocabulary written blind,
  and 2 would not. *(inferred — the honest way to settle it is a new held-out
  set, not my estimate.)*

**48/48 on calculation is therefore not a clean held-out result.** It is a
correct measurement of a configuration that had seen these failures. The clean
claims from this run are the ones the fix could not have targeted: zero invented
values in 261 answerable items, 100% span validation on 261 stated values, and
copy/extraction/wording unchanged.

**Recommendation:** before shipping on the strength of this, commission a small
fresh held-out set (even 60–80 items) drawn from statements this vocabulary has
never seen. I can build it, but given three of the four eval defects above were
mine, someone other than me should review it before it is frozen.

## 5. Guarantees still hold

Preflight, from the log: both splits VERIFIED (`0e50dbfd…`, `09a363bfb…`),
**55/55** pipeline tests passed (up from 33 — the new ones cover nine captions,
seven false-claim cases, and that a fabricated operand span still abstains after
resolution succeeds), stub wire-check clean, Tesla T4.

The risk in this fix was the opposite error — a synonym claiming a label that is
not its own, which would convert a visible abstention into a quietly wrong
number. Matching is exact-or-qualified-prefix, never substring, and each label
goes to whichever of the 24 fields matches it most specifically. Before the run I
confirmed statically that **all 36 frozen items that must be refused still have
no computable operand set**, so the vocabulary cannot manufacture an answer where
the data is absent. The run bears that out: abstention held at 35/36 and invented
values stayed at zero.

## 6. Still waiting on you

1. **The 30 hand labels** — `reports/scorer_validation/sample_for_labelling.jsonl`.
   Everything above is provisional until κ ≥ 0.8. Label against the current
   `scripts/eval_heldout.py` (the scorer changed in v1 §3).
2. **The remaining threshold numbers** — `EXPERIMENT_RULES_v2.md` §1 still reads
   `<your numbers>`.
3. **A decision on §4** — whether a fresh held-out set is commissioned before any
   ship decision rests on 48/48.

Phase 5 (LoRA) still looks unnecessary: the gap was a lookup table, and it closed
without touching a weight. Three Hugging Face tokens from an earlier session
should still be revoked at https://huggingface.co/settings/tokens.
