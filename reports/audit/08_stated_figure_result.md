# Reading a stated figure instead of recomputing it — measured

Frozen split, n=313 / 297 scored. Same model, same scorer, same decoding.

| Gate | Before | After | Delta |
|---|---|---|---|
| copy | 68/69 | 68/69 | 0 |
| **extraction** | 60/69 | **67/69** | **+7** |
| **wording** | 64/75 | **72/75** | **+8** |
| calculation | 48/48 | 48/48 | 0 |
| abstention | 35/36 | 35/36 | 0 |
| **OVERALL** | **275/297 = 92.59%** | **290/297 = 97.64%** | **+15** |

CI [95.2, 98.8]. Invented values 0/261. Span rule 100% of 273 stated values,
zero violations. Over-refusal unchanged at 2/16 — both of those are the
mislabelled quick-ratio controls from Task 2.

## The change is confined to what it was meant to touch

Exact McNemar on all 313 items: **15 newly correct, 0 newly wrong**, p = 0.0001.
Extraction p = 0.0156, wording p = 0.0078. Calculation, copy and abstention have
**zero discordant pairs** — not "no significant difference", but no item changed
at all.

Every one of the 15 moved from the calculation path to the extraction path:

| Item | Before | After | Answer |
|---|---|---|---|
| extraction_002 | calculation | extraction | `9,560.00` |
| extraction_003 | calculation | extraction | `38.33%` |
| extraction_021 | calculation | extraction | `1.42` |
| extraction_022 | calculation | extraction | `1.37` |
| extraction_023 | calculation | extraction | `4,600.00` |
| extraction_025 | calculation | extraction | `5,214.32` |
| … 9 more | calculation | extraction | |

Component counts confirm it: calculation 101 → 86, extraction 210 → 225. Exactly
15 items, and the 48 genuine calculations are untouched.

## What this does to the Task 4 comparison

| | Model alone | Pipeline before | Pipeline after |
|---|---|---|---|
| extraction | 68/69 = 98.5% | 60/69 = 87.0% | **67/69 = 97.1%** |
| overall | 249/297 = 83.84% | 275/297 = 92.59% | **290/297 = 97.64%** |
| invented values | 0.77% | 0.00% | **0.00%** |

The audit's finding was that the pipeline *cost* 11.6 points of extraction
against the unaided model. That deficit is now **1.4 points** — a single item —
while the pipeline's overall advantage grows from +8.75 to **+13.80 points**,
with span validation and Python-only arithmetic intact throughout.

The earlier recommendation — route extraction around the pipeline — would have
achieved less and cost the guarantee. The deficit was never an extraction
weakness; it was questions being computed that should have been read.

## What remains

Seven misses: two extraction refusals, two wording wrong-fields, one wording
refusal, one copy ordinal miscount ("the fourth item"), one abstention item
answered that should have been refused. The two wording wrong-fields are the
goodwill and cash-generated-from-operations cases from Task 3, where a valid
span cited the wrong line — the failure mode worth watching, and unchanged by
this work.
