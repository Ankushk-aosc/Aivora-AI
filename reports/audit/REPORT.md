# Audit report — Aivora ("Nivora") v1.1

Branch `audit/nivora-v1.1`. Every figure comes from a run whose raw output is
saved under `reports/`. Where something could not be measured it says NOT
MEASURED and why.

---

# ONE PAGE

## The finding that matters most

**The Comparisons view is serving fabricated prior-year figures.** Each FY2024
line is FY2025's margin applied to FY2024's revenue — exactly, to the rupee —
and none of them is in the filing, which states one FY2024 number: revenue.

| Served as FY2024 | Value | In the filing? |
|---|---|---|
| Revenue | 8,000,000 | yes |
| COGS | 5,600,000 | **no** (= 70.0% × 8M) |
| Gross profit | 2,400,000 | **no** (= 30.0% × 8M) |
| Operating expenses | 1,680,000 | **no** (= 21.0% × 8M) |
| Operating profit | 720,000 | **no** (= 9.0% × 8M) |

The view consequently shows identical margins in both years, which reads as a
business finding and is an artefact of the derivation. `data/demo_company.json`
is untracked, so the change has no commit, author or message. Not reverted — it
looks deliberate — but **it should not ship in this state.**

## 1. Test status

| | Before | After |
|---|---|---|
| test_analyst_pipeline | 28/28 | **35/35** |
| test_tool_pipeline | 55/55 | 55/55 |
| test_intent_routing | 80/80 | 80/80 |
| test_benchmark | 47/47 | 47/47 |
| test_response_pipeline | 26/26 | 26/26 |
| test_api_security | 25/25 | 25/25 |
| test_scorer_regressions | 23/23 | 23/23 |
| test_training_format | 23/23 | 23/23 |
| test_numbers_in | 19/19 | 19/19 |
| test_knowledge_retrieval | 11/11 | 11/11 |
| test_aivora_system | OK | OK |
| **test_product_views** | **FAIL** | **FAIL** |
| **test_alerts_and_analyses** | **53/58** | **53/58** |
| verify_frontend_clean | FAIL (no server) | passes with a server running |

The two failures **pre-date this audit** and come from on-disk edits made
outside the session. Nothing the audit changed broke a test.

## 2. Reproduced result (Task 0)

**275/297 = 92.59%, CI [89.04, 95.06] — matches exactly.** Per gate: copy 68/69,
extraction 60/69, wording 64/75, calculation 48/48, abstention 35/36, invented
values 0/261.

Reproduced by re-scoring preserved raw outputs. Re-*generating* them is **NOT
MEASURED**: the figure came from Qwen2.5-1.5B on a Kaggle T4 and this machine has
~1 GB free against the ~7 GB required.

## 3. The 16 unscored items (Task 2)

313 − 16 = 297, exactly. All 16 are **answerable abstention controls**
(`expected: False`), excluded from the overall figure by design and reported as
over-refusal instead; counting them in the abstention gate would give a
never-refusing system a free 16/16 on a metric meant to measure refusal.

Two of the 16 are themselves mislabelled — `abstention_043` and `abstention_048`
ask for the quick ratio from a context with no inventory, which the formula
requires. Correctly labelled, over-refusal is **0 of 14**, not 2 of 16.

## 4. Miss classification (Task 3)

| Gate | abstained | wrong_value_valid_span | wrong_value_invalid_span | other |
|---|---|---|---|---|
| extraction (9) | 6 | 3 | **0** | 0 |
| wording (11) | 9 | 2 | **0** | 0 |
| copy (1) | 0 | 1 | **0** | 0 |
| abstention (1) | 0 | 1 | **0** | 0 |
| abstention controls (2) | 2 | 0 | **0** | 0 |

**Zero `wrong_value_invalid_span`.** The span rule holds under an audit that
re-checks every cited span against its context independently of the pipeline's
own validator.

**The dominant failure mode is refusal, not error**: 15 of 24 misses are
abstentions.

**Three of the seven wrong values are not wrong.** `extraction_002`, `_003` and
`_023` return exactly the expected value and are scored
`ambiguous_multiple_values` because the pipeline prints its working beside it —
`9,560.00 (total assets = 15,800.00, total liabilities = 6,240.00)` against an
expected `9,560.00`. The value alone scores correct.

> Extraction would be **63/69 (91.3%)** and overall **278/297 (93.60%)** if
> showing working were not penalised. The scorer is **not** changed — that would
> re-score history. The decision is yours.

The four genuine errors: a wrong label picked for goodwill (`Other intangibles`)
and for cash generated from operations (`Other revenue`); an ordinal miscount on
"the fourth item"; and an inventory figure returned for an inventory-turnover
question that should have been refused.

## 5. Baseline without the pipeline (Task 4) — MEASURED

| Gate | Model alone | + pipeline | Delta |
|---|---|---|---|
| copy | 97.1% | 98.6% | +1.5 |
| **extraction** | **98.5%** | **87.0%** | **−11.6 (−8 items)** |
| **wording** | **88.0%** | **85.3%** | **−2.7 (−2 items)** |
| calculation | 52.1% | 100.0% | **+47.9 (+23)** |
| abstention | 63.9% | 97.2% | **+33.3 (+12)** |
| **OVERALL** | **249/297 = 83.84%** | **275/297 = 92.59%** | **+8.75** |
| invented values | 0.77% (2/261) | 0.00% | −0.77 |

**The pipeline makes extraction worse.** The same model unaided scores 98.5%
where the pipeline scores 87.0%. The documentation presents 87.0% as the
pipeline's extraction capability and never states the comparison.

The whole +8.75 comes from calculation (+23 items) and abstention (+12), partly
spent paying for the losses. **The pipeline is a trade, not a general
improvement**: it buys arithmetic (the plain model gets half of all calculation
wrong) and refusal (it answers 13 of 36 questions it should decline), at the
cost of refusing eight extraction items the model would have got right.

Recommended, not implemented: route extraction-only questions around the
pipeline, keep it for calculation and refusal — roughly 283/297 on these
numbers, guarantees intact.

## 6. Leakage verdict (Task 5)

**CONTAMINATED — and by my own hand.**

| Date | Commit | Event |
|---|---|---|
| 2026-09-30 | `2c8673b` | held-out set frozen |
| 2026-10-04 | `3166974` | `FORMULAS` added — **before** any frozen scoring |
| 2026-10-05 | `7ae5a97` | frozen scored: calculation 83.3%, 8 failures inspected |
| 2026-10-05 | `ac16f13` | **`SYNONYMS` — all 147 captions — added after inspecting those failures** |
| 2026-10-05 | `30b63c8` | frozen re-scored: calculation **100%** |

Two captions remain marked `TAILORED` in the source because they came from
frozen items. **The 100% calculation figure is not a clean held-out result.**

**The calculation capability is the caption table, not the model.** The same
capability, measured three ways:

| Measurement | Score |
|---|---|
| Frozen set — captions written *after* seeing its failures | 48/48 = **100.0%** |
| Fresh contexts, captions the vocabulary **has** | 44/44 = **100.0%** |
| Fresh contexts, captions the vocabulary **lacks** | **5/44 = 11.4%** |

When the table knows the wording the pipeline is perfect; when it does not, the
model bridges the gap 5 times in 44. The 147 captions carry essentially the
whole result, and the earlier 83.3% → 100% "capability improvement" was coverage.

**The failure is the safe one.** Zero wrong values, zero invented values, zero
span violations — 39 of 39 failures are refusals. Faced with unknown wording the
system declines rather than picking a plausible-looking wrong line. Task 3 found
two wrong-line cases on the frozen set, so this was a real possibility.

A real filing will use wording outside the table, so it will meet refusals, not
errors. The honest statement is: **calculation is 100% on wordings the
vocabulary covers and refuses most of what it does not. Coverage, not
arithmetic, is the limit.** Calculation should be reported as coverage ×
accuracy-within-coverage, with coverage measured against real filings.

**The leakage did not inflate the result.** The capability reproduces exactly on
items the pipeline has never seen, with expected answers computed independently
of it.

The caveat: the fresh split tests new instances of captions the vocabulary
*already covers* (44 of 44). It closes "did the vocabulary overfit to those eight
frozen items" — it did not. It does **not** close "does the vocabulary generalise
to captions nobody wrote into it". A split built from captions outside the table
would answer that, and would be expected to score lower.

## 7. Real filing (Task 6)

**SKIPPED: no filing supplied.** `data/real_filings/` created with a README.
No synthetic substitute — a generated filing cannot test whether the system
reads a real one.

This is the most important missing measurement in the project. Every number
reported anywhere comes from generated contexts of the form `Label: value`. Real
filings have tables, footnotes, multi-column layouts and scales declared far
from the figure. None of that is represented, so none of the accuracy reported
here transfers to a real filing without this test.

## 8. Documented vs actual counts (Task 7)

| Claim | Documented | Actual | |
|---|---|---|---|
| Endpoints | "roughly sixty" | **99 routes, 82 paths** | **WRONG** |
| Alert metrics | 13 | 13 | ok |
| Formulas | 21 | 21 | ok |
| Operations | 8 | 8 | ok |
| Operand fields | 24 | 24 | ok |
| Captions | 147 | 147 | ok |

"No external service dependency" is **true in the default configuration, false
as an unqualified claim**: enabling the Analyst pipeline downloads ~3 GB from
the Hugging Face Hub on first use.

`grep -ril nivora` → **0 files**. The product is Aivora throughout. No rename
performed: it would touch 145 files including a filename referenced inside
frozen eval items, changing the frozen set's content hash, which ground rule 2
forbids.

## 9. Files changed

| Commit | Files |
|---|---|
| `6dd9c6c` | `reports/audit/00_orientation.md`, `02_summary.md`, `02_unscored_items.csv`, `tests_before.txt` |
| `35425da` | `pipeline/tool_pipeline.py`, `scripts/build_fresh_calculation_split.py`, `data/eval_fresh/*`, `reports/audit/03_misses.csv`, `ASSUMPTIONS.md` |
| `e91fb3e` | `data/real_filings/README.md`, `reports/audit/07_counts.md`, `07_dependencies.md` |
| `696043d` | `app/backend/server.py`, `app/backend/services/analyst_pipeline.py`, `tests/test_analyst_pipeline.py`, `reports/audit/tests_after.txt` |
| `be68eeb` | `scripts/run_tool_pipeline_eval.py` (`--items-file`), `scripts/build_fresh_calculation_split.py`, `data/eval_fresh/*` |
| `762e21a` | `reports/audit/04_05_results.md`, `audit_gpu_results.json` |
| `4b9ee82` | Task 5b: `scripts/build_uncovered_caption_split.py`, `data/eval_fresh/calculation_uncovered.jsonl` |

### Two defects fixed in the code

1. **`context_labels()` split each line on the first colon**, so the Companies
   Act caption `Creditors: amounts falling due within one year` yielded the
   label `Creditors` and its value was unreachable — despite the full caption
   being in the vocabulary. Now splits on the last colon when a number follows.
2. **Task 1**: the pipeline default, the refusal path, and three defects in my
   own change found by verifying live rather than trusting the unit tests — the
   HTTP layer dropped the remedy, the status endpoint claimed the legacy engine
   was serving while every question was refused, and the refusal carried a badge
   saying the value was not in the source when nothing had been looked for.

## 10. Not measured, and why

| Item | Why |
|---|---|
| End-to-end regeneration of 275/297 | ~7 GB model, ~1 GB free locally. Task 4 did re-run the model on these items unaided, so the hardware path is proven; the pipeline configuration was not re-generated. |
| Task 6 real filing | no filing supplied |
| Whether the routing fix prevents full MoE collapse | 4,000-step pilot against 247,850; onset reproduced, endpoint extrapolated |

Tasks 4, 5 and 5b are now measured. The remaining gaps need one file from you
(a real filing) and a longer training run.

---

Detail: `00_orientation.md` · `02_summary.md` · `02_unscored_items.csv` ·
`03_misses.csv` · `07_counts.md` · `07_dependencies.md` · `ASSUMPTIONS.md`
