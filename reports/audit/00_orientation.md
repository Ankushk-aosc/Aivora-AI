# Task 0 — orientation, and a finding that cannot wait

Branch `audit/nivora-v1.1`. All figures below come from runs whose raw output is
saved under `reports/`.

---

## CRITICAL FINDING — fabricated prior-year figures are being served

This was found while establishing the test baseline, before any audit task
proper. It is reported first because the comparison view is client-facing.

`data/demo_company.json` now carries a `prior_period` block of FY2024 figures,
and `app/backend/services/product_views.py` has been rewritten to use them, with
the same values hardcoded as fallback defaults in the function body.

**None of those figures exist in the source filing.** The filing states exactly
one FY2024 number:

> "total consolidated revenue of $10,000,000 (AUD), compared to $8,000,000 in
> FY2024"

Checked against `data/documents/Aivora_Enterprise_Client_FY2025_Report.txt`:

| Figure served as FY2024 | Value | In the filing? |
|---|---|---|
| Revenue | 8,000,000 | **yes** — the only FY2024 figure present |
| COGS | 5,600,000 | **absent** |
| Gross profit | 2,400,000 | **absent** as a FY2024 figure. The string appears once, as "Cash and cash equivalents at period end were $2,400,000" — a different quantity in a different year |
| Operating expenses | 1,680,000 | **absent** |
| Operating profit | 720,000 | **absent** |
| Interest 180,000 · Tax 160,000 · Net profit 380,000 | — | **absent** |

They are not arbitrary. Every one is **FY2025's margin applied to FY2024's
revenue**, to the rupee:

| Line | FY2025 margin | × 8,000,000 | Value served |
|---|---|---|---|
| COGS | 70.0% | 5,600,000 | 5,600,000 |
| Gross profit | 30.0% | 2,400,000 | 2,400,000 |
| Operating expenses | 21.0% | 1,680,000 | 1,680,000 |
| Operating profit | 9.0% | 720,000 | 720,000 |

The consequence is that the Comparisons view now shows FY2024 and FY2025 with
*identical* gross margin (30.0%) and operating margin (9.0%). That reads as a
finding about the business. It is an artefact of the derivation.

`data/demo_company.json` is **untracked** — it has never been committed, so the
change has no commit, no author and no message.

This replaced an implementation written specifically to prevent it, which
reported 1 of 9 comparison lines as having a prior-year figure and marked the
other 8 "Not in source" with a reason. The regression is visible in the test
suite: `tests/test_product_views.py` asserts "five lines have no prior-year
figure" and now measures 0.

**Recommendation.** Do not ship the comparison view in this state. Either
restore source-bound behaviour, or — if modelled prior-year figures are wanted —
label them as modelled, state the assumption applied, and never present them in
the same column as audited figures. I have not reverted it: the change is
deliberate and the decision is the owner's.

---

## Test status before the audit (ground rule 3)

Recorded on the branch before any change. The baseline is **not green**, and
the failures pre-date this audit.

| Suite | Result | Cause |
|---|---|---|
| test_product_views | **FAIL** | the fabricated-figures change above |
| test_alerts_and_analyses | **53/58** | `alerts.py` edited on disk (+13 lines) outside this session |
| verify_frontend_clean | **FAIL** | environmental: needs a running server, none started in this session |
| test_analyst_pipeline | 28/28 | |
| test_tool_pipeline | 55/55 | |
| test_intent_routing | 80/80 | |
| test_response_pipeline | 26/26 | |
| test_api_security | 25/25 | |
| test_benchmark | 47/47 | |
| test_scorer_regressions | 23/23 | |
| test_training_format | 23/23 | |
| test_numbers_in | 19/19 | |
| test_knowledge_retrieval | 11/11 | |
| test_aivora_system | OK | |

Raw output: `reports/audit/tests_before.txt`.

---

## How the evaluation runs, end to end

1. **The set.** `data/eval_heldout/heldout_frozen.jsonl`, 313 items across five
   gates, built with 8-gram overlap removal against the training corpus, seed
   424242. Integrity is checked on every scored run by `verify_split()` against
   a content hash over the parsed items in canonical order — platform
   independent, unlike a byte hash.
2. **The scorer.** `scripts/eval_heldout.py` — one scorer for every system, so
   no system is judged by its own standard. `score_item()` holds the
   correctness rules; `evaluate()` produces per-gate k/n with Wilson intervals.
3. **The system under test.** Any `answer_fn(question, context) -> str`.
4. **The pipeline configuration.** `scripts/run_tool_pipeline_eval.py` wraps
   `pipeline/tool_pipeline.ToolPipeline` around a model backend and additionally
   re-audits every stated value's source span independently of the pipeline's
   own validator.

### Which backend each configuration uses

| Configuration | Backend | Where it ran |
|---|---|---|
| Aivora baseline / SFT_001 / SFT_002 / SFT_003 | local DeepSeekV3 checkpoint, 101.7M | CPU, this machine |
| Phase 3 base models | Hugging Face instruct models, 135M–1.7B | Kaggle T4 |
| **Tool pipeline (the 275/297 result)** | **Qwen2.5-1.5B-Instruct, 0-shot, behind the tool pipeline** | **Kaggle T4** |
| Analyst in the running app | local DeepSeekV3, SFT_003 | CPU — the pipeline bridge is gated off, see §11 of the model doc |

## Reproducing 275/297

The documented figure came from Qwen2.5-1.5B-Instruct on a Kaggle T4. That model
needs about 7 GB; this machine has ~1 GB free, so **the model cannot be re-run
here**. What is reproducible locally, and what was run, is a re-score of the
preserved raw outputs — the model ran once, its answers are fixed, and the
scorer is applied to them again:

```
deepseek_env/Scripts/python.exe - <<'EOF'
import json, sys; sys.path.insert(0, ".")
from scripts.eval_heldout import evaluate, FROZEN
payload = json.load(open("reports/heldout/"
    "Qwen_Qwen2.5-1.5B-Instruct_pipeline_frozen_synonyms.json", encoding="utf-8"))
by_id = {r["id"]: r["answer"] for r in payload["records"]}
items = [json.loads(l) for l in open(FROZEN, encoding="utf-8") if l.strip()]
queue = iter([by_id[i["id"]] for i in items])
summary, _ = evaluate(lambda q, c: next(queue), items=items,
                      label="reproduction", split="frozen")
print(summary["overall"])
EOF
```

**Result: 275/297 = 92.59%, CI [89.04, 95.06] — matches the documented figure
exactly.**

| Gate | Reproduced |
|---|---|
| copy | 68/69 — 98.5% |
| extraction | 60/69 — 87.0% |
| wording | 64/75 — 85.3% |
| calculation | 48/48 — 100.0% |
| abstention | 35/36 — 97.2% |
| invented values | 0/261 |
| over-refusal | 2/16 |

To re-run the model end to end rather than re-score it, on a machine with ~8 GB
free:

```
deepseek_env/Scripts/python.exe scripts/run_tool_pipeline_eval.py \
    --model Qwen/Qwen2.5-1.5B-Instruct --split frozen --tag synonyms
```

**Caveat (ground rule 1).** This reproduces the *scoring* of fixed outputs, not
the generation. A full end-to-end reproduction is NOT MEASURED here, because the
hardware cannot host the model.
