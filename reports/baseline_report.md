# Aivora v0.1-base - measured baseline

Benchmark: `data/benchmark/dev.jsonl`, 795 items. Produced by `scripts/baseline_report.py`; every number below is measured, not estimated.

* checkpoint `checkpoint_247850.pt`, sha256 `6e9e50ca59e5...`
* 101,723,264 parameters, CPU, torch 2.13.0+cpu
* sampling {'temperature': 0.7, 'top_k': 40, 'top_p': 0.9, 'repetition_penalty': 1.3, 'max_new_tokens': 40}, torch re-seeded to 1234 per question
* git revision `343e01f62a34`

The hidden test split was not used.

## Headline

| configuration | accuracy | hallucination rate | over-abstention | abstention accuracy | s/question |
| --- | --- | --- | --- | --- | --- |
| model_only | **0.88%** (7/795) | 98.42% | 0.66% | 0.0% | 6.786 |
| plus_calculator | **54.21%** (431/795) | 31.88% | 15.28% | 83.33% | 0.992 |
| plus_retrieval | **4.91%** (39/795) | 94.2% | 0.66% | 0.0% | 6.806 |
| full_pipeline | **58.36%** (464/795) | 27.67% | 15.28% | 86.11% | 0.102 |

*hallucination rate* = asserted a wrong answer to a question that had one. *over-abstention* = declined a question that had an answer. *abstention accuracy* = correctly declined the unanswerable items. A system can only look good on all three at once by actually knowing which questions it can answer.

## By category

| category | model_only | plus_calculator | plus_retrieval | full_pipeline |
| --- | --- | --- | --- | --- |
| accounting | 2.5% (1/40) | 2.5% (1/40) | 22.5% (9/40) | 22.5% (9/40) |
| concepts | 4.35% (1/23) | 4.35% (1/23) | 82.61% (19/23) | 82.61% (19/23) |
| corporate_finance | 0.0% (0/97) | 50.52% (49/97) | 0.0% (0/97) | 50.52% (49/97) |
| extraction | 0.0% (0/100) | 23.0% (23/100) | 0.0% (0/100) | 23.0% (23/100) |
| general | 12.5% (1/8) | 12.5% (1/8) | 62.5% (5/8) | 62.5% (5/8) |
| hallucination | 0.0% (0/36) | 83.33% (30/36) | 0.0% (0/36) | 86.11% (31/36) |
| interpretation | 16.67% (2/12) | 16.67% (2/12) | 25.0% (3/12) | 25.0% (3/12) |
| ratios | 0.0% (0/140) | 70.71% (99/140) | 0.0% (0/140) | 70.71% (99/140) |
| reasoning | 0.0% (0/109) | 100.0% (109/109) | 0.0% (0/109) | 100.0% (109/109) |
| reporting | 7.69% (1/13) | 7.69% (1/13) | 23.08% (3/13) | 23.08% (3/13) |
| statements | 0.0% (0/117) | 63.25% (74/117) | 0.0% (0/117) | 63.25% (74/117) |
| valuation | 1.0% (1/100) | 41.0% (41/100) | 0.0% (0/100) | 40.0% (40/100) |

## By difficulty level

| level | model_only | plus_calculator | plus_retrieval | full_pipeline |
| --- | --- | --- | --- | --- |
| level 1 | 0.0% | 20.18% | 8.77% | 28.95% |
| level 2 | 1.79% | 56.7% | 10.27% | 65.18% |
| level 3 | 0.36% | 50.54% | 1.08% | 51.26% |
| level 4 | 0.0% | 44.78% | 1.49% | 47.76% |
| level 5 | 1.77% | 98.23% | 1.77% | 98.23% |

## By item kind

| kind | model_only | plus_calculator | plus_retrieval | full_pipeline |
| --- | --- | --- | --- | --- |
| authored | 8.22% (6/73) | 8.22% (6/73) | 53.42% (39/73) | 53.42% (39/73) |
| generated | 0.14% (1/722) | 58.86% (425/722) | 0.0% (0/722) | 58.86% (425/722) |

## What each component is worth

* **plus_calculator**: +53.33 points against model_only (model + deterministic calculator and derived identities)
* **plus_retrieval**: +4.03 points against model_only (model + curated glossary and retrieval layer)
* **full_pipeline**: +57.48 points against model_only (the served pipeline: calculator + knowledge + quality guard)

## Sample failures

* `dev_extract_0041` (extraction, hallucination): 'Answer: EBITDA Margin\nFormula / Breakdown: EBITDA / Revenue x 100\nInputs: ebitda = 695.00, revenue = 8,075.00\nResult: 8.61%'
* `dev_stmt_0028` (accounting, hallucination): 'Answer: Income Statement Analysis\nFormula / Breakdown: Gross Profit = 1,202.25 (35.0% margin), Operating Income = 687.00 (20.0% margin), Profit Before Tax (PBT)'
* `dev_extract_0066` (extraction, hallucination): 'Answer: EBITDA Margin\nFormula / Breakdown: EBITDA / Revenue x 100\nInputs: ebitda = 1,655.00, revenue = 8,480.00\nResult: 19.52%'
* `dev_corp_0085` (corporate_finance, over_abstention): "I could not identify a complete calculation from the values provided. Values I parsed: {'net_income': 905.0}. Available calculations: cagr, current_ratio, dcf,"
* `dev_val_0053` (valuation, hallucination): 'CAGR (Compound Annual Growth Rate) represents the annualized rate of return for an investment over a multi-year period. Formula: CAGR = ((Ending Value / Beginni'
* `dev_stmt_0098` (statements, hallucination): 'Answer: Income Statement Analysis\nFormula / Breakdown: Gross Profit = 2,381.75 (35.0% margin), Operating Income = 1,701.25 (25.0% margin), Profit Before Tax (PB'
* `dev_val_0034` (valuation, over_abstention): "I could not identify a complete calculation from the values provided. Values I parsed: {'net_income': 220.0}. Available calculations: cagr, current_ratio, dcf,"
* `dev_corp_0064` (corporate_finance, over_abstention): "I could not identify a complete calculation from the values provided. Values I parsed: {'equity': 70.0}. Available calculations: cagr, current_ratio, dcf, debt_"
