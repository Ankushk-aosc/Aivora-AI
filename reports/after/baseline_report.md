# Aivora v0.1-base - measured baseline

Benchmark: `data/benchmark/dev.jsonl`, 795 items. Produced by `scripts/baseline_report.py`; every number below is measured, not estimated.

* checkpoint `checkpoint_247850.pt`, sha256 `6e9e50ca59e5...`
* 101,723,264 parameters, CPU, torch 2.13.0+cpu
* sampling {'temperature': 0.7, 'top_k': 40, 'top_p': 0.9, 'repetition_penalty': 1.3, 'max_new_tokens': 40}, torch re-seeded to 1234 per question
* git revision `aeb9b0ba39b4`

The hidden test split was not used.

## Headline

| configuration | accuracy | hallucination rate | over-abstention | abstention accuracy | s/question |
| --- | --- | --- | --- | --- | --- |
| model_only | **0.88%** (7/795) | 98.42% | 0.66% | 0.0% | 29.776 |
| plus_calculator | **90.94%** (723/795) | 8.7% | 0.0% | 83.33% | 0.798 |
| plus_retrieval | **4.91%** (39/795) | 94.86% | 0.0% | 0.0% | 0.232 |
| full_pipeline | **95.85%** (762/795) | 4.35% | 0.0% | 100.0% | 0.131 |

*hallucination rate* = asserted a wrong answer to a question that had one. *over-abstention* = declined a question that had an answer. *abstention accuracy* = correctly declined the unanswerable items. A system can only look good on all three at once by actually knowing which questions it can answer.

## By category

| category | model_only | plus_calculator | plus_retrieval | full_pipeline |
| --- | --- | --- | --- | --- |
| accounting | 2.5% (1/40) | 62.5% (25/40) | 22.5% (9/40) | 82.5% (33/40) |
| concepts | 4.35% (1/23) | 4.35% (1/23) | 82.61% (19/23) | 82.61% (19/23) |
| corporate_finance | 0.0% (0/97) | 100.0% (97/97) | 0.0% (0/97) | 100.0% (97/97) |
| extraction | 0.0% (0/100) | 100.0% (100/100) | 0.0% (0/100) | 100.0% (100/100) |
| general | 12.5% (1/8) | 12.5% (1/8) | 62.5% (5/8) | 62.5% (5/8) |
| hallucination | 0.0% (0/36) | 83.33% (30/36) | 0.0% (0/36) | 100.0% (36/36) |
| interpretation | 16.67% (2/12) | 16.67% (2/12) | 25.0% (3/12) | 25.0% (3/12) |
| ratios | 0.0% (0/140) | 100.0% (140/140) | 0.0% (0/140) | 100.0% (140/140) |
| reasoning | 0.0% (0/109) | 100.0% (109/109) | 0.0% (0/109) | 100.0% (109/109) |
| reporting | 7.69% (1/13) | 7.69% (1/13) | 23.08% (3/13) | 23.08% (3/13) |
| statements | 0.0% (0/117) | 100.0% (117/117) | 0.0% (0/117) | 100.0% (117/117) |
| valuation | 1.0% (1/100) | 100.0% (100/100) | 0.0% (0/100) | 100.0% (100/100) |

## By difficulty level

| level | model_only | plus_calculator | plus_retrieval | full_pipeline |
| --- | --- | --- | --- | --- |
| level 1 | 0.0% | 87.72% | 8.77% | 96.49% |
| level 2 | 1.79% | 85.27% | 10.27% | 93.75% |
| level 3 | 0.36% | 96.75% | 1.08% | 97.83% |
| level 4 | 0.0% | 79.1% | 1.49% | 89.55% |
| level 5 | 1.77% | 98.23% | 1.77% | 98.23% |

## By item kind

| kind | model_only | plus_calculator | plus_retrieval | full_pipeline |
| --- | --- | --- | --- | --- |
| authored | 8.22% (6/73) | 9.59% (7/73) | 53.42% (39/73) | 54.79% (40/73) |
| generated | 0.14% (1/722) | 99.17% (716/722) | 0.0% (0/722) | 100.0% (722/722) |

## What each component is worth

* **plus_calculator**: +90.06 points against model_only (model + deterministic calculator and derived identities)
* **plus_retrieval**: +4.03 points against model_only (model + curated glossary and retrieval layer)
* **full_pipeline**: +94.97 points against model_only (the served pipeline: calculator + knowledge + quality guard)

## Sample failures

* `dev_auth_0037` (concepts, hallucination): 'Valuation is the process of estimating the economic worth of an asset or company using methodologies such as Discounted Cash Flow (DCF), Comparable Companies, o'
* `dev_auth_0012` (accounting, hallucination): "Operating Cash Flow is the cash a company's core operations actually generate in a period, before investment and financing flows. It is harder to flatter than r"
* `dev_auth_0058` (interpretation, hallucination): 'Current Ratio evaluates whether a firm has enough liquid assets to pay short-term debt obligations due within one year. Formula: Current Ratio = Current Assets'
* `dev_auth_0052` (reporting, hallucination): 'The document refers to an initial public offering (IPO) or a secondary offering, and the main purpose of this Form 1099-B is to help investors decide whether it'
* `dev_auth_0045` (reporting, hallucination): 'A Fiscal Year is the 12-month period a company uses for reporting, which need not match the calendar year (many run April-March or October-September).\n\n(Closest'
* `dev_auth_0060` (interpretation, hallucination): "Operating Cash Flow is the cash a company's core operations actually generate in a period, before investment and financing flows. It is harder to flatter than r"
* `dev_auth_0013` (accounting, hallucination): 'Inventory is the goods a company holds to sell, plus the raw materials and work in progress used to make them. It sits in current assets, and turning it into sa'
* `dev_auth_0040` (reporting, hallucination): "The price of the stock, or a dividend payment, represents how much cash a company earns in dividends. It's generally used to make money by selling shares at ano"
