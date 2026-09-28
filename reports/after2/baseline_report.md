# Aivora v0.1-base - measured baseline

Benchmark: `data/benchmark/dev.jsonl`, 795 items. Produced by `scripts/baseline_report.py`; every number below is measured, not estimated.

* checkpoint `checkpoint_247850.pt`, sha256 `6e9e50ca59e5...`
* 101,723,264 parameters, CPU, torch 2.13.0+cpu
* sampling {'temperature': 0.7, 'top_k': 40, 'top_p': 0.9, 'repetition_penalty': 1.3, 'max_new_tokens': 40}, torch re-seeded to 1234 per question
* git revision `da46e4859960`

The hidden test split was not used.

## Headline

| configuration | accuracy | hallucination rate | over-abstention | abstention accuracy | s/question |
| --- | --- | --- | --- | --- | --- |
| plus_calculator | **90.94%** (723/795) | 8.7% | 0.0% | 83.33% | 0.609 |
| full_pipeline | **99.5%** (791/795) | 0.53% | 0.0% | 100.0% | 0.069 |

*hallucination rate* = asserted a wrong answer to a question that had one. *over-abstention* = declined a question that had an answer. *abstention accuracy* = correctly declined the unanswerable items. A system can only look good on all three at once by actually knowing which questions it can answer.

## By category

| category | plus_calculator | full_pipeline |
| --- | --- | --- |
| accounting | 62.5% (25/40) | 97.5% (39/40) |
| concepts | 4.35% (1/23) | 95.65% (22/23) |
| corporate_finance | 100.0% (97/97) | 100.0% (97/97) |
| extraction | 100.0% (100/100) | 100.0% (100/100) |
| general | 12.5% (1/8) | 100.0% (8/8) |
| hallucination | 83.33% (30/36) | 100.0% (36/36) |
| interpretation | 16.67% (2/12) | 91.67% (11/12) |
| ratios | 100.0% (140/140) | 100.0% (140/140) |
| reasoning | 100.0% (109/109) | 100.0% (109/109) |
| reporting | 7.69% (1/13) | 92.31% (12/13) |
| statements | 100.0% (117/117) | 100.0% (117/117) |
| valuation | 100.0% (100/100) | 100.0% (100/100) |

## By difficulty level

| level | plus_calculator | full_pipeline |
| --- | --- | --- |
| level 1 | 87.72% | 100.0% |
| level 2 | 85.27% | 99.11% |
| level 3 | 96.75% | 99.64% |
| level 4 | 79.1% | 100.0% |
| level 5 | 98.23% | 99.12% |

## By item kind

| kind | plus_calculator | full_pipeline |
| --- | --- | --- |
| authored | 9.59% (7/73) | 94.52% (69/73) |
| generated | 99.17% (716/722) | 100.0% (722/722) |

## What each component is worth

* **full_pipeline**: +8.56 points against model_only (the served pipeline: calculator + knowledge + quality guard)

## Sample failures

* `dev_auth_0038` (concepts, hallucination): "The amount that can be calculated using the example of $5,000 in cash or stock options. It's only $100 in total for an asset such as stocks and bonds on the sam"
* `dev_auth_0051` (reporting, hallucination): 'No document is currently loaded. Upload a financial document first.'
* `dev_auth_0003` (accounting, hallucination): "The amount that can be calculated using the example above is $20,000. This means that you don't have to pay taxes on your home, but if it's used for other purpo"
* `dev_auth_0064` (interpretation, hallucination): 'Falling interest coverage means earnings cover interest by a smaller margin, so less room remains before the company cannot service its debt from operations. Co'
