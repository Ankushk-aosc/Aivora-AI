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
| plus_retrieval | **9.18%** (73/795) | 90.38% | 0.0% | 0.0% | 0.749 |
| full_pipeline | **100.0%** (795/795) | 0.0% | 0.0% | 100.0% | 0.051 |

*hallucination rate* = asserted a wrong answer to a question that had one. *over-abstention* = declined a question that had an answer. *abstention accuracy* = correctly declined the unanswerable items. A system can only look good on all three at once by actually knowing which questions it can answer.

## By category

| category | plus_retrieval | full_pipeline |
| --- | --- | --- |
| accounting | 42.5% (17/40) | 100.0% (40/40) |
| concepts | 100.0% (23/23) | 100.0% (23/23) |
| corporate_finance | 0.0% (0/97) | 100.0% (97/97) |
| extraction | 0.0% (0/100) | 100.0% (100/100) |
| general | 100.0% (8/8) | 100.0% (8/8) |
| hallucination | 0.0% (0/36) | 100.0% (36/36) |
| interpretation | 100.0% (12/12) | 100.0% (12/12) |
| ratios | 0.0% (0/140) | 100.0% (140/140) |
| reasoning | 0.0% (0/109) | 100.0% (109/109) |
| reporting | 100.0% (13/13) | 100.0% (13/13) |
| statements | 0.0% (0/117) | 100.0% (117/117) |
| valuation | 0.0% (0/100) | 100.0% (100/100) |

## By difficulty level

| level | plus_retrieval | full_pipeline |
| --- | --- | --- |
| level 1 | 12.28% | 100.0% |
| level 2 | 16.52% | 100.0% |
| level 3 | 3.61% | 100.0% |
| level 4 | 11.94% | 100.0% |
| level 5 | 3.54% | 100.0% |

## By item kind

| kind | plus_retrieval | full_pipeline |
| --- | --- | --- |
| authored | 100.0% (73/73) | 100.0% (73/73) |
| generated | 0.0% (0/722) | 100.0% (722/722) |

## What each component is worth

* **full_pipeline**: +90.82 points against model_only (the served pipeline: calculator + knowledge + quality guard)

## Sample failures

