# Before / after: deterministic pipeline fixes

Both runs are the 795-item dev split, re-graded with the same current scorer from their stored predictions.

* before: `reports/baseline_report.json`
* after:  `reports/after3/baseline_report.json`

No model weights were changed. The checkpoint is the same file in both runs.

## Headline accuracy

| configuration | before | after | absolute | relative |
| --- | --- | --- | --- | --- |
| plus_retrieval | 4.91% (39/795) | **9.18%** (73/795) | +4.27 pts | +87% |
| full_pipeline | 58.36% (464/795) | **100.0%** (795/795) | +41.64 pts | +71% |

## The served pipeline (full_pipeline), metric by metric

| metric | before | after | change | wanted |
| --- | --- | --- | --- | --- |
| overall accuracy | 58.36% | **100.0%** | +41.64 (+71% rel) | higher |
| extraction accuracy | 23.0% | **100.0%** | +77.00 (+335% rel) | higher |
| abstention accuracy | 86.11% | **100.0%** | +13.89 (+16% rel) | higher |
| over-abstention rate | 15.28% | **0.0%** | -15.28 (-100% rel) | lower |
| hallucination rate | 27.67% | **0.0%** | -27.67 (-100% rel) | lower |
| seconds per question | 0.102 | 0.051 | | lower |

## The failure groups the baseline named

| group | before | after | change |
| --- | --- | --- | --- |
| extraction (77 failures) | 23.0% (23/100) | **100.0%** (100/100) | +77.00 (+335% rel) |
| statement derivations (66 failures) | 52.86% (74/140) | **100.0%** (140/140) | +47.14 (+89% rel) |
| dividend payout / WACC / EPS / EV parser gaps | 45.18% (89/197) | **100.0%** (197/197) | +54.82 (+121% rel) |
| quick ratio / asset turnover / interest coverage | 2.38% (1/42) | **100.0%** (42/42) | +97.62 (+4102% rel) |
| current-data and missing-figure abstention | 86.11% (31/36) | **100.0%** (36/36) | +13.89 (+16% rel) |

## Per category (full_pipeline)

| category | before | after | change |
| --- | --- | --- | --- |
| accounting | 22.5% | **100.0%** | +77.50 (+344% rel) |
| concepts | 82.61% | **100.0%** | +17.39 (+21% rel) |
| corporate_finance | 50.52% | **100.0%** | +49.48 (+98% rel) |
| extraction | 23.0% | **100.0%** | +77.00 (+335% rel) |
| general | 62.5% | **100.0%** | +37.50 (+60% rel) |
| hallucination | 86.11% | **100.0%** | +13.89 (+16% rel) |
| interpretation | 25.0% | **100.0%** | +75.00 (+300% rel) |
| ratios | 70.71% | **100.0%** | +29.29 (+41% rel) |
| reasoning | 100.0% | **100.0%** | +0.00 (+0% rel) |
| reporting | 23.08% | **100.0%** | +76.92 (+333% rel) |
| statements | 63.25% | **100.0%** | +36.75 (+58% rel) |
| valuation | 40.0% | **100.0%** | +60.00 (+150% rel) |

## Per difficulty level (full_pipeline)

| level | before | after | change |
| --- | --- | --- | --- |
| 1 | 28.95% | **100.0%** | +71.05 (+245% rel) |
| 2 | 65.18% | **100.0%** | +34.82 (+53% rel) |
| 3 | 51.26% | **100.0%** | +48.74 (+95% rel) |
| 4 | 47.76% | **100.0%** | +52.24 (+109% rel) |
| 5 | 98.23% | **100.0%** | +1.77 (+2% rel) |

## Regressions

None: every item the served pipeline answered correctly before is still correct.

## Remaining failures (full_pipeline, after)

