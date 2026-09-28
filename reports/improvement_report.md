# Before / after: deterministic pipeline fixes

Both runs are the 795-item dev split, re-graded with the same current scorer from their stored predictions.

* before: `reports/baseline_report.json`
* after:  `reports/after/baseline_report.json`

No model weights were changed. The checkpoint is the same file in both runs.

## Headline accuracy

| configuration | before | after | absolute | relative |
| --- | --- | --- | --- | --- |
| model_only | 0.88% (7/795) | **0.88%** (7/795) | +0.00 pts | +0% |
| plus_calculator | 54.21% (431/795) | **90.94%** (723/795) | +36.73 pts | +68% |
| plus_retrieval | 4.91% (39/795) | **4.91%** (39/795) | +0.00 pts | +0% |
| full_pipeline | 58.36% (464/795) | **95.85%** (762/795) | +37.49 pts | +64% |

## The served pipeline (full_pipeline), metric by metric

| metric | before | after | change | wanted |
| --- | --- | --- | --- | --- |
| overall accuracy | 58.36% | **95.85%** | +37.49 (+64% rel) | higher |
| extraction accuracy | 23.0% | **100.0%** | +77.00 (+335% rel) | higher |
| abstention accuracy | 86.11% | **100.0%** | +13.89 (+16% rel) | higher |
| over-abstention rate | 15.28% | **0.0%** | -15.28 (-100% rel) | lower |
| hallucination rate | 27.67% | **4.35%** | -23.32 (-84% rel) | lower |
| seconds per question | 0.102 | 0.131 | | lower |

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
| accounting | 22.5% | **82.5%** | +60.00 (+267% rel) |
| concepts | 82.61% | **82.61%** | +0.00 (+0% rel) |
| corporate_finance | 50.52% | **100.0%** | +49.48 (+98% rel) |
| extraction | 23.0% | **100.0%** | +77.00 (+335% rel) |
| general | 62.5% | **62.5%** | +0.00 (+0% rel) |
| hallucination | 86.11% | **100.0%** | +13.89 (+16% rel) |
| interpretation | 25.0% | **25.0%** | +0.00 (+0% rel) |
| ratios | 70.71% | **100.0%** | +29.29 (+41% rel) |
| reasoning | 100.0% | **100.0%** | +0.00 (+0% rel) |
| reporting | 23.08% | **23.08%** | +0.00 (+0% rel) |
| statements | 63.25% | **100.0%** | +36.75 (+58% rel) |
| valuation | 40.0% | **100.0%** | +60.00 (+150% rel) |

## Per difficulty level (full_pipeline)

| level | before | after | change |
| --- | --- | --- | --- |
| 1 | 28.95% | **96.49%** | +67.54 (+233% rel) |
| 2 | 65.18% | **93.75%** | +28.57 (+44% rel) |
| 3 | 51.26% | **97.83%** | +46.57 (+91% rel) |
| 4 | 47.76% | **89.55%** | +41.79 (+88% rel) |
| 5 | 98.23% | **98.23%** | +0.00 (+0% rel) |

## Regressions

None: every item the served pipeline answered correctly before is still correct.

## Remaining failures (full_pipeline, after)

* **10** reporting / hallucination - e.g. `dev_auth_0052`: 'The document refers to an initial public offering (IPO) or a secondary offering, and the main purpose of this'
* **9** interpretation / hallucination - e.g. `dev_auth_0058`: 'Current Ratio evaluates whether a firm has enough liquid assets to pay short-term debt obligations due within'
* **7** accounting / hallucination - e.g. `dev_auth_0012`: "Operating Cash Flow is the cash a company's core operations actually generate in a period, before investment a"
* **4** concepts / hallucination - e.g. `dev_auth_0037`: 'Valuation is the process of estimating the economic worth of an asset or company using methodologies such as D'
* **3** general / hallucination - e.g. `dev_auth_0065`: 'A Bond is a loan made to a company or government in tradeable form: the issuer pays periodic interest (the cou'
