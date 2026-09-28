# Aivora v0.1-base - measured baseline

Benchmark: `data/benchmark/test.jsonl`, 204 items. Produced by `scripts/baseline_report.py`; every number below is measured, not estimated.

* checkpoint `checkpoint_247850.pt`, sha256 `6e9e50ca59e5...`
* 101,723,264 parameters, CPU, torch 2.13.0+cpu
* sampling {'temperature': 0.7, 'top_k': 40, 'top_p': 0.9, 'repetition_penalty': 1.3, 'max_new_tokens': 40}, torch re-seeded to 1234 per question
* git revision `85a39a021cc0`

The hidden test split was not used.

## Headline

| configuration | accuracy | hallucination rate | over-abstention | abstention accuracy | s/question |
| --- | --- | --- | --- | --- | --- |
| plus_calculator | **87.25%** (178/204) | 12.89% | 0.52% | 100.0% | 0.911 |
| full_pipeline | **94.61%** (193/204) | 5.15% | 0.52% | 100.0% | 0.226 |

*hallucination rate* = asserted a wrong answer to a question that had one. *over-abstention* = declined a question that had an answer. *abstention accuracy* = correctly declined the unanswerable items. A system can only look good on all three at once by actually knowing which questions it can answer.

## By category

| category | plus_calculator | full_pipeline |
| --- | --- | --- |
| accounting | 46.15% (6/13) | 84.62% (11/13) |
| concepts | 25.0% (2/8) | 75.0% (6/8) |
| corporate_finance | 100.0% (19/19) | 100.0% (19/19) |
| extraction | 100.0% (25/25) | 100.0% (25/25) |
| general | 0.0% (0/4) | 0.0% (0/4) |
| hallucination | 100.0% (10/10) | 100.0% (10/10) |
| interpretation | 33.33% (2/6) | 50.0% (3/6) |
| ratios | 100.0% (35/35) | 100.0% (35/35) |
| reasoning | 100.0% (25/25) | 100.0% (25/25) |
| reporting | 0.0% (0/5) | 100.0% (5/5) |
| statements | 100.0% (29/29) | 100.0% (29/29) |
| valuation | 100.0% (25/25) | 100.0% (25/25) |

## By difficulty level

| level | plus_calculator | full_pipeline |
| --- | --- | --- |
| level 1 | 92.86% | 100.0% |
| level 2 | 77.42% | 93.55% |
| level 3 | 91.04% | 94.03% |
| level 4 | 89.47% | 94.74% |
| level 5 | 92.86% | 92.86% |

## By item kind

| kind | plus_calculator | full_pipeline |
| --- | --- | --- |
| authored | 13.33% (4/30) | 63.33% (19/30) |
| generated | 100.0% (174/174) | 100.0% (174/174) |

## What each component is worth

* **full_pipeline**: +7.36 points against model_only (the served pipeline: calculator + knowledge + quality guard)

## Sample failures

* `test_auth_0026` (general, hallucination): "A stock (equity) represents fractional ownership in a corporation. Stockholders are entitled to a share of the company's assets and profits (via dividends and c"
* `test_auth_0029` (general, hallucination): 'An Interest Rate is the price of borrowing money, expressed as a percentage of the amount borrowed per period. Policy rates set by central banks influence rates'
* `test_auth_0021` (interpretation, hallucination): 'Answer: Debt/EBITDA\nFormula / Breakdown: Total Debt / EBITDA\nInputs: total_debt = 5.00, ebitda = 2.00\nResult: 2.50x'
* `test_auth_0028` (general, hallucination): 'Liquidity is how easily an asset converts to cash without losing value, and how readily a company can meet short-term obligations. Cash is the most liquid asset'
* `test_auth_0022` (interpretation, over_abstention): "I could not identify a complete calculation from the values provided. Values I parsed: {'beginning_value': 30.0, 'ending_value': 75.0}. Available calculations:"
* `test_auth_0009` (concepts, hallucination): 'Inventory Turnover shows how many times inventory is sold and replaced in a period. Formula: Inventory Turnover = COGS / Average Inventory. Higher usually means'
* `test_auth_0024` (interpretation, hallucination): 'The gap between the two margins is everything below the gross line: operating expenses - selling, general and administrative costs, research, marketing, depreci'
* `test_auth_0027` (general, hallucination): "A dividend is a token reward paid out from earnings or reserves to a company's shareholders, typically quarterly or annually.\n\n(Educational information from a r"
