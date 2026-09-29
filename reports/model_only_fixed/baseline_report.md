# Aivora v0.1-base - measured baseline

Benchmark: `data/benchmark/dev.jsonl`, 795 items. Produced by `scripts/baseline_report.py`; every number below is measured, not estimated.

* checkpoint `checkpoint_247850.pt`, sha256 `6e9e50ca59e5...`
* 101,723,264 parameters, CPU, torch 2.13.0+cpu
* sampling {'temperature': 0.7, 'top_k': 40, 'top_p': 0.9, 'repetition_penalty': 1.3, 'max_new_tokens': 40}, torch re-seeded to 1234 per question
* git revision `aae7c68dd225`

The hidden test split was not used.

## Headline

| configuration | accuracy | hallucination rate | over-abstention | abstention accuracy | s/question |
| --- | --- | --- | --- | --- | --- |
| model_only | **1.64%** (13/795) | 97.63% | 0.66% | 0.0% | 8.239 |

*hallucination rate* = asserted a wrong answer to a question that had one. *over-abstention* = declined a question that had an answer. *abstention accuracy* = correctly declined the unanswerable items. A system can only look good on all three at once by actually knowing which questions it can answer.

## By category

| category | model_only |
| --- | --- |
| accounting | 2.5% (1/40) |
| concepts | 8.7% (2/23) |
| corporate_finance | 0.0% (0/97) |
| extraction | 0.0% (0/100) |
| general | 25.0% (2/8) |
| hallucination | 0.0% (0/36) |
| interpretation | 16.67% (2/12) |
| ratios | 0.71% (1/140) |
| reasoning | 1.83% (2/109) |
| reporting | 7.69% (1/13) |
| statements | 0.85% (1/117) |
| valuation | 1.0% (1/100) |

## By difficulty level

| level | model_only |
| --- | --- |
| level 1 | 0.0% |
| level 2 | 3.12% |
| level 3 | 0.72% |
| level 4 | 0.0% |
| level 5 | 3.54% |

## By item kind

| kind | model_only |
| --- | --- |
| authored | 10.96% (8/73) |
| generated | 0.69% (5/722) |

## What each component is worth


## Sample failures

* `dev_auth_0028` (concepts, hallucination): "The amount that can be calculated using a formula. For example, an asset's price of $1 million and the current market value of $2 million it will be worth $4 mi"
* `dev_extract_0041` (extraction, hallucination): 'The total debt of the company includes all other capital items such as accounts receivable, inventory, and interest expense ratios; basic financial information'
* `dev_stmt_0028` (accounting, hallucination): "The shareholder's ownership percentage stands at 0.01%. It can be calculated by dividing the total number of shares outstanding in this case as a whole, based o"
* `dev_extract_0066` (extraction, hallucination): "The total obligations mentioned are calculated using the company's most recent quarterly or annual report, net of taxes payable and other current liabilities. I"
* `dev_corp_0085` (corporate_finance, hallucination): "The dividend payout ratio of 0.10% is 18.20%. It's a percentage that can be used to compare with other companies in terms of company performance.The U.S.-China"
* `dev_val_0053` (valuation, hallucination): 'The CRSR is calculated as a percentage of its revenue on an annual basis, and the CAGR represents 10% of total revenues for that year.The company’s operating in'
* `dev_stmt_0098` (statements, hallucination): 'EBITDA (net income) - Net Income/Earnings from Assets (LTC), Expenses & Cash Flow - Cash Flows - Cash Flows In addition to operating cash flow, you'
* `dev_val_0034` (valuation, hallucination): "The price per share of the stock depends on a variety of factors such as company performance, market conditions, and other relevant information. It's also worth"
