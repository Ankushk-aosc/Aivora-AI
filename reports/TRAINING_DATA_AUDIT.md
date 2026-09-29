# Training data audit

Phase 7. Record-level measurement of every training dataset on disk, plus what the pretraining manifest can tell us about the corpus that lives on Kaggle. Produced by `scripts/audit_training_data.py`; nothing was changed.

## Pretraining corpus (from the checkpoint's own manifest)

| | |
| --- | --- |
| unique train tokens | **120,857,129** |
| tokens processed | 1,964,851,200 |
| **epochs over the corpus** | **16.26** |
| datasets | 19 |
| exact duplicates removed at prepare time | 6500 |
| evaluation leakage removed at prepare time | 1 |

_the pretraining shards live in the Kaggle environment; this is from the manifest the checkpoint embedded, and record-level measures are not available locally_

## financial_sft_v1

`data\sft\financial_sft_v1.jsonl`, sha256 `329372998e3a...`

| measure | value |
| --- | --- |
| records | 7,849 |
| malformed (empty question or answer) | 0 |
| total tokens | 522,736 |
| mean / median length | 66.6 / 58 tokens |
| p95 / max length | 130 / 156 tokens |
| exact duplicates | 0 |
| near-duplicate rate | 0.014% (28,500 pairs compared, Jaccard >= 0.8 on 5-word shingles) |
| rows with a numeric answer | 3,890 |
| **numeric answers disagreeing with their own prose** | **0** |

Task types: calculation 3,083, extraction 2,373, definition 610, interpretation 483, abstention 473, reasoning 407, grounded 400, explanation 20

Verification: verified_present_in_context 2,373, verified_arithmetic 2,183, verified_arithmetic_and_calculator 1,144, verified_curated 630, verified_pattern 483, verified_insufficient 466, verified_grounded 400, verified_arithmetic_and_substitution 163, verified_current_data 7

Domain coverage (records mentioning the domain's vocabulary - an estimate, not a classification):

| domain | records |
| --- | --- |
| ratios | 2,900 |
| cash_flow | 2,155 |
| financial_statements | 942 |
| valuation | 915 |
| corporate_finance | 909 |
| accounting | 658 |
| derivatives | 386 |
| markets | 302 |
| risk | 207 |
| banking | 37 |
| economics | 32 |
| fixed_income | 16 |
| portfolio_management | 10 |

Task signals: definitions 5,387, reasoning 4,263, extraction 2,010, current_data 1,957, interpretation 483, abstention 473, calculations 0

Synthetic vs sourced: 7,849 generated or curated in this repository, 0 from an external source.

## tiny_overfit

`data\tiny_overfit\tiny_financial.jsonl`, sha256 `1e538695a23e...`

| measure | value |
| --- | --- |
| records | 54 |
| malformed (empty question or answer) | 0 |
| total tokens | 1,886 |
| mean / median length | 34.9 / 34 tokens |
| p95 / max length | 45 / 47 tokens |
| exact duplicates | 0 |
| near-duplicate rate | 0.0% (996 pairs compared, Jaccard >= 0.8 on 5-word shingles) |
| rows with a numeric answer | 12 |
| **numeric answers disagreeing with their own prose** | **0** |

Task types: definition 12, extraction 12, calculation 12, explanation 6, interpretation 6, abstention 6

Verification: verified_definition 12, verified_present_in_context 12, verified_arithmetic 12, verified_explanation 6, verified_interpretation 6, verified_insufficient 4, verified_current_data 2

Domain coverage (records mentioning the domain's vocabulary - an estimate, not a classification):

| domain | records |
| --- | --- |
| ratios | 21 |
| cash_flow | 13 |
| accounting | 6 |
| corporate_finance | 5 |
| risk | 3 |
| markets | 2 |
| financial_statements | 1 |
| economics | 1 |
| valuation | 0 |
| banking | 0 |
| portfolio_management | 0 |
| fixed_income | 0 |
| derivatives | 0 |

Task signals: definitions 40, reasoning 12, interpretation 6, abstention 6, current_data 5, extraction 0, calculations 0

Synthetic vs sourced: 54 generated or curated in this repository, 0 from an external source.

## instruction_legacy

`data\instruction\financial_instructions.jsonl`, sha256 `4c1e83d60f93...`

| measure | value |
| --- | --- |
| records | 800 |
| malformed (empty question or answer) | 0 |
| total tokens | 194,576 |
| mean / median length | 243.2 / 181 tokens |
| p95 / max length | 627 / 2070 tokens |
| exact duplicates | 0 |
| near-duplicate rate | 0.0% (18,900 pairs compared, Jaccard >= 0.8 on 5-word shingles) |
| rows with a numeric answer | 0 |
| **numeric answers disagreeing with their own prose** | **0** |

Task types: unlabelled 800

Verification: none 800

Domain coverage (records mentioning the domain's vocabulary - an estimate, not a classification):

| domain | records |
| --- | --- |
| banking | 250 |
| derivatives | 187 |
| ratios | 182 |
| risk | 170 |
| markets | 156 |
| accounting | 150 |
| economics | 87 |
| valuation | 62 |
| corporate_finance | 62 |
| portfolio_management | 59 |
| fixed_income | 57 |
| cash_flow | 10 |
| financial_statements | 5 |

Task signals: reasoning 346, current_data 137, definitions 69, calculations 47, abstention 7, interpretation 4, extraction 2

Synthetic vs sourced: 0 generated or curated in this repository, 800 from an external source.

## What this audit says

* The new SFT set is **7,849 records / 522,736 tokens**, entirely generated or curated in this repository. Nothing in it came from an external instruction dataset, which is a limitation worth stating: it inherits this project's idea of a good answer.
* **0** numeric answers disagree with their own explanation text - the builder computes both from one expression, and the independent validator checks it again.
* Thin or absent domains are visible in the coverage table above and are the honest gap list for the next dataset version: portfolio management, derivatives and fixed income are barely represented, and banking only incidentally.
* The legacy instruction file is finance-alpaca content, which is Reddit-style personal finance rather than financial analysis. It is kept for provenance, not used by the new SFT experiment.
