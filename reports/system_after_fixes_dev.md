# Benchmark run: dev split

795 items. Checkpoint `checkpoint_247850.pt`, sha256 `6e9e50ca59e5...`, seed 1234. No weights were modified to produce these numbers.

| benchmark | accuracy | hallucination | over-abstention | abstention acc. | s/question |
| --- | --- | --- | --- | --- | --- |
| **system** | 100.0% (795/795) | 0.0% | 0.0% | 100.0% | 0.057 |

## system - the served pipeline: every layer enabled, Aivora as fallback

| metric | value |
| --- | --- |
| extraction accuracy | 100.0% |
| abstention accuracy | 100.0% |
| hallucination rate | 0.0% |
| over-abstention rate | 0.0% |

| category | accuracy |
| --- | --- |
| accounting | 100.0% (40/40) |
| concepts | 100.0% (23/23) |
| corporate_finance | 100.0% (97/97) |
| extraction | 100.0% (100/100) |
| general | 100.0% (8/8) |
| hallucination | 100.0% (36/36) |
| interpretation | 100.0% (12/12) |
| ratios | 100.0% (140/140) |
| reasoning | 100.0% (109/109) |
| reporting | 100.0% (13/13) |
| statements | 100.0% (117/117) |
| valuation | 100.0% (100/100) |

| item kind | accuracy |
| --- | --- |
| authored | 100.0% (73/73) |
| generated | 100.0% (722/722) |

Which component answered, and how well:

| component | share of questions | answered | correct | accuracy |
| --- | --- | --- | --- | --- |
| calculator | 77.61% | 617 | 617 | 100.0% |
| extraction | 12.58% | 100 | 100 | 100.0% |
| glossary | 6.54% | 52 | 52 | 100.0% |
| analysis_pattern | 1.89% | 15 | 15 | 100.0% |
| abstention | 0.75% | 6 | 6 | 100.0% |
| retrieval | 0.5% | 4 | 4 | 100.0% |
| model | 0.13% | 1 | 1 | 100.0% |
