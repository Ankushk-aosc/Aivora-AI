# Benchmark run: dev split

60 items. Checkpoint `checkpoint_247850.pt`, sha256 `6e9e50ca59e5...`, seed 1234. No weights were modified to produce these numbers.

| benchmark | accuracy | hallucination | over-abstention | abstention acc. | s/question |
| --- | --- | --- | --- | --- | --- |
| **system** | 100.0% (60/60) | 0.0% | 0.0% | 100.0% | 0.01 |

## system - the served pipeline: every layer enabled, Aivora as fallback

| metric | value |
| --- | --- |
| extraction accuracy | 100.0% |
| abstention accuracy | 100.0% |
| hallucination rate | 0.0% |
| over-abstention rate | 0.0% |

| category | accuracy |
| --- | --- |
| accounting | 100.0% (2/2) |
| concepts | 100.0% (2/2) |
| corporate_finance | 100.0% (9/9) |
| extraction | 100.0% (16/16) |
| hallucination | 100.0% (3/3) |
| ratios | 100.0% (9/9) |
| reasoning | 100.0% (6/6) |
| statements | 100.0% (4/4) |
| valuation | 100.0% (9/9) |

| item kind | accuracy |
| --- | --- |
| authored | 100.0% (3/3) |
| generated | 100.0% (57/57) |

Which component answered, and how well:

| component | share of questions | answered | correct | accuracy |
| --- | --- | --- | --- | --- |
| calculator | 68.33% | 41 | 41 | 100.0% |
| extraction | 26.67% | 16 | 16 | 100.0% |
| glossary | 5.0% | 3 | 3 | 100.0% |
