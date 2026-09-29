# Benchmark run: test split

204 items. Checkpoint `checkpoint_247850.pt`, sha256 `6e9e50ca59e5...`, seed 1234. No weights were modified to produce these numbers.

| benchmark | accuracy | hallucination | over-abstention | abstention acc. | s/question |
| --- | --- | --- | --- | --- | --- |
| **model** | 3.43% (7/204) | 96.39% | 0.0% | 0.0% | 7.526 |

## model - Aivora alone: calculator, glossary, retrieval, patterns and guard all off

| metric | value |
| --- | --- |
| extraction accuracy | 0.0% |
| abstention accuracy | 0.0% |
| hallucination rate | 96.39% |
| over-abstention rate | 0.0% |

| category | accuracy |
| --- | --- |
| accounting | 0.0% (0/13) |
| concepts | 25.0% (2/8) |
| corporate_finance | 0.0% (0/19) |
| extraction | 0.0% (0/25) |
| general | 0.0% (0/4) |
| hallucination | 0.0% (0/10) |
| interpretation | 33.33% (2/6) |
| ratios | 5.71% (2/35) |
| reasoning | 0.0% (0/25) |
| reporting | 0.0% (0/5) |
| statements | 0.0% (0/29) |
| valuation | 4.0% (1/25) |

| item kind | accuracy |
| --- | --- |
| authored | 13.33% (4/30) |
| generated | 1.72% (3/174) |

Which component answered, and how well:

| component | share of questions | answered | correct | accuracy |
| --- | --- | --- | --- | --- |
| model | 100.0% | 204 | 7 | 3.43% |
