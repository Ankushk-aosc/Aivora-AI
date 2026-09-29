# Benchmark run: test split

204 items. Checkpoint `checkpoint_247850.pt`, sha256 `6e9e50ca59e5...`, seed 1234. No weights were modified to produce these numbers.

| benchmark | accuracy | hallucination | over-abstention | abstention acc. | s/question |
| --- | --- | --- | --- | --- | --- |
| **system** | 96.08% (196/204) | 4.12% | 0.0% | 100.0% | 0.266 |

## system - the served pipeline: every layer enabled, Aivora as fallback

| metric | value |
| --- | --- |
| extraction accuracy | 100.0% |
| abstention accuracy | 100.0% |
| hallucination rate | 4.12% |
| over-abstention rate | 0.0% |

| category | accuracy |
| --- | --- |
| accounting | 84.62% (11/13) |
| concepts | 87.5% (7/8) |
| corporate_finance | 100.0% (19/19) |
| extraction | 100.0% (25/25) |
| general | 0.0% (0/4) |
| hallucination | 100.0% (10/10) |
| interpretation | 100.0% (6/6) |
| ratios | 100.0% (35/35) |
| reasoning | 100.0% (25/25) |
| reporting | 80.0% (4/5) |
| statements | 100.0% (29/29) |
| valuation | 100.0% (25/25) |

| item kind | accuracy |
| --- | --- |
| authored | 73.33% (22/30) |
| generated | 100.0% (174/174) |

Which component answered, and how well:

| component | share of questions | answered | correct | accuracy |
| --- | --- | --- | --- | --- |
| calculator | 73.04% | 149 | 149 | 100.0% |
| extraction | 12.25% | 25 | 25 | 100.0% |
| glossary | 8.33% | 17 | 14 | 82.35% |
| analysis_pattern | 2.94% | 6 | 6 | 100.0% |
| model | 2.94% | 6 | 1 | 16.67% |
| retrieval | 0.49% | 1 | 1 | 100.0% |
