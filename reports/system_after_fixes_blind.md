# Benchmark run: blind split

144 items. Checkpoint `checkpoint_247850.pt`, sha256 `6e9e50ca59e5...`, seed 1234. No weights were modified to produce these numbers.

| benchmark | accuracy | hallucination | over-abstention | abstention acc. | s/question |
| --- | --- | --- | --- | --- | --- |
| **system** | 13.89% (20/144) | 82.64% | 3.47% | None% | 4.931 |

## system - the served pipeline: every layer enabled, Aivora as fallback

| metric | value |
| --- | --- |
| extraction accuracy | None% |
| abstention accuracy | None% |
| hallucination rate | 82.64% |
| over-abstention rate | 3.47% |

| category | accuracy |
| --- | --- |
| accounting | 24.0% (6/25) |
| concepts | 12.0% (3/25) |
| corporate_finance | 20.0% (3/15) |
| general | 22.22% (2/9) |
| interpretation | 0.0% (0/15) |
| reporting | 8.0% (2/25) |
| risk | 20.0% (3/15) |
| valuation | 6.67% (1/15) |

| item kind | accuracy |
| --- | --- |
| authored | 13.89% (20/144) |

Which component answered, and how well:

| component | share of questions | answered | correct | accuracy |
| --- | --- | --- | --- | --- |
| model | 66.67% | 96 | 6 | 6.25% |
| glossary | 16.67% | 24 | 6 | 25.0% |
| retrieval | 11.11% | 16 | 7 | 43.75% |
| calculator | 2.78% | 4 | 0 | 0.0% |
| analysis_pattern | 1.39% | 2 | 1 | 50.0% |
| abstention | 1.39% | 2 | 0 | 0.0% |
