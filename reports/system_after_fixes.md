# System state after the pre-training fixes

Produced by `scripts/consolidate_after_fixes.py` from the per-split runs of `scripts/run_benchmarks.py`. No model weights were modified anywhere in this work.

## Splits

| split | items | what it measures |
| --- | --- | --- |
| `blind` (authored prose) | 144 | 144 authored questions, 13.7% of the 1,000 the brief asked for; a fair test of the code, an upper bound for the glossary |
| `dev` (development) | 795 | iterated against while building the pipeline: it describes this work rather than measuring it |
| `test` (hidden) | 204 | measurement only, spent once; its prose items were authored by Claude earlier in this session |

## SYSTEM vs MODEL

| split | benchmark | accuracy | hallucination | over-abstention | abstention acc. |
| --- | --- | --- | --- | --- | --- |
| blind | system | **13.89%** (20/144) | 82.64% | 3.47% | None% |
| blind | model | **7.64%** (11/144) | 92.36% | 0.0% | None% |
| dev | system | **100.0%** (795/795) | 0.0% | 0.0% | 100.0% |
| test | system | **96.08%** (196/204) | 4.12% | 0.0% | 100.0% |

## Part 20 metric table

| metric | blind/system | blind/model | dev/system | test/system |
| --- | --- | --- | --- | --- |
| overall accuracy | 13.89% | 7.64% | 100.0% | 96.08% |
| extraction accuracy | None% | None% | 100.0% | 100.0% |
| abstention accuracy | None% | None% | 100.0% | 100.0% |
| hallucination rate | 82.64% | 92.36% | 0.0% | 4.12% |
| over-abstention rate | 3.47% | 0.0% | 0.0% | 0.0% |
| calculation accuracy | 13.33% | 10.0% | 100.0% | 100.0% |
| reasoning accuracy | None% | None% | 100.0% | 100.0% |
| interpretation accuracy | 0.0% | 6.67% | 100.0% | 100.0% |
| concept accuracy | 14.67% | 6.67% | 100.0% | 84.62% |

## Which component answers, and how well

**blind / system**

| component | share | answered | correct | accuracy |
| --- | --- | --- | --- | --- |
| model | 66.67% | 96 | 6 | 6.25% |
| glossary | 16.67% | 24 | 6 | 25.0% |
| retrieval | 11.11% | 16 | 7 | 43.75% |
| calculator | 2.78% | 4 | 0 | 0.0% |
| analysis_pattern | 1.39% | 2 | 1 | 50.0% |
| abstention | 1.39% | 2 | 0 | 0.0% |

**blind / model**

| component | share | answered | correct | accuracy |
| --- | --- | --- | --- | --- |
| model | 99.31% | 143 | 11 | 7.69% |
| abstention | 0.69% | 1 | 0 | 0.0% |

**dev / system**

| component | share | answered | correct | accuracy |
| --- | --- | --- | --- | --- |
| calculator | 77.61% | 617 | 617 | 100.0% |
| extraction | 12.58% | 100 | 100 | 100.0% |
| glossary | 6.54% | 52 | 52 | 100.0% |
| analysis_pattern | 1.89% | 15 | 15 | 100.0% |
| abstention | 0.75% | 6 | 6 | 100.0% |
| retrieval | 0.5% | 4 | 4 | 100.0% |
| model | 0.13% | 1 | 1 | 100.0% |

**test / system**

| component | share | answered | correct | accuracy |
| --- | --- | --- | --- | --- |
| calculator | 73.04% | 149 | 149 | 100.0% |
| extraction | 12.25% | 25 | 25 | 100.0% |
| glossary | 8.33% | 17 | 14 | 82.35% |
| analysis_pattern | 2.94% | 6 | 6 | 100.0% |
| model | 2.94% | 6 | 1 | 16.67% |
| retrieval | 0.49% | 1 | 1 | 100.0% |
