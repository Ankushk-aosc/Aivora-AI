# Model comparison

Step 14. Everything below was measured in this repository with one scorer, except
the rows marked **not measured**, which state why rather than estimating.

## Headline

| Model | Params | Model-only accuracy | Interpretation | Reasoning | Hallucination |
| --- | --- | --- | --- | --- | --- |
| Aivora (hidden split, 204) | 101.7M | **3.43%** (7/204) | 33.33% (2/6) | 0% (0/25) | 96.39% |
| Aivora (blind prose, 144) | 101.7M | **7.64%** (11/144) | 6.67% | n/a | 92.36% |
| Aivora (dev split, 795) | 101.7M | **1.64%** (13/795) | - | - | 97.63% |
| Qwen2.5-1.5B-Instruct (base) | 1.5B | **not measured** | not measured | not measured | not measured |
| Qwen2.5-1.5B-Instruct + finance LoRA | 1.5B + ~15M | **not measured** | not measured | not measured | not measured |

### Why the Qwen rows are empty

Not an oversight and not a scheduling problem: **this machine cannot hold the
model.** 8 GB RAM total with 0.4 GB free at the time of measurement, against
~3.1 GB of weights, and `Qwen2.5-1.5B-Instruct` is not in the local Hugging Face
cache. `scripts/compare_models.py` is written, runs all three models through the
same evaluator, and refuses with that message rather than thrashing the machine.
It runs unchanged on Kaggle, which is where this comparison has to happen.

Filling those rows requires ~20 minutes of GPU time. Until then, treating any
Qwen figure as measured would be fabrication.

### Historical Qwen evidence, on a different and smaller set

From earlier work in this repository, on the 45-question category set through the
same `evaluate_generator` scorer:

| model | score |
| --- | --- |
| Qwen2.5-1.5B-Instruct, untuned | **38/45** (84%) |
| Qwen2.5-1.5B-Instruct + this project's finance LoRA | **40/45** (89%) |
| Aivora, model only | **1-5/45** (2-11%) |
| Aivora after distillation from that teacher | 4/45 |

This is real evidence and it is not the same measurement as the table above: a
different benchmark, 45 items, and sampled rather than greedy decoding for
Aivora. It is reported here because it is the only external-model data this
project has, not as a substitute for Step 8.

## Full-system results, for context

The system is the pipeline: routing, extraction, calculator, glossary, retrieval,
diagnostic patterns, guard, with Aivora as the fallback.

| benchmark | system | model-only | what the gap is |
| --- | --- | --- | --- |
| dev (795, saturated) | 100% | 1.64% | deterministic layers answer 99.87% of questions |
| hidden (204, spent) | 96.08% | 3.43% | 189 of 204 correct answers came from questions the model got wrong |
| blind prose (144) | 13.89% | 7.64% | on unseen prose the layers have no coverage, so the model answers 66.7% of it |

### The decisive comparison (hidden split, 204 items)

| group | count | share |
| --- | --- | --- |
| system correct / model correct | 7 | 3.43% |
| **system correct / model wrong** | **189** | **92.65%** |
| **system wrong / model correct** | **0** | **0.00%** |
| system wrong / model wrong | 8 | 3.92% |

**The model contributed no unique wins.** In 204 questions there is not one the
model answered and the pipeline did not. 92.65 points of the 96.08% system score
sit on questions the model got wrong. The deterministic components are not
assisting Aivora; they are standing in for it.

## Aivora generation behaviour (hidden split, after the inference fix)

| metric | value |
| --- | --- |
| mean / median generated tokens | 39.9 / 40 (the cap) |
| min / max | 19 / 40 |
| max-token stops | 201 (98.5%) |
| repetition stops | 3 (1.5%) |
| **EOS stops** | **0 (0.0%)** |
| `<|endoftext|>` emitted anywhere | **0** |
| premature stops (<20 tokens) | 1 (0.5%) |
| empty or near-empty answers | 0 (0.0%) |
| abstention rate | **0.00%** |

The path is healthy: the truncation bug that killed 76.7% of greedy answers is
gone, nothing is empty, and generation runs to its budget. The model simply never
ends an answer, and `docs/AIVORA_TRAINING_DATA_AUDIT.md` shows why - the token
was never in its training data.

## By intent, Aivora model-only on the hidden split

| intent | accuracy |
| --- | --- |
| interpretation | 33.33% (2/6) - two rubric coincidences on a six-item sample |
| concept | 6.67% (2/30) |
| calculation | 2.78% (3/108) |
| **extraction** | **0% (0/25)** |
| **reasoning** | **0% (0/25)** |
| **abstention** | **0% (0/10)** |

By difficulty: 3.57%, 4.84%, 1.49%, 5.26%, 3.57% across levels 1-5. **Flat.** A
model with partial competence degrades as difficulty rises; this one does not,
because it is not on the curve at any level.

## Reading the numbers honestly

* Every Aivora figure here **overstates** its ability. Hand-reading the correct
  answers shows most are rubric coincidences: *"Cash rose sharply while free cash
  flow was negative. Where did the cash come from?"* was scored correct for *"The
  cash went down in price when a company bought back its own"*.
* The hidden split has now been evaluated three times (system twice, model once).
  It is reference data, not a blind benchmark.
* The blind prose set was authored by Claude in the same session as the pipeline
  it measures, so its 13.89% is an upper bound for the curated layers.
* The one measurement in this document that is clean is the four-group comparison:
  it needs no rubric judgement, only agreement between two runs on the same items.
