# Aivora: pre-training fix and validation report

Covers the whole brief: inference validation, the intent layer, extraction,
retrieval precision, the diagnostic layer, abstention, the separated benchmarks
and the training decision. **No model weights were modified, and no training was
started.**

Companion files: `reports/model_inference_validation.md` (Parts 1-4),
`reports/system_after_fixes.{json,md}` (Parts 19-20),
`reports/remaining_bottleneck.md` and `reports/improvement_report.md` (the
earlier round), `data/benchmark/blind_manifest.json` (Part 18).

---

## The one-paragraph version

The 0.88% model-only result was contaminated by a real inference bug - the
repetition detector counted prompt tokens, so any answer that named the term it
was asked about was truncated after one or two tokens. Fixing it moved model-only
from 0.88% to 1.64%, which settles the question: the bug was worth 0.8 points and
the remaining 98.4% is the model. Meanwhile a new 144-question authored set
exposed something the earlier splits could not: on unseen prose the served
pipeline scores **13.89%**, not the 100% it scores on the development split, and
two thirds of those answers come from the model at 6.25% accuracy. The
deterministic layers are excellent at what they cover (100% of unseen generated
items) and cover almost none of open-ended financial language.

---

## 1. Is the Aivora inference path working correctly?

**Now yes; before, no.** Details and token-level evidence in
`reports/model_inference_validation.md`.

| check | verdict |
| --- | --- |
| checkpoint is step 247,850, sha256 `6e9e50ca59e5…` | correct, and not one of the earlier checkpoints |
| config comes from the checkpoint's own sidecar | correct |
| tokenizer GPT-2 BPE, vocab 50,257 matching the model | correct |
| special tokens: `<|endoftext|>`=50256, no separate BOS/PAD | correct for this tokenizer |
| prompt template, context truncation, dtype, device | correct |
| repetition detector | **was broken** - window spanned prompt + generated |
| EOS handling | **was absent** - added, though the model never emits the token |
| KV cache | **absent** - a performance defect, ~8 s per answer on CPU |

The defect, with token ids:

```
prompt    "Question: What is EBITDA?\nAnswer:"   [..., 412, 26094, 5631, 30, 198, 33706, 25]
generated [412, 26094]  = " E","BIT"
stop      repeated_bigram at step 2 - the bigram was already in the PROMPT
```

Greedy premature-stop rate was **76.7%** (23 of 30), production **20%**. After
the fix: 20% and **0%**.

## 2. Why was model-only accuracy only ~0.88%?

About 0.8 points of it was the truncation bug. The rest is capability. After the
fix the model still answers like this:

> "What is EBITDA?" → *"EBITDA is the difference between EBITDA and EBITDA…"*

A sampling matrix of 12 temperature/top_p combinations and three length budgets
produced 0% premature stops and 100% mechanically-usable output at every setting,
and accuracy did not follow. There is no decoding configuration that rescues it.

## 3. How much did routing and extraction improve?

Measured on the 795-item development split, full pipeline:

| | before | after |
| --- | --- | --- |
| overall | 58.36% | **100%** |
| extraction | 23.0% | **100%** |
| over-abstention | 15.28% | **0%** |
| hallucination | 27.67% | **0%** |

On the hidden split, **94.61% → 96.08%**. Generated items (unseen values) are at
**100%**, which is the meaningful part: the parser and router work on the shape
of a problem, not on memorised instances.

## 4-6. What does each component contribute?

Component shares, measured per question by which layer produced the answer.

**Development split (795):** calculator 77.6%, extraction 12.6%, glossary 6.5%,
patterns 1.9%, abstention 0.8%, retrieval 0.5%, **model 0.13%**. The model
answers one question in 800.

**Unseen authored prose (144):**

| component | share | answered | correct | accuracy |
| --- | --- | --- | --- | --- |
| model | 66.7% | 96 | 6 | **6.25%** |
| glossary | 16.7% | 24 | 6 | 25.0% |
| retrieval | 11.1% | 16 | 7 | **43.75%** |
| calculator | 2.8% | 4 | 0 | 0.0% |
| analysis_pattern | 1.4% | 2 | 1 | 50.0% |

Retrieval is the most accurate layer when it fires and fires rarely; the
calculator is decisive where it applies and misfires on four prose questions;
the model is the default answer for anything uncovered and is almost always
wrong.

## 7. What is Aivora's actual model-only capability?

| measurement | result |
| --- | --- |
| dev split, all layers off | **1.64%** (13/795), hallucination 97.6% |
| authored prose, all layers off | **7.64%** (11/144), hallucination 92.4% |
| abstention accuracy | **0%** - it never declines |

Both numbers flatter it. Hand-reading the correct answers shows most are rubric
coincidences (*"The operating margin for 2015 was 810.01/20 = 678.50.00."*).
True capability is **at or below 1%** on arithmetic-shaped questions and
**under 8%** on prose where a loose rubric can be satisfied by fluent
approximation.

## 8. What is the complete system capability?

| split | what it is | system accuracy | hallucination |
| --- | --- | --- | --- |
| dev (795) | iterated against - describes this work | 100% | 0% |
| hidden (204) | spent twice, nothing tuned after | **96.08%** | 4.12% |
| blind prose (144) | authored, unseen by the code | **13.89%** | 82.64% |

The gap between 96% and 14% is the honest summary of this system: it is
excellent on questions whose shape it has machinery for, and poor on open-ended
financial language.

## 9. Remaining MODEL failures

Uniform. It produces grammatical, confident, wrong financial prose; it cannot do
arithmetic; it never abstains; it does not emit its own stop token. There is no
subset it handles reliably.

## 10. Remaining SYSTEM failures

On the blind prose set, 124 of 144:

* **96 questions with no deterministic coverage** fall to the model (6.25%
  correct). This is the whole problem, not a list of gaps.
* **Interpretation 0/15, levels 4-5 at 0%.** The 20 curated diagnostic patterns
  cover common divergences, and unseen prose asks about others.
* **4 prose questions reached the calculator** and were answered with
  arithmetic - a residual routing misfire.
* **Glossary 25% and retrieval 43.75% when they fire**, so even the curated
  layers are imprecise on questions phrased in unfamiliar ways.
* One residual case of a definition answering an explanation question ("Why is
  EBITDA criticised as a profit measure?"), where the predicate test passes on a
  shared word.

### The honesty/usefulness trade-off, measured

`withhold_unsupported_model_answers` (off by default) refuses when no
deterministic layer covers the question:

| blind prose, 144 items | default | withhold |
| --- | --- | --- |
| accuracy | 13.89% | 9.72% |
| hallucination | **82.64%** | **20.14%** |
| over-abstention | 3.47% | 70.14% |

Four points of accuracy buys a 62-point reduction in hallucination. For a
financial assistant I would take that trade, but it makes the product mostly
silent on prose, so the default is unchanged and the decision is yours.

## 11. Is additional training justified?

**Yes - but not of this model, and not as the next step.**

What the evidence supports, in order:

1. **The capability to target is financial concepts and interpretation prose.**
   Not "more training". Specifically: answering a conceptual or diagnostic
   question in language it has not seen, which is where 96 of 144 blind failures
   sit and where the deterministic layers cannot follow, because the space of
   phrasings is unbounded while a glossary is not.

2. **Training *this* checkpoint is not the way to get it.** Four pretraining
   sessions to 247,850 steps, two instruction-tuning runs and one distillation
   all left model-only accuracy at 1-2%. A 101M-parameter model on ~2B tokens
   producing 6% on prose will not reach useful accuracy through more of the
   same, and the measurement above shows decoding is not the limit either.

3. **The measured alternative already exists in this repository.**
   Qwen2.5-1.5B-Instruct scored 38/45 untuned and 40/45 with the project's
   finance LoRA, against Aivora's 1-5/45, on the same questions through the same
   scorer. If the goal is a working product, serving a larger open model behind
   the existing pipeline is the intervention with evidence behind it - the
   backend abstraction for it is already built and tested.

4. **If training Aivora is the goal for its own sake** - a from-scratch model as
   the point of the exercise - then the target is scale, not schedule: more
   parameters and more tokens, with the prose benchmark as the measure. Another
   run at 101M will reproduce 1.64%.

**Before any of that**, two things are worth more per hour spent: finishing the
blind authored set (144 of 1,000 - below that size, prose measurements move on
single items), and deciding the abstention trade-off above, which changes the
hallucination rate by 62 points with no training at all.

---

## What was done

| part | status |
| --- | --- |
| 1-4 inference validation, diagnostic, EOS, sampling | done - `scripts/validate_inference.py`, report written |
| 5-6 intent layer; intent outranks numbers | done - `Route.INTERPRETATION`/`CURRENT_DATA`, decided before figures |
| 7-8 extraction, extraction vs calculation | done - `financial_values.py`, normalized records with provenance |
| 9 calculator safety | done - declines with the missing input named |
| 10 current-data routing | done - matched by shape, not phrase lists |
| 11 retrieval precision | done - `supports_question()` on the question/answer pair |
| 12 knowledge-layer discipline | partial - entries are reusable concepts with aliases; the full schema (related concepts, applicable question types) is **not** built, and is recommended |
| 13-14 pattern selection and safety | done - scored by evidence, minimum two signal groups |
| 15 honest abstention | done, and extended with the withhold option |
| 16 regression tests | done - 80 checks in `tests/test_intent_routing.py` |
| 17 hidden test untouched | followed - no rule, entry, pattern or grader changed because of a hidden-split failure |
| 18 blind authored set | **partial - 144 of ~1,000 (13.7%)**, and it is not blind to its author |
| 19 system/model benchmark separation | done - `scripts/run_benchmarks.py` |
| 20 re-measurement with invocation shares | done - `reports/system_after_fixes.md` |
| 21 stop before training | followed |

**Test suites:** 80 routing, 47 benchmark, 26 response-pipeline, 25 security,
11 retrieval, 8 backends - all passing.

## Limitations of this report

* The development split is saturated at 100% and no longer measures anything;
  its prose questions are spent.
* The hidden split has now been run twice. Nothing was changed after either run,
  but it is no longer pristine and should be replaced before it is trusted again.
* The blind set was authored by Claude in the same session as the code it
  measures. It is a fair test of routing, parsing and pattern logic, and an
  upper bound for the glossary.
* Latency figures across runs are not comparable: some runs shared the machine
  with others.
* There is no KV cache, so every measurement involving the model cost ~8 s per
  question. That is the reason model-only runs take hours, and it is the obvious
  next performance fix.
