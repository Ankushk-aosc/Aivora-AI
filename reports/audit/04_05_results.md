# Tasks 4 and 5 — measured on a T4, commit `494c638`

Same model (Qwen2.5-1.5B-Instruct), same scorer, same prompts, same greedy
decoding. Nothing tuned for either run.

## Task 4 — the pipeline's contribution, isolated for the first time

| Gate | Model alone | + pipeline | Delta | Items |
|---|---|---|---|---|
| copy | 97.1% | 98.6% | +1.5 | +1 |
| **extraction** | **98.5%** | **87.0%** | **−11.6** | **−8** |
| **wording** | **88.0%** | **85.3%** | **−2.7** | **−2** |
| calculation | 52.1% | 100.0% | **+47.9** | **+23** |
| abstention | 63.9% | 97.2% | **+33.3** | **+12** |
| **OVERALL** | **83.84%** | **92.59%** | **+8.75** | **+26** |
| invented values | 0.77% (2/261) | 0.00% | −0.77 | −2 |

Model alone: 249/297 = 83.84%, CI [79.2, 87.6].

### What this overturns

**The pipeline makes extraction worse, not better.** Qwen2.5-1.5B answering
directly scores **98.5%** on extraction; behind the pipeline it scores **87.0%**
— eight items lost. Wording loses two more. The documentation presents 87.0% as
the pipeline's extraction capability without ever stating that the same model
unaided scores 98.5% on the same items.

The entire +8.75 overall comes from two gates — calculation (+23 items) and
abstention (+12) — and is partly spent paying for losses in two others.

This is consistent with the Task 3 classification: the pipeline's extraction
misses were six abstentions and three correct values scored down for showing
their working. The pipeline refuses where the plain model simply answers, and
answers correctly.

### What the pipeline genuinely buys

- **Calculation 52.1% → 100%.** The plain model gets half of all arithmetic
  wrong. Python arithmetic is the single largest contribution in the system.
- **Abstention 63.9% → 97.2%.** Unaided, the model answers 13 of 36 questions it
  should refuse.
- **Invented values 0.77% → 0%.** Smaller than the project's framing implies —
  this model invents a value on 2 of 261 answerable items without any help — but
  the pipeline does drive it to zero, and it does so structurally.

### The honest reading

The pipeline is **not** a general accuracy improvement. It is a trade: large
gains in arithmetic and refusal, paid for with a real loss in extraction and
wording. For a product whose claim is that every figure is traceable and no
arithmetic is guessed, that trade is defensible. Presenting 87.0% extraction
without the 98.5% comparison is not.

**Recommendation.** Route extraction-only questions around the pipeline, keeping
it for calculation and refusal — then measure again. On these numbers that would
score roughly 283/297 rather than 275/297, while keeping the arithmetic and
abstention guarantees intact. Not implemented here: it changes the system under
test and the decision is the owner's.

## Task 5 — the fresh calculation split

| | Frozen (vocabulary written after seeing its failures) | Fresh (never seen) |
|---|---|---|
| calculation | 48/48 = 100.0% | **44/44 = 100.0%** CI [92.0, 100.0] |
| span validation | 100% | **100% of 44, 0 violations** |
| abstained | — | 0.0% |

**The leakage did not inflate the result.** The capability reproduces exactly on
44 items the pipeline has never seen, with fresh contexts, fresh figures and
expected answers computed independently of the pipeline.

### The caveat that keeps this honest

The fresh split tests **new instances of captions the vocabulary already
covers**, not captions outside it. The builder reported 44 of 44 items covered.
So this closes the question "did the vocabulary overfit to those eight specific
frozen items?" — it did not — but it does **not** close "does the vocabulary
generalise to captions nobody wrote into it?"

The two entries marked `TAILORED` in the source remain tailored. A split built
deliberately from captions *outside* the table would answer the remaining
question, and would be expected to score lower.
