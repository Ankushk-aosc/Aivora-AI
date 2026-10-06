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

---

## Task 5b — captions the vocabulary does NOT have

44 items, 34 distinct captions, every one verified absent from `SYNONYMS`.

| | Result |
|---|---|
| accuracy | **5/44 = 11.4%** CI [5.0, 24.0] |
| abstained | 88.6% |
| **wrong values** | **0** |
| invented values | **0/44** |
| span violations | **0** (100% of the 5 answered) |

### The same capability, measured three ways

| Measurement | Score |
|---|---|
| Frozen set — captions written *after* seeing its failures | 48/48 = **100.0%** |
| Fresh contexts, captions the vocabulary **has** | 44/44 = **100.0%** |
| Fresh contexts, captions the vocabulary **lacks** | 5/44 = **11.4%** |

### What this settles

**The calculation capability is the caption table, not the model.** When the
table knows the wording, the pipeline is perfect. When it does not, the model
bridges the gap 5 times in 44 — it does not generalise. The 147 captions are
carrying essentially the whole result.

This reframes the earlier synonym fix. Operand resolution took calculation from
83.3% to 100% on the frozen set, and that was read as a capability improvement.
It was coverage. The same change on wording outside the table buys nothing.

It also settles the leakage question properly. The first fresh split showed the
vocabulary did not overfit to the eight items it was written from. This one
shows why that was never the real risk: the risk is that **every** reported
calculation figure is conditional on the filing using wording someone already
entered, and no reported figure has ever tested otherwise.

### What is genuinely reassuring

**Zero wrong values. Zero invented values. Zero span violations.** 39 of 39
failures are refusals. This is the safe failure mode, and it is the one the
architecture was built for: faced with wording it does not know, the system
declines rather than picking a plausible-looking wrong line. Task 3 found two
wrong-line cases on the frozen set, so this was a real possibility, not a
foregone conclusion.

### What it means for a real filing

A real 10-K or ASX report will use wording outside the table. On this evidence
the system will **refuse** those calculations rather than answer them wrongly —
safe, and much less useful than 100% suggests. The honest statement of the
capability is:

> Calculation is 100% accurate on statement wordings the operand vocabulary
> covers, and refuses most of what it does not cover. Coverage, not arithmetic,
> is the limit.

**Recommendation.** Stop reporting calculation as a single number. Report it as
coverage × accuracy-within-coverage, and measure coverage against real filings —
which is Task 6, still blocked on a filing.
