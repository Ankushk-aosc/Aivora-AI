# Knowledge-layer fixes: dev 100%, hidden test 94.61%

Companion to the generated `reports/improvement_report.md`. Covers the retrieval
precision guard, the diagnostic-pattern layer and the 33 new glossary entries.
No model weights were touched at any point.

## The numbers

| metric (full_pipeline) | baseline | after routing/parser | after knowledge layer |
| --- | --- | --- | --- |
| dev overall | 58.36% | 95.85% | **100.0%** (795/795) |
| dev hallucination | 27.67% | 4.35% | **0.0%** |
| dev over-abstention | 15.28% | 0.0% | **0.0%** |
| dev interpretation | 25.0% | 25.0% | **100%** |
| dev reporting | 23.08% | 23.08% | **100%** |
| **hidden test overall** | not run | not run | **94.61%** (193/204) |

## Read the dev 100% as saturation, not success

I chose the glossary entries and diagnostic patterns after reading which dev
questions failed. The entries are real definitions rather than rubric keywords,
and the patterns cover the diagnostics an analyst actually checks - but dev has
stopped being a measurement of this work and has become a description of it.
A split you iterate against cannot also score you.

So the hidden split was spent, once, with nothing changed after seeing it. Its
result is the number worth quoting:

| hidden test, by item kind | accuracy |
| --- | --- |
| generated (templates, unseen values) | **100.0%** (174/174) |
| authored prose | **63.33%** (19/30) |

That contrast is the finding. **The code generalises; the curated content does
not.** Extraction, ratios, statement arithmetic, valuation, corporate finance
and multi-step reasoning are at 100% on values never seen - the routing, the
nearest-label parser and the calculator work on the shape of the problem, not on
memorised instances. Prose accuracy fell from 100% on dev to 63% on unseen
questions, which is the size of the dev-fitting effect, measured rather than
guessed.

One caveat that weakens the hidden split as a blind test: **I authored its prose
questions myself** earlier in this session. It is a fair test of the code and of
the generated items; for the authored items I had prior exposure, so 63.33% may
still flatter the prose layer.

## What the hidden split found that dev could not

Three defects, all introduced by these fixes and invisible on dev. **I have not
fixed them**, because fixing failures seen on the hidden split is how a hidden
split stops being one. They should be reproduced with new dev items first.

**1. The numeric-signal rule is too eager on interpretation questions.**
"Two or more labelled figures" was added so CAGR questions would route to the
calculator. But an interpretation question carrying figures now goes there too:

> "Debt/EBITDA rose from 2x to 5x in a year. What does that imply?"
> → *Answer: Debt/EBITDA … Inputs: total_debt = 5.00, ebitda = …*
>
> "Days sales outstanding rose from 30 to 75. What happened?"
> → *I could not identify a complete calculation …*

Both should have reached the diagnostic patterns, which contain the right answer.
The fix is to check interpretive form before the numeric signal, not after.

**2. Diagnostic patterns are matched in declaration order, so a general pattern
can shadow a specific one.** "Operating margin improved while gross margin fell"
matched the general gross-vs-operating pattern rather than the specific one
written for exactly that divergence. Patterns need ordering by specificity, or
scoring by number of matched signals rather than first hit.

**3. Allowing a glossary entry to answer an interpretive question admits
near-misses.** This change fixed "Why is depreciation called a non-cash expense?"
on dev, where the depreciation entry says precisely that. On unseen questions it
lets a definition stand in for an explanation:

> "Why do interest rate rises usually reduce equity valuations?" → the definition
> of *interest rate*
> "What does a high inventory turnover suggest?" → the definition of *inventory
> turnover*
> "Why is EBITDA criticised as a profit measure?" → the definition of *EBITDA*

The entry is about the right subject, so the focus check passes and the kind
check passes. What is missing is a check that the entry answers the *question
form*: "why" and "what does X suggest" need an explanation, and an entry that
only defines X should defer. Five of the eleven hidden failures are this.

Two further failures are plain coverage gaps (cash expense vs accrual, why
identical cash flows can yield different profits), and one is a rubric wanting
both halves of a comparison where the entry gave one.

## Judgement calls I made, recorded because they could be argued the other way

* **The interest-coverage pattern was reworded to say "default risk rises"**
  after a rubric rejected it. The statement is true and is the implication a
  finance answer should give - but the edit was prompted by a grader, which is
  tuning text against a metric.
* **"time value of money" is excluded from semantic retrieval.** Its name is made
  of words common across finance, and it retrieved for a question about inflation
  at 0.545, above legitimate paraphrase matches at 0.455-0.488. No threshold
  separates them, so the entry stays for literal lookup only.
* **A retrieval test's expectation was left as "must decline"** rather than
  changed to accept the inflation entry, because the index still ranks the wrong
  entry higher; the test documents a real limitation instead of being relaxed.
* **The focus check is not applied to paraphrase questions.** Requiring a named
  subject disabled exactly what semantic retrieval is for, so it applies only
  where the question names its subject.

## Where this leaves the project

* Deterministic components: **100% on unseen generated items.** Extraction,
  parsing, routing, calculator and abstention are done, on the evidence
  available.
* Prose: **63% on unseen authored questions**, limited by knowledge coverage and
  by answering the question's form rather than its topic - not by the model,
  which still stands at 0.88% alone.
* The benchmark needs more authored items before prose work can be measured
  honestly: 30 hidden prose questions is too thin a target, and dev's are now
  spent. That is the prerequisite for any judgement about SFT or distillation,
  which remain unjustified on this evidence.
