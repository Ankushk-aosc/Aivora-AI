# Experiment rules - pre-registered

Committed **before** any model is evaluated or trained under this plan. Nothing
in this file may be edited once the first system has been scored on the frozen
set. If a rule turns out to be wrong, the correction goes in the final report as
a deviation, with the reason; it does not get quietly rewritten here.

Frozen evaluation: `data/eval_heldout/heldout_frozen.jsonl`, 313 items,
sha256 `e02c3dc3b8d67e85...`. Selection split: `selection.jsonl`, 61 items.

---

## 1. Hypothesis

A pre-trained instruct model of 0.5-2B parameters, wrapped in a tool pipeline
that (a) forces JSON extraction validated against a source span and (b) performs
all arithmetic in Python, will reach usable financial extraction and Q&A
accuracy **without any fine-tuning**, and will beat the from-scratch Aivora model
by a wide margin.

The secondary hypothesis, tested only if the first falls short: LoRA fine-tuning
on the existing SFT data closes the remaining gap.

## 2. One variable per experiment

| experiment | the single variable | everything else held fixed |
| --- | --- | --- |
| E-A zero-shot base models | which base model | prompt, decoding, scorer, eval set |
| E-B few-shot | number of in-context examples (0 vs 3) | base model, prompt skeleton, decoding |
| E-C tool pipeline | JSON + span validation + Python arithmetic, on or off | base model, decoding, eval set |
| E-D LoRA | fine-tuned adapter, on or off | base model, tool pipeline, decoding |

A run that changes two of these at once is not reported as evidence for either.

## 3. Decoding and prompting, fixed for every system

* greedy decoding, `temperature = 0`, `top_p = 1`, no sampling
* `max_new_tokens = 160` for extraction and abstention, 256 for calculation
* the same system prompt and the same user template for every model
* the same scoring code, `scripts/eval_heldout.py`, for every system
* one seed (1234) - single-seed variance is a stated limitation, not a hidden one

## 4. Success thresholds

Measured on the **frozen** set, per gate, as exact-match k/n with a 95% Wilson
interval.

| gate | threshold to call the system usable | rationale |
| --- | --- | --- |
| extraction | **>= 60%** exact match | below this a human re-checks every answer, so the system saves no work |
| wording generalisation | **>= 50%** exact match | phrasing robustness, held lower than extraction on purpose |
| copy | **>= 80%** | copying is the easiest sub-task; failure here invalidates the rest |
| calculation | **>= 80%** correct to tolerance | arithmetic is done in Python, so anything below this is a pipeline defect, not a model limit |
| abstention | **>= 70%** correct refusals **and** over-refusal **<= 15%** | a system that refuses everything is not abstaining, it is useless |
| hallucination (invented values on answerable items) | **<= 10%** | the headline safety number |

**Overall "ship" bar:** every threshold above met simultaneously on the frozen
set, plus a failure analysis showing no single category above 40% of remaining
errors.

## 5. Stop rules

* **Stop a training run** if the copy gate on the selection split falls below
  80% of the pre-training value at any checkpoint.
* **Stop a training run** if the target gate (extraction) is flat or worse across
  two consecutive checkpoint evaluations.
* **Do not start** a GPU run unless the preceding CPU or zero-shot experiment has
  already shown movement in the target gate.
* **Abandon the fine-tuning branch entirely** if the tool pipeline alone meets
  the thresholds in section 4 - there is nothing left to buy.
* **Abandon a base model** if its zero-shot copy gate is below 50%: it cannot
  follow the instruction format well enough for the pipeline to help.

## 6. Statistical test

* Per gate, the comparison between two systems is a **two-proportion test**,
  reported as the difference with a 95% confidence interval; where counts are
  small, **Fisher's exact test**, one-sided, with the p value stated.
* A gate difference is called **real** only at **p < 0.05**. Anything else is
  reported as "not distinguishable at this sample size", not as an improvement.
* Where gates are pooled to gain power, the pooling is stated explicitly and the
  per-gate numbers are still shown. (This was necessary once already: SFT_002's
  improvement was significant only pooled, p = 0.014, and not on any single gate.)
* No result is reported from training loss. Loss is recorded and ignored for
  capability claims.

## 7. What counts as a correct answer

* **extraction / copy / wording**: the expected string appears in the model's
  answer, or the parsed numeric value matches within 0.01. The answer must not
  also contain a second, contradictory value.
* **calculation**: numeric match within 0.05 absolute or 0.5% relative,
  whichever is larger.
* **abstention**: the answer expresses refusal (marker list in
  `evaluation/financial_metrics.py`) for items whose expected value is null; and
  does **not** express refusal for the answerable controls.
* **hallucination**: an answerable item answered with a number that appears
  neither in the context nor as a correct computation from it.

## 8. Integrity rules

* The frozen set is never used to train, to choose a checkpoint, or to tune a
  prompt. Checkpoint choice uses `selection.jsonl` only.
* Every extracted value must be traceable to a span in the context; a value that
  cannot be located is rejected and becomes an abstention.
* The LLM never performs arithmetic that Python can do.
* Raw outputs from every run are saved, including failed runs.
* All Aivora artifacts - the baseline tag, SFT_001, SFT_002, their datasets and
  results - are preserved regardless of outcome.

## 9. Reference rows to be recorded before anything new is built

So later systems are compared against something, not against nothing:

* Aivora frozen baseline (`checkpoint_247850`), model-only
* Aivora SFT_002, model-only

Both scored on the frozen set with the same scorer as every other system.
Expectation, stated in advance so it cannot be rationalised later: **both will
score near zero on extraction and calculation**, based on their gate results
(extraction 0/20 and 2/20 respectively). If either scores materially higher on
the frozen set than on its own gates, the scorer is wrong and must be
investigated before any other result is trusted.
