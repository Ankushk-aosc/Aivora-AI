# Aivora experiment plan: four strategies, with kill criteria

Step 12. **Nothing here has been run.** Each strategy states what it would prove,
what would count as failure, and how to get back. Success metrics are deliberately
absolute where the evidence supports an absolute floor, and comparative where it
does not.

A rule that applies to all four: **the measurement must come from a benchmark
this repository did not author.** The development split is saturated at 100%, the
hidden split has been run twice, and the blind set was written by the same author
as the glossary it tests. `evaluation/external.py` accepts an external benchmark
without code changes; supplying one is a prerequisite, not an optional extra.

---

## Strategy A - better pretraining only

**Hypothesis.** Aivora's weakness is a starved corpus and a training format that
could not teach question answering, not its parameter count. Feeding the same
architecture ~20 unique tokens per parameter, with document separators and a
consistent template, will move model-only accuracy materially above 1.64%.

**Why it is first.** It is the only experiment that changes one thing. The audits
establish that the completed run used 121M unique tokens (1.19/parameter) over
16.3 epochs, with no `<|endoftext|>` in the data at all - so no conclusion about
capacity is currently available.

| | |
| --- | --- |
| **data** | ~2.0B unique tokens, licence-clean, financial share ≥40%; separators written between documents; the missing categories added (ratio interpretation, valuation reasoning, risk analysis, abstention) |
| **compute** | ~53 GPU hours at measured throughput; ~2 weeks of free Kaggle quota |
| **experiment** | one run, V2-A configuration, capability evaluation every ~2,000 optimizer updates alongside loss |
| **success metric** | model-only accuracy on the external benchmark **≥15%** with hallucination **<70%**, and the model emits `<|endoftext|>` on at least 50% of completions (proving it learned answer boundaries) |
| **failure threshold** | model-only below **5%** after 1B tokens, or `<|endoftext|>` still unlearned - stop the run, do not finish it |
| **risks** | the corpus may not be assemblable at licence-clean volume; 12-hour session caps mean ~5 resumes, each a chance to lose outputs; a null result costs two weeks |
| **rollback** | the current checkpoint and all reports are frozen at tag `AIVORA_PRE_TRAINING_BASELINE`; a failed run changes nothing that is serving |

**What a null result would prove.** That 101.7M parameters is genuinely
insufficient - which is currently an assumption, and would become evidence.

---

## Strategy B - pretraining plus financial SFT

**Hypothesis.** Pretraining gives representations; supervised fine-tuning on
prompt-masked question/answer pairs with one template gives the *behaviour* of
answering. The two together clear the bar that pretraining alone does not.

| | |
| --- | --- |
| **data** | Strategy A's corpus, plus ≥50k instruction pairs with prompt masking: definitions, explanations, ratio interpretation, statement analysis, and **explicit abstention examples**, which no dataset here currently contains |
| **compute** | Strategy A's ~53 hours, plus ~2-4 hours of SFT |
| **experiment** | A's run, then SFT at 1e-5 to 5e-6 with early stopping on a held-out slice of the SFT data |
| **success metric** | model-only accuracy **≥25%**, abstention accuracy **>50%** (from ~0% today), hallucination **<50%** |
| **failure threshold** | overfitting inside 500 steps as in both previous attempts, with no improvement over A's checkpoint |
| **risks** | the two previous SFT runs on this project both overfitted within 500 steps on 800-38k rows; 50k *distinct* pairs is the mitigation and also the cost |
| **rollback** | keep A's checkpoint; SFT produces a separate artifact and is discarded on failure |

**Caution from the record.** SFT has been tried twice here and failed twice. What
is different this time is the base model (properly fed) and the data volume - if
neither changes, expect the same outcome.

---

## Strategy C - pretraining plus SFT plus teacher distillation

**Hypothesis.** A stronger teacher's outputs carry more signal per token than raw
text, so distillation lifts a small student past what its own corpus supports.

| | |
| --- | --- |
| **data** | B's data, plus ~25-50k teacher completions (Qwen2.5-1.5B-Instruct or larger) on financial prompts, **verified** before use - the previous attempt had no verification stage |
| **compute** | B's, plus ~4-8 hours to generate teacher outputs and ~3 hours to distil |
| **experiment** | B's pipeline, then distillation on teacher outputs that pass a correctness filter (the deterministic calculator can verify every numeric answer automatically) |
| **success metric** | **+10 points** of model-only accuracy over B's checkpoint on the external benchmark |
| **failure threshold** | style improves while accuracy does not - which is exactly what happened last time (2.22% to 8.89% on style, facts unchanged) |
| **risks** | distilling unverified teacher output teaches the teacher's errors; the verification stage is the part that was missing and is the part that costs effort |
| **rollback** | B's checkpoint remains the artifact |

---

## Strategy D - stronger base model plus financial LoRA

**Hypothesis.** The product goal is answered fastest by a base model that already
has language competence, with this project's pipeline in front of it.

| | |
| --- | --- |
| **data** | the SFT set from B, ~10-50M tokens; no pretraining corpus needed |
| **compute** | 2-6 GPU hours |
| **experiment** | LoRA on Qwen2.5-1.5B-Instruct (`training/hf_lora_sft.py` already exists and is tested), served through `HFBackend`, which is already built and covered by tests |
| **success metric** | system accuracy on the external benchmark **above the current 96.08%/13.89% pair**, specifically prose accuracy **>50%** where the current system scores 13.89% |
| **failure threshold** | no improvement over the untuned base, which already scores 38/45 against Aivora's 1-5/45 |
| **risks** | **it is not a from-scratch model** - this may defeat the project's purpose; the 1.5B weights do not fit this machine (8 GB RAM, 0.4 GB free), so all work is remote |
| **rollback** | trivial: the backend abstraction lets the served model be switched back in one call |

**Evidence already in hand.** Qwen2.5-1.5B-Instruct: 38/45 untuned, 40/45 with
this project's finance LoRA. Aivora: 1-5/45. Same questions, same scorer.

---

## Recommended order, and why

1. **Strategy A**, because it is the only experiment that isolates the variable
   the audits identified, and because a null result there is the evidence needed
   to justify anything larger.
2. **Strategy B** if A shows movement - SFT on a properly-fed base is a different
   experiment from SFT on a starved one.
3. **Strategy D in parallel**, if the goal is a working product rather than an
   answer about scratch training. It costs hours, not weeks, and its outcome does
   not depend on A.
4. **Strategy C last.** It is the most complex and its previous attempt failed in
   a way that is well understood: no verification.

**Before any of them:** an external benchmark. Every success metric above is
unmeasurable without one, and the three benchmarks this repository owns are all
spent or self-authored.

## What would change my recommendation

* If 2B licence-clean tokens turn out not to be assemblable, A becomes V2-B
  (smaller model, matched corpus) rather than being abandoned.
* If paid compute becomes available, V2-C moves from infeasible to the strongest
  candidate, and A becomes a pilot for it rather than an end in itself.
* If the goal is explicitly a product on a deadline, D moves first and A becomes
  research.
