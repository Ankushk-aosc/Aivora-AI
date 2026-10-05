# Phase 4 — the tool pipeline on the frozen held-out set

**Configuration: Qwen2.5-1.5B-Instruct, 0-shot, behind the tool pipeline.** Your
choice, 2026-10-05. The model names a field and copies a value with its source
span; Python validates the span, picks the formula and does all arithmetic; rules
abstain when validation fails or an operand is missing.

Date: 2026-10-05 · Kaggle T4 · repo `240ccbd` · model run **once per split**
Raw: [`phase4_tool_pipeline.json`](phase4_tool_pipeline.json), per-item outputs in
`reports/heldout/Qwen_Qwen2.5-1.5B-Instruct_pipeline_{selection,frozen}*.json`,
kernel log [`kaggle_phase4.log`](kaggle_phase4.log)

**Status: PROVISIONAL.** Your 30 hand labels are still outstanding, so κ is
unproven. The scorer changed during this phase — see §3, which you should read
before §2.

---

## 1. Preflight — all gates passed before anything was scored

The run refuses to produce a number if any of these fail. From the kernel log:

| check | result |
|---|---|
| selection set vs manifest | VERIFIED, `09a363bfb8619ec3…` |
| frozen set vs manifest | VERIFIED, `0e50dbfd51c43c40…` |
| pipeline guarantee tests | 33/33 passed |
| stub wire-check on real items | 100% abstention, 0 invented, 0 span violations |
| GPU | Tesla T4, 15.6 GB |

The frozen hash matched **exactly** between the T4 and this machine, which is the
portable content hash from last commit doing its job — the thing that produced a
false alarm in Phase 3 now agrees across platforms.

## 2. Results

**Frozen split, n=313, scored once.** Re-scored figures (see §3); the as-run
figures are in brackets.

| gate | k/n | accuracy | 95% CI |
|---|---|---|---|
| copy | 68/69 | 98.6% | [92.2, 99.7] |
| extraction | 60/69 | 87.0% | [77.0, 93.0] |
| wording | 64/75 | 85.3% | [75.6, 91.6] |
| calculation | 40/48 | **83.3%** (was 68.8%) | [70.4, 91.3] |
| abstention | 35/36 | 97.2% | [85.8, 99.5] |
| **overall** | **267/297** | **89.9%** (was 87.5%) | [85.9, 92.9] |
| invented values | 0/261 | **0.0%** | — |
| source-span validation | 251/251 | **100%** | — |
| over-refusal | 4/16 | 25.0% | — |

Selection split, n=61: overall 93.3%, copy 100%, extraction 93.3%, wording 82.4%,
calculation 100% (6/6), abstention 100%, invented 0%, span 100%, over-refusal 0/1.

Cost: 1.23 LLM calls per item, 2.27 s per item. Components: extraction 210,
calculation 101, current-data rule 2.

### Against the thresholds you set

| rule (EXPERIMENT_RULES_v2 §1) | measured | verdict |
|---|---|---|
| every extracted value has a validated source span — 100% | 251/251 = 100% | **PASS** |
| a system failing the hallucination gate is not shipped | 0/261 invented | **PASS** |
| calculation ≥ 95% for the full pipeline | 83.3%, CI [70.4, 91.3] | **FAIL** |

**The pipeline does not meet your calculation threshold and therefore does not
ship as it stands.** The interval's upper bound is 91.3%, below 95%, so this is
not a sample-size artifact — 48 items are enough to say it falls short. §5 says
why, and the cause is fixable without training.

The two ship-blocking rules both hold: not one answer in 261 carried an invented
number, and not one stated value lacked a validated span.

## 3. The scorer changed during this phase — read this before trusting §2

While diagnosing calculation I found the pipeline's seven `wrong_value` failures
were **arithmetically correct**: expected 2.5, answered `2.50x (revenue =
6,000.00, total assets = 2,400.00)`. 6,000/2,400 is 2.5.

The scorer's `numbers_in()` could not read `2.50x`. Worse, it did not reject the
token — it **backtracked and returned 2.0**, silently truncating. That is not a
near miss: it can score a *wrong* answer **correct** (expected 2.0, answered
"2.50x"). The boundary rule that caused it was added to stop numbers matching
inside identifiers like `Q3`, which is correct intent with a buggy
implementation. *(measured)*

Fixed so a token that cannot be read whole is rejected outright — the discipline
`pipeline/tool_pipeline.py numeric()` already applied to `2,60.65` — and so a
ratio's `x` unit is read as a unit. `tests/test_numbers_in.py` covers 19 cases
including the false-positive path, version strings, `Q3`, `FY2024` and malformed
separators.

**I must flag the protocol risk plainly: I found this bug because the system I
had just run scored badly, and the fix helps only that system.** That is exactly
the shape of a motivated scorer change, so:

- The defect is real independent of who benefits — silent truncation that can
  score wrong answers correct is a bug under any system's outputs.
- **Every saved system was re-scored**, not just the pipeline
  (`scripts/rescore_saved.py`, output
  [`rescore_2026-10-05.json`](rescore_2026-10-05.json)). Result: 2 of 11 moved,
  both the pipeline (frozen +2.36pp, selection +1.67pp). **No Aivora score
  changed at all**, and nothing anywhere got worse.
- Both numbers are reported above, and the as-run raw outputs are preserved
  unmodified alongside the re-scored ones.

This is what preserving raw outputs is for: the model ran once, its answers are
fixed, and only the judgement of them changed — no re-run, no GPU, no second look
at the frozen set.

## 4. Against Aivora — the paired comparison, finally valid

Both Aivora checkpoints now have selection-split scores, so the exact McNemar
test applies on **identical items** for the first time. The earlier attempts
failed only because I ran them under a 30-minute background cap; raised to two
hours, both finished in about ten minutes. *(measured)*

**Tool pipeline vs Aivora SFT_002**, 61 shared items:

| gate | pipeline | SFT_002 | pipeline only | SFT_002 only | p |
|---|---|---|---|---|---|
| copy | 15 | 8 | 7 | 0 | 0.0156 |
| extraction | 14 | 1 | 13 | 0 | 0.0002 |
| wording | 14 | 2 | 12 | 0 | 0.0005 |
| calculation | 6 | 0 | 6 | 0 | 0.0312 |
| abstention | 8 | 2 | 6 | 0 | 0.0312 |
| **all** | **57** | **13** | **44** | **0** | **<0.0001** |

44 items better, **zero** worse. Every gate significant.

**SFT_002 vs the Aivora baseline** (paired): 10 better, 0 worse, p=0.0020; copy
p=0.0312. SFT_002 is a genuine improvement over the baseline — the first
statistically defensible Aivora claim in this project. It is also nowhere near
enough.

### The finding that settles the "too small" question

**SmolLM2-135M-Instruct beats Aivora SFT_002 on identical items** — 14 better, 4
worse, p=0.0309 overall; against the baseline, 20 better, 0 worse, p<0.0001.

135M parameters versus Aivora's 101.7M. **Parameter count is approximately
matched, and the pre-trained model still wins.** So the deficit is not capacity,
and the original brief's "too small" framing is only half right: Aivora saw
120.9M unique tokens at 1.19 tokens per parameter, against roughly 20 for
compute-optimal. It is **under-trained**, not undersized — and no amount of
tuning a 1.19-ratio model closes a gap that a same-size model with proper
pre-training already clears. *(measured, parameter-matched, paired)*

## 5. Why calculation falls short — one cause, no training needed

All 8 remaining calculation failures are the **same failure**, and none is an
arithmetic error. Every one is `incorrect_abstention`: the pipeline refused.

| item | question | labels actually in the context |
|---|---|---|
| calculation_001 | return on equity | "Profit attributable to owners", "Total owners' funds" |
| calculation_004 | current ratio | "Assets falling due within one year", … |
| calculation_006 | working capital | "Stocks and debtors and cash", "Creditors due within one year" |
| calculation_011 | operating margin | "Trading profit", "Sales for the half year" |
| calculation_015 | asset turnover | "Turnover for the year", "Total resources employed" |
| calculation_016 | gearing ratio | "Borrowings", "Owners' funds" |
| calculation_023 | free cash flow | "Net cash inflow from trading", "Payments for fixed assets" |
| calculation_026 | interest cover | "Profit before interest and tax", "Finance charges" |

`FORMULAS` in `pipeline/tool_pipeline.py` demands literal operand labels —
`total debt`, `shareholders' equity`, `revenue`, `operating profit`, `interest
expense`. The contexts use ordinary British accounting synonyms. The model
truthfully reports that "total debt" is not present, and the pipeline correctly
abstains. **Every component behaved as designed; the operand vocabulary is too
narrow.** *(measured)*

This is the honest trade the pipeline makes, and it is the right one: it refused
8 items rather than inventing 8 numbers, which is why invented values are 0/261.
But over-refusal is a failure too, and here it is the *only* thing standing
between 83.3% and your 95% bar.

**Recommended fix — a synonym table for operand labels. Rules and data only, no
training.** All 8 are plain accounting synonyms, and the three-condition span
validation stays exactly as it is, so the hallucination and span guarantees are
untouched. A fixed pipeline is a new configuration and gets its own single frozen
run. If all 8 resolve, calculation reaches 48/48; even 6 of 8 gives 95.8%, over
the bar.

I have not made this change. It alters the system under test, and you asked to
review before the phase that follows.

## 6. Two defects in my own eval set

**Duplicated items.** 313 frozen items contain only 311 distinct question+context
pairs. `calculation_016` and `abstention_002` are the same question and context
in different gates, as are `calculation_026` and `abstention_027`. They are not
contradictory — the abstention entries are answerable controls, so both want the
item answered — but the same underlying question is scored twice, and the
over-refusal count overlaps the calculation failures: 2 of the 4 over-refusals
are the same refusals already counted above. The frozen set must not be rebuilt
now, so this is disclosed rather than fixed, and the over-refusal figure should
be read as 4/16 with that overlap in mind. *(measured)*

**Over-refusal n is tiny.** 16 answerable controls on the frozen split and 1 on
selection. 25% is 4 items. Any future eval set needs far more controls for this
to be a real measurement.

## 7. What this establishes

*(measured)* Behind a span-validating, Python-calculating pipeline, a 1.5B
pre-trained instruct model scores **89.9%** on the frozen held-out set with
**zero invented values in 261 answerable items** and **100% source-span
validation on 251 stated values**. It beats the from-scratch model by 44 items to
0 on matched items. The approach works.

*(measured)* It misses your 95% calculation bar at 83.3%, from a single
identified cause — a narrow operand vocabulary — that needs no training to fix.

*(not established)* A ship decision. κ is unproven until your hand labels arrive,
and the rest of the §1 thresholds are still `<your numbers>`.

*(not established)* Anything about LoRA (Phase 5), which the brief makes
conditional on the tool pipeline being insufficient. On this evidence the
pipeline is close to sufficient and the gap is a lookup table, not a learning
problem.

## 8. Waiting on you

1. **The 30 hand labels** — `reports/scorer_validation/sample_for_labelling.jsonl`.
   Note the scorer changed in §3, so please label against the current
   `scripts/eval_heldout.py`.
2. **The remaining threshold numbers** — §1 of `EXPERIMENT_RULES_v2.md` still
   reads `<your numbers>`. Calculation (≥95%), span validation (100%) and the
   hallucination gate are set and judged above.
3. **Approval for the synonym fix** (§5) and its one frozen re-run as a new
   configuration.

Still outstanding from earlier sessions: three Hugging Face tokens pasted in an
earlier session should be revoked at
https://huggingface.co/settings/tokens — nothing here uses them.
