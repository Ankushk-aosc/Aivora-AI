# SFT_EXPERIMENT_002 - the recipe works; the model is still weak

Step 6 sanity experiment. CPU only, 149 minutes, **no GPU time spent**.

**Verdict: all five gates pass. A GPU run is justified - modestly sized, and
gated again on the same probes.**

## What was run

| | |
| --- | --- |
| base | frozen `checkpoint_247850` (5/6 copy, 0/20 extraction) |
| dataset | `financial_sft_v2`, 12,159 rows, 227,892 answer tokens |
| train / val | 11,563 / 596, deterministic split by row-id hash |
| tokens | 1,028,429 |
| peak lr | **2e-5** (SFT_001 used 5e-5) |
| steps | 600, batch 4, block 224 |
| checkpoint selection | **best capability**, not lowest loss |
| stop condition | copy below 4/6 - never triggered |

## Trajectory

| step | val loss | copy | extraction (10 probes) |
| --- | --- | --- | --- |
| baseline | - | **5/6** | 0/20 |
| 100 | 2.1063 | 4/6 | 0/10 |
| 200 | 1.4925 | 4/6 | 0/10 |
| 300 | 1.2524 | 4/6 | 1/10 |
| 400 | 1.1753 | 4/6 | 0/10 |
| 500 | 1.0571 | **6/6** | 1/10 |
| 600 | **1.0184** | 4/6 | 2/10 |

## Full gate suite against the recorded baseline

| gate | baseline | SFT_002 | |
| --- | --- | --- | --- |
| A copy | 5/6 | 4/6 | PASS (at the floor) |
| B extraction, unseen contexts | 0/20 | **2/20 (10%)** | PASS |
| C generalisation, unseen wording and label spellings | 0/15 | **3/15 (20%)** | PASS |
| D calculation | 0/20 | 1/20 | PASS |
| D abstention | 0/5 | 1/5 | PASS |
| D over-abstention | 0/2 | **0/2** | PASS - does not refuse answerable questions |
| D interpretation | 0/3 | 2/3 | PASS |
| E hallucination | 100% | **75%** | PASS |
| EOS | 0/6 | **5/6 (83%)** | mean 9.7 tokens |

## Is it real, or small-sample luck?

Asked honestly, with Fisher's exact test:

| comparison | p (one-sided) | |
| --- | --- | --- |
| Gate B alone, 2/20 vs 0/20 | 0.244 | not significant |
| Gate C alone, 3/15 vs 0/15 | 0.112 | not significant |
| **B + C pooled, 5/35 vs 0/35** | **0.027** | significant at 0.05 |
| all unseen probes incl. calculation, 6/55 vs 0/55 | **0.014** | significant |

**No single gate is significant on its own.** Pooled across 55 unseen-context
probes the improvement is - six hits against zero, p = 0.014. That is the honest
statement: a real but small effect, detectable only when the probes are combined.

## What SFT_002 fixed, measured against SFT_001

| | SFT_001 | SFT_002 |
| --- | --- | --- |
| copy-from-context | 5/6 → **0/6** destroyed | 5/6 → **4/6**, reached 6/6 mid-run |
| extraction | 0% throughout | **0% → 10%** |
| generalisation to unseen wording | not measured | **20%** |
| hallucination | 89.6% (blind) | **75%** |
| what stopped the run | nothing - loss looked fine | copy floor, armed and watching |

The three dataset changes did their job: extraction's token share (10% → 37.6%),
the entropy floor that killed the canned-sentence shortcut, and the 1,200
finance-free copy rows plus real-text replay that protected the copy ability.

## Step 8 learning table

| Question | Result |
| --- | --- |
| What hypothesis were we testing? | That SFT_001 failed because of dataset design - token imbalance and canned answers - rather than because the model cannot learn these tasks |
| What changed? | Extraction token share 10% → 37.6%, an enforced distinct-answer floor, 1,200 copy rows, a real-text replay slice, peak lr 5e-5 → 2e-5, capability-based early stopping and checkpoint selection |
| Did the pipeline work? | Yes. 11,563 examples trained, gates ran at every checkpoint, the stop condition was armed throughout |
| Did model capability improve? | **Yes, slightly and measurably.** Pooled unseen probes 6/55 vs 0/55, p = 0.014 |
| Did model-only accuracy improve? | Extraction 0% → 10%, generalisation 0% → 20%, calculation 0% → 5% |
| Did extraction improve? | **Yes - first time in this project.** 0/20 → 2/20 on unseen contexts |
| Did copy ability improve/hold? | **Held.** 5/6 → 4/6, touching 6/6 mid-run. SFT_001 destroyed it entirely |
| Did hallucination improve? | **Yes. 100% → 75%** |
| Did reasoning improve? | Not separately measured in the gates; calculation moved 0 → 1/20 |
| Did interpretation improve? | Yes, 0/3 → 2/3, on three probes only |
| Did abstention improve? | Yes, 0/5 → 1/5, without over-refusing (0/2 answerable questions declined) |
| Did EOS behavior improve? | **Yes. 0/6 → 5/6**, mean output 9.7 tokens |
| Did validation loss change? | Yes, 2.1063 → 1.0184 |
| Did lower loss correspond to better capability? | **This time, partly yes** - but loss fell smoothly while copy oscillated 4/6, 6/6, 4/6, so loss still did not track capability step by step |
| What failed? | Nothing failed outright. Copy sits one point below baseline, and every effect is small in absolute terms: the model remains close to useless on its own |
| What did we learn? | The SFT_001 failure was a dataset-design failure, not a capability ceiling. Balancing by answer tokens, enforcing answer entropy, and protecting a behaviour with dedicated rows all work. Small probes cannot resolve small effects: pool them |
| What should change in the next experiment? | More probes per gate (20 → 60+) so single gates can resolve; more copy and replay data to lift copy above baseline rather than to the floor; longer training now that the recipe is safe |
| **Is another GPU run justified?** | **YES** - the Step 7 conditions are met: pipeline correct, labels and masking verified, extraction improving, copy preserved, and Gate C shows generalisation to wording never trained on rather than memorisation |

## Recommended GPU run, and its budget

Conditions in Step 7 are satisfied, so:

* **scope**: the same v2 recipe, 3-4 epochs over the 1.03M-token set instead of
  600 steps - roughly 12,000-16,000 steps at batch 16 on a T4
* **estimated cost**: **1.5-2.5 GPU hours**, inside the weekly quota with room
  for a repeat
* **non-negotiable**: gates run every 1,000 steps with the copy floor armed, and
  the checkpoint is selected on capability
* **what would make me stop it**: copy below 4/6, or extraction flat after two
  consecutive gate evaluations

**What this run is not.** It will not produce a good financial model. Extraction
at 10% and calculation at 5% are still failing grades. The case for spending the
GPU time is that the recipe now provably moves capability in the right direction
without destroying anything, and 600 CPU steps is too short to see how far it
goes.
