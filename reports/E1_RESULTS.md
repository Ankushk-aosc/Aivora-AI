# E1: did formatting fix capability? No - it fixed stopping.

Format-controlled comparison. Two arms, identical corpus, seed, architecture,
optimizer, schedule, batch and token budget; the only difference is three
data-format flags. Both trained from scratch for 241,999,872 token passes
(29,541 steps, 7,385 optimizer updates) on Kaggle.

**Outcome: CASE B.** Arm 2 changed behaviour decisively and did not improve
capability.

## The comparison

| metric | frozen baseline | arm 1 (control) | **arm 2 (treated)** |
| --- | --- | --- | --- |
| document separators in corpus | 0 | 0 | **433,850** |
| masked prompt tokens | 0 | 0 | **2,760,282** |
| train loss | - | 3.7901 | **3.6944** |
| validation loss (final / best) | 4.4258 / 4.3664 | 4.2264 / 4.1365 | **4.1305 / 4.0588** |
| **EOS emitted** | **0/204** | **0/10** | **10/10 (100%)** |
| mean generated tokens | 39.9 (hit cap) | 128.0 (hit cap) | **38.9 (stopped itself)** |
| model-only dev (200) | 1.64%* | 3.5% | 0.5% |
| model-only blind prose (144) | 7.64% | 10.42% | 8.33% |
| dev hallucination | ~97.6%* | 95.83% | 99.48% |
| blind hallucination | 92.36% | 89.58% | 91.67% |
| extraction | 0/25 | 0% | **0%** |
| calculation (ratios + statements) | ~0.7%* | ~3.9% | 0% |
| interpretation | 0%* | - | 0% |
| reasoning | 0/25 | 0% | 0% |
| abstention accuracy | 0% | 0% | **0%** |
| copy-from-context | **5/6** | not measured** | not measured** |

\* over the full 795-item dev split.
\*\* the arm checkpoints are 1.2 GB each and live in Kaggle's output; the copy
probe needs the weights locally, and disk is at 8.5 GB free. Recorded as not
measured rather than guessed.

## What the three format changes actually did

**They worked, precisely and only as specified.** `<|endoftext|>` appeared 433,850
times in arm 2's corpus and the model went from never emitting it to emitting it
in 10 of 10 probes, ending answers at 38.9 tokens instead of running to the 128
cap. Validation loss improved too (4.0588 vs 4.1365). Prompt masking applied to
2.76M tokens.

**Capability did not follow.** Extraction stayed at 0%, reasoning at 0%,
abstention at 0%. Arm 2's dev accuracy is *lower* than arm 1's.

## The caveat I am testing rather than assuming

Arm 1 rambles to 128 tokens; arm 2 stops at 39. The scorer accepts a matching
number or keyword anywhere in the output, so a longer answer buys more chances of
an accidental match. Part of arm 1's apparent advantage may therefore be
verbosity rather than capability.

A length-controlled test is running: one fixed model, one set of questions, only
`max_new_tokens` differing (32 vs 128). If accuracy rises with length for a model
whose weights never change, then the arm1-over-arm2 gap is partly an artifact of
the metric, and the honest conclusion is that **neither arm has capability** -
which is what every other number here already says.

This does not change the CASE B verdict either way.

## Step 8 learning table

| Question | Result |
| --- | --- |
| What hypothesis were we testing? | That Aivora is weak because it was trained to continue text rather than to answer and stop - no document separators, no template, no prompt masking |
| What changed? | Only those three data-format flags. Corpus, seed, architecture, optimizer, schedule, batch and token budget identical |
| Did the pipeline work? | Yes. 433,850 separators and 2,760,282 masked tokens in arm 2's corpus; arm 1 verified to have none of either |
| Did model capability improve? | **No** |
| Did model-only accuracy improve? | No. dev 3.5% -> 0.5%, blind 10.42% -> 8.33% |
| Did extraction improve? | **No. 0% in both arms** |
| Did copy ability improve/hold? | Not measured on the arm checkpoints (1.2 GB each, remote). Baseline is 5/6 |
| Did hallucination improve? | No. dev 95.83% -> 99.48%, blind 89.58% -> 91.67% |
| Did reasoning improve? | No. 0% in both arms |
| Did interpretation improve? | No. 0% in arm 2 |
| Did abstention improve? | No. 0% in both arms. SFT_001 is the only run that ever moved it (62.5%) |
| Did EOS behavior improve? | **Yes, completely. 0% -> 100%, and output length halved because the model now stops on its own** |
| Did validation loss change? | Yes, improved: 4.1365 -> 4.0588 best |
| Did lower loss correspond to better capability? | **No.** Arm 2 has the better loss and the worse accuracy. Second time this has been demonstrated in this project |
| What failed? | The hypothesis that formatting was the binding constraint. It was a real defect and fixing it did not produce capability |
| What did we learn? | Formatting is necessary and nowhere near sufficient. A model can learn the shape of an answer - start, stop, length - without learning to produce the content. Also: the scorer rewards verbose output, so length must be controlled when comparing models |
| What should change in the next experiment? | Keep arm 2's format as the default for all future training. Stop testing format. Move to the SFT recipe, where SFT_001 showed the failure is dataset design - token imbalance and canned answers - and where abstention has already been moved once |
| **Is another GPU run justified?** | **NO - not yet.** The next run must be the CPU sanity experiment for SFT_002, gated on copy retention and extraction improvement. GPU time is justified only if that passes |

## Decisions taken from this result

1. **Arm 2's format becomes the default** for every future training run: separators
   written, one template used in training and at inference, prompts masked. It is
   the only intervention so far that changed a behaviour completely and it costs
   nothing.
2. **Format is now a closed question.** No further experiment will vary it.
3. **Neither arm checkpoint is promoted.** The frozen baseline remains the
   reference, as does its tag.
4. **The next experiment is SFT_002 on CPU**, small, gated on capability rather
   than loss. No GPU until it passes.
5. **Output length must be controlled in every future model comparison**, because
   the scorer pays for verbosity.
