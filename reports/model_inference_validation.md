# Aivora inference validation

Parts 1-4 of the pre-training brief: is the 0.88% model-only result a model
problem or an inference problem? Both, in a measurable proportion. Produced by
`scripts/validate_inference.py`; raw records in
`reports/model_inference_validation_before.json` and `..._after.json`.

No weights were modified.

## 1. Checkpoint and tokenizer identity

| check | result |
| --- | --- |
| checkpoint | `checkpoints/final/checkpoint_247850.pt` |
| step recorded inside the file | **247,850** - matches the expected completed step |
| sha256 | `6e9e50ca59e5…` |
| not an earlier checkpoint | confirmed: `checkpoint_237362` and `checkpoint_191120` exist in `checkpoints/previous_checkpoints/` and were not loaded |
| sidecar config used | yes - `load_model_for_inference` reads `checkpoint_247850.json` and builds the config the checkpoint was trained with, not a default |
| parameters | 101,723,264 |
| tokenizer | GPT-2 BPE via `tiktoken`, vocab 50,257 |
| vocab vs model | matches (`vocab_size` 50257 in the model config) |
| special tokens | `<|endoftext|>` = 50256 only. GPT-2 BPE has no separate BOS or PAD, and training concatenated documents separated by this token, so it is the only stop token the model could have learned |
| dtype / device | float32 on CPU |
| KV cache | **none** - every step recomputes the full forward pass, which is why generation costs ~8 s for 40 tokens |
| context length | 1024, and the backend truncates the prompt to `block_size - max_new_tokens` |

## 2. The defect: the repetition detector counted prompt tokens

`models/model.py` compared its repetition window against prompt **and**
generated tokens (`idx[:, -repetition_window:]`). An answer that begins by
naming the term it was asked about therefore repeats a bigram from the
*question*, and generation stopped. With token ids printed:

```
prompt   "Question: What is EBITDA?\nAnswer:"
         [..., 412, 26094, 5631, 30, 198, 33706, 25]     " E","BIT","DA","?","\n","Answer",":"
generated [412, 26094]                                    " E","BIT"
stop      repeated_bigram, step 2
window   [25, 1867, 318, 412, 26094, 5631, 30, 198, 33706, 25, 412, 26094]
                        ^^^^^^^^^^^  the bigram was already in the prompt
```

The bigram rule also fired at **two** occurrences - a phrase recurring once,
which is not degeneration.

## 3. Premature-stop measurement, before and after

Fixed 30-question diagnostic: 10 definitions, 10 calculations, 10 explanations.

| | before | after |
| --- | --- | --- |
| production config: premature stops | 20.0% | **0.0%** |
| production config: usable rate | 80.0% | **100%** |
| production config: mean generated tokens | 32.5 | 40.0 (the cap) |
| greedy: premature stops | **76.7%** (23/30 `repeated_bigram`) | 20.0% |
| greedy: usable rate | 16.7% | **63.3%** |
| greedy: mean / min generated tokens | 10.7 / **1** | 34.6 / 12 |

Fix: the window covers only generated tokens, and a bigram must occur three
times.

## 4. End-of-sequence handling

There was **no EOS check at all**: generation ended only on `max_new_tokens` or
the repetition detector, so a finished answer ran straight into a fresh
`"Question:"`. `eos_token_id` is now supported and the backend passes
`<|endoftext|>`.

This was **not** what truncated anything. Measured `eot_emitted_rate` is
**0.0%** across all 60 diagnostic generations before and after: the model never
emits the token. The fix is correctness, not a recovery of accuracy - and the
fact that a model trained on `<|endoftext|>`-separated documents never predicts
it is itself a sign of how weakly the objective was learned.

## 5. Sampling matrix

10 questions per configuration, 128-token budget.

| temperature | top_p | mean output tokens | premature stops | usable |
| --- | --- | --- | --- | --- |
| 0.0 (greedy) | 0.8 / 0.9 / 0.95 | 124.6 | 0.0% | 100% |
| 0.3 | 0.8 / 0.9 / 0.95 | 126.3 / 125.8 / 128.0 | 0.0% | 100% |
| 0.5 | 0.8 / 0.9 / 0.95 | 106.7 / 119.1 / 119.1 | 0.0% | 100% |
| 0.7 | 0.8 / 0.9 / 0.95 | 118.7 / 124.1 / 122.6 | 0.0% | 100% |

`max_new_tokens` 64 / 128 / 256: usable 100% at all three.

**Read "usable" carefully.** It is a mechanical check - enough words, not a
loop, not an echo of the question - deliberately not a grade. It says the
decoder works. It says nothing about whether the answers are true, and mostly
they are not:

> "What is EBITDA?" → *"EBITDA is the difference between EBITDA and EBITDA. It
> is calculated by dividing EBITDA by the total revenue…"*
>
> "What is working capital?" → *"The purpose of working capital is to provide a
> company with a high net worth and a high net worth."*

No sampling configuration rescues accuracy, which is the finding: the ceiling is
not a decoding choice. The production configuration (temperature 0.7, top_k 40,
top_p 0.9, repetition penalty 1.3) is kept, now that it no longer truncates.

## 6. What this does to the model-only number

Re-measured on all 795 dev items, with every deterministic layer off:

| | before fix | after fix |
| --- | --- | --- |
| model-only accuracy | 0.88% (7/795) | **1.64%** (13/795) |
| hallucination rate | 98.42% | 97.63% |
| abstention accuracy | 0% | 0% |
| generated items | 0.14% | 0.69% |
| authored prose | 8.22% | 10.96% |

The bug was worth about **0.8 accuracy points**. The remaining 98.4% is the
model. And 1.64% still flatters it: reading the 13 correct answers by hand, most
are rubric coincidences -

> *"The operating margin for 2015 was 810.01/20 = 678.50.00."* (counted correct
> because a number matched)
> *"A potential negative effects of an economy may have a significant impact…"*

so true capability is **at or below 1%**.

## Answers to the brief's questions 1, 2 and 7

**1. Is the inference path working correctly?** It is now. It was not: the
repetition detector truncated three quarters of greedy answers and a fifth of
sampled ones, and EOS was never checked. Checkpoint, tokenizer, vocabulary,
special tokens, context handling, dtype and device were all correct throughout.
The one remaining defect is performance, not correctness: there is no KV cache,
so generation is O(n²) and costs ~8 s per answer on CPU.

**2. Why was model-only accuracy 0.88%?** Roughly 0.8 points of it was the
truncation bug. The rest is capability: a 101M-parameter model trained on ~2B
tokens produces fluent, grammatical, confidently wrong financial prose.

**7. What is Aivora's actual model-only capability?** 1.64% measured, below 1%
once coincidental rubric passes are discounted, with a 97.6% hallucination rate
and no ability to abstain.
