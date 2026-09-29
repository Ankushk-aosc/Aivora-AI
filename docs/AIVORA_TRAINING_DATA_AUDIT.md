# Aivora training data audit

Step 9. Read from `checkpoints/final/checkpoint_247850.json` (the manifest the
training run embedded in the checkpoint), `data/dataset_manifest.json`,
`data_sources/` and the local shards. **Nothing was changed.**

The headline is not a percentage breakdown. It is this:

| | |
| --- | --- |
| tokens the run processed | **1,964,851,200** |
| **unique tokens in the corpus** | **120,857,129** |
| **epochs over that corpus** | **16.3** |
| unique tokens per parameter | **1.19** |
| compute-optimal tokens per parameter (Chinchilla) | ~20 |
| share of compute-optimal data actually seen | **~6%** |

The project has been describing this model as "trained on ~2B tokens". It was
trained on **121M tokens, sixteen times over**. Every previous conclusion that
attributed Aivora's weakness to its parameter count rests on that
misreading - including a conclusion in `docs/AIVORA_ARCHITECTURE.md`, which this
audit corrects.

## 1. Corpus composition

Unique training tokens, from the checkpoint's embedded manifest. The first seven
rows are a small local preparation; the larger rows are the Kaggle preparation
the real run used.

| dataset | licence | train tokens | share |
| --- | --- | --- | --- |
| fineweb_edu | odc-by | 28,535,064 | 23.6% |
| financial_qa_sujet (Sujet-Finance-Instruct-177k) | apache-2.0 | 23,499,207 | 19.4% |
| financial_reports_edgar (EDGAR filings) | apache-2.0 | 18,934,426 | 15.7% |
| general_wikipedia | cc-by-sa | 14,113,139 | 11.7% |
| financial_text_investopedia | cc-by-nc-4.0 | 9,663,359 | 8.0% |
| tinystories | cdla-sharing-1.0 | 9,495,170 | 7.9% |
| financial_reasoning_fino1 | (per registry) | 2,859,245 | 2.4% |
| general_instruction_dolly | cc-by-sa-3.0 | 2,447,489 | 2.0% |
| financial_instruction_alpaca | mit | 2,434,114 | 2.0% |
| general_qa_squad | cc-by-sa-4.0 | 2,433,624 | 2.0% |
| general_dialogue_oasst1 | apache-2.0 | 2,411,627 | 2.0% |
| general_reasoning_gsm8k | mit | 922,024 | 0.8% |
| the seven small local-prep rows | mixed | 3,108,641 | 2.6% |

Grouped:

| group | tokens | share |
| --- | --- | --- |
| **financial** | ~57.4M | **47.5%** |
| general web / encyclopaedic / stories | ~52.1M | 43.1% |
| general instruction, dialogue, QA, maths | ~8.2M | 6.8% |
| small local prep | 3.1M | 2.6% |

Configured mixture weights (`dataset_config.dataset_mix`): financial_text 20%,
fineweb_edu 20%, financial_qa 15%, financial_reasoning 10%, financial_reports
10%, financial_instruction 5%, general_dialogue 5%, general_instruction 5%,
general_qa 5%, general_reasoning 5%.

### By the capability the brief asks about

| capability | present? | evidence |
| --- | --- | --- |
| financial definitions | thin | Investopedia text (8%) is explanatory prose, not definition pairs |
| financial explanations | thin | the same 8%, plus parts of Sujet-Finance |
| causal reasoning | **very thin** | fino1 2.4% + gsm8k 0.8%; neither is financial causal explanation at scale |
| ratio interpretation | **absent as a category** | no dataset targets "what does this ratio imply" |
| financial calculations | thin | FinQA-style content in fino1 (2.4%) |
| statement analysis | raw only | EDGAR filings (15.7%) are documents, with no questions attached |
| valuation reasoning | **absent as a category** | nothing targets it |
| risk analysis | **absent as a category** | nothing targets it |
| question-answer behaviour | 6.8% of tokens | dolly, squad, oasst1, alpaca - as plain text, see the format section |
| instruction following | 4% of tokens | alpaca + dolly, as plain text |
| abstention / uncertainty | **absent** | no dataset teaches declining to answer, which is exactly what the model never does |

## 2. The format defect, which matters more than the mixture

`data_sources/prepare.py` and `shard_writer.py` both tokenise with
`enc.encode_ordinary(record["text"])` and write the ids straight into the shard.
`encode_ordinary` **excludes special tokens**, and no separator is inserted
between documents. Consequences, all of them measurable in the model's
behaviour:

* **`<|endoftext|>` never appears in the training data.** The model therefore
  cannot emit it, and the measured `eot_emitted_rate` is **0.0%** across 60
  diagnostic generations. It has no representation of "the answer is finished".
  This is why it runs on into a fresh `"Question:"`.
* **Documents are concatenated with no boundary.** A 1024-token window regularly
  spans the end of one document and the start of an unrelated one, and the model
  is trained to predict that junction as if it were continuous prose. Over 16.3
  epochs it sees the same junctions repeatedly.
* **Instruction and QA data was used as plain text.** The pretraining objective
  is next-token prediction over the whole sequence, with no prompt masking and
  no template: an alpaca row contributes its instruction and its answer as
  undifferentiated text. Nothing in the objective distinguishes "this part is
  the question" from "this part is the answer you should produce".

So the honest answer to "what was Aivora trained to do?" is: **continue
financial-flavoured text**. It was not trained to answer a question, to stop, or
to decline.

## 3. Duplication

| measure | value | how |
| --- | --- | --- |
| exact duplicate records removed at prepare time | 37 | manifest counters (`records_removed_duplicate`), summed |
| duplicate rate within records | <0.1% | same |
| **effective duplication from repetition** | **16.3x** | tokens processed / unique tokens |
| near-duplicate rate across documents | **not measured** | requires the Kaggle-side shards; the local shards hold only the 3.1M-token local prep |
| synthetic share | **not measured directly**; likely material | Sujet-Finance-Instruct-177k (19.4%) and finance-alpaca (2.0%) are model-generated instruction sets |
| real financial documents | 15.7% | EDGAR filings |

The number that matters is not the 37 removed records. It is that every token
was shown to the model sixteen times.

## 4. What this audit can and cannot say

**Measured here:** unique token count, epochs, per-dataset composition and
licences, mixture weights, the absence of a document separator, the absence of
prompt masking, exact-duplicate counts, and which capability categories have no
dataset behind them at all.

**Not measured, and why:** near-duplicate rate, per-document length
distributions and a text-level synthetic/real classification all need the
121M-token shards, which live in the Kaggle environment - the local
`data/shards/` directory holds 6 MB, the small local preparation. The local
instruction file `data/instruction/financial_instructions.jsonl` holds 800 rows
(~207k tokens); the 38,477-row build described in the architecture document was
produced on Kaggle.

## 5. Conclusions that follow directly

1. **The corpus is ~6% of compute-optimal size for 101.7M parameters.** At 1.19
   unique tokens per parameter, no conclusion about the architecture's capacity
   is available from this run. A model this size trained on 2B *unique* tokens
   has never been tried here.
2. **Repetition, not volume, produced the 1.965B figure.** 16.3 epochs on a small
   corpus is a recipe for memorisation, and it is consistent with a validation
   loss that stopped moving (4.37) while accuracy stayed at ~1%.
3. **The data contains almost nothing that teaches the target behaviours.**
   Ratio interpretation, valuation reasoning, risk analysis and abstention have
   no dataset behind them, and they are exactly the categories where the system
   measures 0% without its deterministic layers.
4. **The training format could not teach question answering.** No separator, no
   masking, no template.

These are data and objective findings. They do not prove that 101.7M parameters
are sufficient - they prove that the question has not yet been tested.
