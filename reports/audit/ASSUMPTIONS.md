# Assumptions logged during the audit (ground rule 5)

1. **"Nivora" is this repository's "Aivora".** The brief names Nivora,
   `NIVORA_ANALYST_PIPELINE` and paths like `services/`; this repo is Aivora,
   with `AIVORA_ANALYST_PIPELINE` and `app/backend/services/`. Treated as the
   same product under a different name. No rename performed — recorded for
   Task 7 instead.

2. **Reproducing 275/297 means re-scoring preserved outputs.** The figure came
   from Qwen2.5-1.5B on a Kaggle T4; this machine has ~1 GB free against the
   ~7 GB the model needs. Re-generation is NOT MEASURED locally and is stated
   as such. The model ran once, its answers are fixed, and the scorer is
   applied to those fixed answers.

3. **The failing baseline tests are pre-existing.** `test_product_views`,
   `test_alerts_and_analyses` (53/58) and `verify_frontend_clean` already failed
   before any audit change, from on-disk edits made outside this session and
   from the absence of a running server. Recorded in `tests_before.txt`.

4. **The scorer is not modified.** Three extraction items return the correct
   value and are scored wrong for showing their working beside it. The
   conservative option is to leave the pre-registered scorer alone and report
   the counterfactual, not to re-score history.

5. **The fabricated prior-year figures are not reverted.** The change looks
   deliberate. The conservative option is to report it loudly and leave the
   decision with the owner.

6. **Fresh-split captions were written from accounting terminology, not from
   the pipeline's vocabulary table.** Coverage of that table is then reported as
   a measurement. This cannot be perfect — the table had been read earlier in
   the project — so the mitigation is disclosure, not a claim of blindness.

7. **The fresh split is synthetic.** It is generated with a seeded RNG, not
   drawn from real filings, because no real filing is present in the repository
   (see Task 6). Operand magnitudes are related so the statements are
   financially plausible; the first version produced gross margins of -35.8%
   and net margins of 1045%, which would not have tested statement reading.
