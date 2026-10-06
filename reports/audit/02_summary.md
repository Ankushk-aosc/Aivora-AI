# Task 2 — why 313 items score as 297

The held-out set holds 313 items. The five gates sum to 297 because
**16 items are deliberately excluded from the overall figure, and are reported
separately instead.**

All 16 are *answerable abstention controls*: items in the abstention gate whose
`expected` is `False`, meaning "this question CAN be answered from the context,
so the system must not refuse it". They are not must-refuse items, so counting
them inside the abstention gate would mix two opposite tasks — a system that
never refuses would score a free 16/16 on them, inflating an abstention figure
that is supposed to measure refusal.

`scripts/eval_heldout.evaluate()` therefore builds the overall figure from
`scored_records`, excluding exactly `gate == "abstention" and expected is False`,
and reports those 16 as **over-refusal** (k/n) in their own line. The abstention
gate itself then reports only the 36 must-refuse items, which is why it reads
35/36 rather than 51/52.

313 − 16 = 297. The reconciliation is exact, the exclusion is intentional, and
nothing is silently dropped: every one of the 16 appears in the over-refusal
statistic.

Per-item detail, with id, gate, origin and reason: `02_unscored_items.csv`.

Two of the 16 are mislabelled, which is a defect in the evaluation set rather
than in the scorer. `abstention_043` and `abstention_048` both ask for the quick
ratio from a context containing no inventory, which the formula requires; the
ratio genuinely cannot be computed, so they are not answerable and should not be
controls. Correctly labelled, over-refusal would be 0 of 14 rather than 2 of 16.
The frozen set is read-only under ground rule 2, so this is reported, not fixed.
