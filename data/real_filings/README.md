# Real filings — drop one here

This directory is empty, so Task 6 of the audit was **skipped**. It was not
substituted with synthetic data: a synthetic filing cannot test whether the
system reads a real one.

## What to put here

One real annual report — a US 10-K or an ASX annual report — as PDF or text.
A single filing is enough.

## What happens then

Re-run the Task 6 harness. It will:

1. Upload the filing through the normal `/api/document/upload` path.
2. Ask 10 questions: 4 extraction, 3 calculation, 3 that should be refused
   (one of them forward-looking).
3. Save every request and response under `reports/audit/06_real_filing/`.
4. Report per question: grounded or not, the value, the cited span, whether
   that span **really appears in the source text** (checked programmatically,
   not asserted), and whether the value is correct — flagged for human review
   where it cannot be decided automatically.

## Why this matters more than the synthetic results

Every number measured so far comes from generated contexts of the form
`Label: value`. Real filings have tables, footnotes, multi-column layouts,
figures in thousands or millions with the scale declared in a header far from
the number, and values split across lines. None of that is represented in the
current evaluation, so none of the reported accuracy transfers to a real filing
without this test.
