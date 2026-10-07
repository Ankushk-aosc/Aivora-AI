# Task 7 — documented counts vs counted from code

| Claim | Documented | Counted from code | Verdict |
|---|---|---|---|
| Endpoints | "roughly sixty" | **99 route entries, 82 distinct paths** | **WRONG — understated by ~40%** |
| Alert metrics | 13 | 13 | correct |
| Formulas | 21 | 21 | correct |
| Operations | 8 | 8 | correct |
| Operand fields | 24 | 24 | correct |
| Captions | 147 | 147 | correct |

Counted by parsing the route table in `app/backend/server.py` for
`("METHOD", "path"):` entries, and by importing `FORMULAS`, `OPERATIONS`,
`SYNONYMS` and `alerts.METRICS` directly.

The endpoint figure is the only mismatch. 99 entries map to 82 distinct paths
because several accept both GET and POST. The documentation should say "around
80 endpoints, 99 method-path routes" — or drop the number, which carries no
meaning for a reader.

# Task 7 — naming

`grep -ril nivora` → **0 files.** The product is Aivora throughout; "Nivora"
appears only in the audit brief. No rename was performed: it would touch 145
files including the evaluation set's document filename
(`Aivora_Enterprise_Client_FY2025_Report.txt`), which is referenced inside
frozen eval items and in the workspace manifest. Renaming it would change the
frozen set's content hash, which ground rule 2 forbids.

If the product is to be called Nivora, the rename must be done as its own piece
of work, with the evaluation set rebuilt and re-frozen deliberately rather than
as a side effect.
