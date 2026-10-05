# Aivora Financial Intelligence — final MVP report

Date: 2026-10-05 · Commits `c79f233`, `e47384d`, `9c7c3af` · pushed to `main`
Verification: every row below was run against the live application at
`http://127.0.0.1:8001` on the date above. Nothing is marked PASS on inspection.

---

## The finding that should be read first

**Xaana is not a financial-analysis product.** It is an enterprise IT services
and automation company built around TechnologyOne ERP: invoice OCR,
accounts-payable automation, fraud alerting, computer vision, managed support,
backups, migrations.

The brief treats Xaana as a benchmark for document Q&A, evidence panels,
dashboards, comparisons, report generation, saved analyses and chat history.
**None of those is visible on the public site.** The brief's own rule — "DO NOT
claim Xaana has a feature unless verified" — therefore governs this report:
those capabilities are recorded as **Aivora-native**, built because they serve a
financial analyst, not because Xaana was observed to have them.

A second limit: the demo is gated and no product UI is public, so **no Xaana UX
pattern could be verified**. The brief asked for UX patterns and workflows; what
is observable is a marketing site. I did not infer interface behaviour from
marketing copy.

Full inventory: [`XAANA_FEATURE_INVENTORY.md`](XAANA_FEATURE_INVENTORY.md).

---

## §52 — Final feature matrix

PASS means exercised on the running application today. "Pre-existing" marks work
that was already in the repository before this task.

| Feature | Xaana verified? | Aivora implemented? | Frontend | Backend | Status | Test |
|---|---|---|---|---|---|---|
| Document capture & extraction | **Yes** (iOCR) | Yes (pre-existing) | Documents | `/api/document/upload` | PASS | live + `test_aivora_system` |
| Document Q&A, grounded | No — Aivora-native | Yes (pre-existing) | Documents | `/api/document/query` | PASS | live: cites the Aivora filing |
| **Workspace document scope** | No — Aivora-native | **Yes (new)** | Documents | `services/workspace.py` | PASS | `test_product_views` (11 checks) |
| Safe refusal | No — Aivora-native | Yes (pre-existing) | Documents, Analyst | `/api/document/query` | PASS | live: FY2030 refused |
| Natural-language analyst | **Yes** (Insights AI) | Yes (pre-existing) | Financial Analyst | `/api/copilot/chat` | PASS | `test_intent_routing` 80/80 |
| Financial calculations | No — Aivora-native | Yes (pre-existing) | Analysis | `/api/financial/calculate` | PASS | `test_response_pipeline` 26/26 |
| Risk assessment | **Yes** (Fraud/anomaly, adapted) | Yes (pre-existing) | Analysis, Reports | `/api/risk/analyze` | PASS | live |
| Structured output | No — Aivora-native | Yes (pre-existing) | Analysis | `/api/structured-output` | PASS | live |
| Overview dashboard | No — Aivora-native | Yes (pre-existing) | Overview | `/api/company` | PASS | live |
| **Comparisons** | No — Aivora-native | **Yes (new)** | Comparisons | `/api/comparison` | PASS | `test_product_views` |
| **Insights with evidence** | No — Aivora-native | **Yes (new)** | Insights | `/api/insights` | PASS | `test_product_views` |
| **Reports + export** | **Yes** (spend analytics, adapted) | **Yes (new)** | Reports | `/api/report/generate` | PASS | live: 2,445-char Markdown |
| **Monitoring conditions** | **Yes** (Fraud AI alerting, adapted) | **Yes (new)** | Insights | `/api/alerts*` | PASS | `test_alerts_and_analyses` |
| **Saved analyses** | No — Aivora-native | **Yes (new)** | Saved analyses | `/api/analyses*` | PASS | survives reload, verified |
| Management focus | No — Aivora-native | Yes | Reports, Overview | `/api/report/generate` | PASS | `test_product_views` |
| Source citations | No — Aivora-native | Partial | all new views | all new endpoints | **PARTIAL** | see L2 |
| ERP connectors | Yes | **No — rejected** | — | — | NOT RELEVANT | — |
| AP straight-through processing | Yes | **No — rejected** | — | — | NOT RELEVANT | — |
| Vision AI / geospatial | Yes | **No — rejected** | — | — | NOT RELEVANT | — |
| Managed services, backup | Yes | **No — rejected** | — | — | NOT RELEVANT | — |
| Chat history / saved conversations | No — unverified | **No — not built** | — | — | NOT BUILT | see L4 |
| Multi-company comparison | No — unverified | **No — not built** | — | — | NOT BUILT | see L3 |

---

## §53 — Final report

### A. Xaana capabilities discovered (verified)

iOCR document capture with field extraction and validation · natural-language
interaction with applications (Insights AI: Gen-AI reporting, anomaly detection,
database query) · advanced retrieval and an AI workspace · Fraud AI anomaly
alerting · spend analytics reporting · 500+ ERP connectors · Vision AI ·
managed support, backup, migrations · security posture (SOC 1, SOC 2 Type 2,
ISO 27001/27002/9001, IRAP, GDPR) · multilingual models (claim only).

### B. Capabilities selected for Aivora

Document capture and extraction → Document Intelligence. Natural-language
querying → Financial Analyst. Advanced retrieval → grounding with source
traceability. Anomaly alerting → monitoring conditions on metrics. Spend
analytics reporting → Reports. Security posture → upload validation, no secrets
client-side, sanitised rendering.

### C. Capabilities rejected

ERP connectors, AP straight-through processing, supplier collaboration, Vision
AI, managed services, backup and recovery, migrations. All are either services
rather than product, or belong to an accounts-payable/ERP problem Aivora does
not address. Building any would add navigation that does nothing for an analyst.

### D. Aivora features implemented in this task

1. **Comparisons** — FY2024 vs FY2025, honest about what the source lacks.
2. **Insights** — six observations, each with figure, claim, arithmetic, source.
3. **Reports** — six sections, generate and export to Markdown.
4. **Monitoring conditions** — 13 metrics, evaluated against the filing.
5. **Saved analyses** — server-side, survive restart.
6. **Workspace document scope** — the bug fix described in L1.

### E. Files changed

`app/backend/server.py` (13 route entries, 9 handlers, workspace scoping in
document query, upload registration) · `app/frontend/index.html` (4 nav items, 4
views, loaders, Markdown export, styles, same-origin fix) · `.gitignore`
(runtime database).

### F. Files added

`app/backend/services/product_views.py` · `app/backend/services/workspace.py` ·
`app/backend/services/alerts.py` · `app/backend/services/saved_analyses.py` ·
`tests/test_product_views.py` · `tests/test_alerts_and_analyses.py` ·
`reports/product/XAANA_FEATURE_INVENTORY.md` · this report.

### G. Files removed

None. No existing file was deleted, and no document was removed from disk.

### H. APIs added (9 paths, 13 route entries — all mine)

`GET /api/comparison` · `GET /api/insights` · `GET|POST /api/report/generate` ·
`GET /api/alerts` · `GET /api/alerts/metrics` · `POST /api/alerts/create` ·
`POST /api/alerts/delete` · `GET /api/analyses` · `POST /api/analyses/save` ·
`GET|POST /api/analyses/open` · `POST /api/analyses/delete`.

### I. APIs modified

`POST /api/document/query` — results scoped to the workspace's documents, and
the workspace's documents indexed on demand. `POST /api/document/upload` /
`/api/rag/upload` — an uploaded document joins the workspace, so it becomes
answerable.

*(Commit `e47384d` also carries the owner's pre-existing, previously uncommitted
endpoints — `/api/financial/*`, `/api/document/*`, `/api/copilot/chat`,
`/api/risk/analyze`, `/api/structured-output`, `/v1/chat/completions`,
`/api/company`. Those are not my work and are attributed in that commit.)*

### J. Tests added

139 checks across two files: `test_product_views.py` (81) and
`test_alerts_and_analyses.py` (58). Both are weighted toward absences and
refusals, because that is where a financial interface is tempted to invent.

### K. Tests passed — full run, today

| Suite | Result |
|---|---|
| test_product_views | **81/81** |
| test_alerts_and_analyses | **58/58** |
| test_tool_pipeline | 55/55 |
| test_intent_routing | 80/80 |
| test_benchmark | 47/47 |
| test_response_pipeline | 26/26 |
| test_api_security | 25/25 |
| test_scorer_regressions | 23/23 |
| test_training_format | 23/23 |
| test_numbers_in | 19/19 |
| test_knowledge_retrieval | 11/11 |
| test_aivora_system | OK |
| test_financial_extraction_bug | OK (9 tests) |
| verify_frontend_clean | PASS |
| check_ai_terms | PASS |

Plus 14 live endpoint probes, all PASS. No suite regressed against the baseline
captured before any change was made.

Responsive (§46): no horizontal overflow at 1440, 1280, 1024, 768 or 375 on any
new view.

### L. Remaining limitations

**L1 — a defect found, and what it implies.** Asking *"What was revenue?"*
returned **Microsoft's** quarterly revenue, labelled `DOCUMENT GROUNDED` with a
genuine citation. The retrieval index held three third-party sample excerpts and
**not** the client's filing, while the Documents library displayed only the
client's filing. Fixed by scoping answers to the workspace. The implication
outlived the fix: a correct-looking citation is not evidence of a correct
source, and nothing in the test suite would have caught it — it was found by
running the §50 flow by hand. *(fixed, verified)*

**L2 — citations are document-level, not passage-level.** Every figure names its
source document and shows its arithmetic, but §18's clickable
page/section-level citation is **not** implemented: the source is a flat `.txt`
with no page structure, and inventing page numbers is forbidden. Honest, and
less than §18 asked for.

**L3 — the comparison is one line deep, and that is the data's fault.** The
FY2025 filing states **only FY2024 revenue**. No FY2024 COGS, gross profit,
operating expenses, operating profit or net profit exist anywhere in the source.
So **1 of 9 lines is genuinely comparable**; the other 8 show FY2025 with "Not in
source" and the reason. §21 asked for a full comparison table and it cannot be
built from real data. Multi-company and document-vs-document comparison are not
built for the same reason: one company, one filing.

**L4 — not built, deliberately.** Chat history and saved conversations (§32):
nothing persists conversations, and §32 says to show them only if implemented.
Global search (§27): the corpus is a single document, where a search box would
be decoration. Both are straightforward once there is a reason for them.

**L5 — monitoring is evaluated, not watched.** There is no scheduler, so
conditions are tested when the Insights view is opened, and the interface says
so in place. A genuine threshold crossing is detected between evaluations and
flagged; nothing is called a notification. Continuous monitoring needs a
scheduler and a change feed, neither of which exists.

**L6 — one workspace, one period.** The workspace → company → documents
structure is in place (`data/workspace.json`, `services/workspace.py`) and
uploads join it, but only Aivora Enterprise FY2025 has data. Multi-company is
designed for, not demonstrated.

**L7 — the model behind the Analyst is the weak component.** The application
loads the from-scratch SFT_003 checkpoint. On this project's own frozen
evaluation, that family of checkpoints scores extraction at 13.0% while a
pre-trained 1.5B model behind the tool pipeline scores 87.0% with zero invented
values ([`PHASE4_V2_REPORT.md`](../phase4/PHASE4_V2_REPORT.md)). The grounding
guard and calculator carry the product; the from-scratch model is its least
reliable part. Swapping the Analyst onto the Phase 4 pipeline is the single
largest available quality improvement and is not done here.

**L8 — no authentication on the new endpoints.** They follow the existing
server's posture, which has `/api/auth/*` but does not gate these routes. The
monitoring and saved-analysis stores are therefore per-installation, not
per-user. Fine for a single-workspace MVP; it must change before multi-user.

---

## §34 compliance — every visible control was exercised

Nav items: 8 areas, all load. Buttons: Generate report, Export, Add condition,
Remove, Save analysis, Open, Delete — each triggered through the UI and
confirmed to do what it says. No feature is displayed that does not work; no
placeholder, fabricated figure or dead control was added. Where a figure does
not exist, the interface says "Not in source" with the reason rather than
rendering a blank that reads like zero.
