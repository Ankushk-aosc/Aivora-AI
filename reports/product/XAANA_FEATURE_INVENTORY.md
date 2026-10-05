# Xaana feature inventory and Aivora mapping

Source: public pages at https://www.xaana.ai/ inspected 2026-10-05 — homepage,
`/xaanaone`, `/iocr`, plus the full public navigation (32 links).
Method: public marketing pages only. No demo access, no product screenshots, no
documentation portal, no public video walkthrough. "Request a demo" is gated.

## The finding that shapes everything below

**Xaana is not a financial-analysis product.** It is an enterprise IT services
and automation company whose platform work centres on TechnologyOne ERP:
integration connectors, invoice OCR and accounts-payable automation, fraud
alerting, computer vision, managed support, backup and migration services.

Nothing on the public site shows a financial statement analysis workspace, a
document Q&A interface, an evidence panel, a comparison view, a report
generator, saved analyses or chat history. Those appear in the brief as assumed
Xaana capabilities; **they are not verifiable, and the brief's own rule is not to
invent them.** They are therefore recorded below as *Aivora-native* — good
product ideas from the brief, owned by Aivora, not adapted from Xaana.

A second limit: because no product UI is publicly visible, **no UX pattern could
be verified**. The brief asked for UX patterns and workflows to be extracted;
what is observable is marketing-site structure only. I have not inferred
interface behaviour from marketing copy.

## A. Capabilities discovered (verified from public pages)

| # | Capability | What the site says it does | Relevant to Aivora? | Priority |
|---|---|---|---|---|
| 1 | iOCR document capture | Capture, recognition, **field extraction and validation** from invoices; error and duplicate detection before data enters the ERP | **Yes** — direct analogue of Aivora extraction | P0 |
| 2 | Insights AI — natural language over applications | "Use natural language to interact with all your applications"; Gen-AI reporting, anomaly detection, database query | **Yes** — analogue of the Financial Analyst | P0 |
| 3 | Advanced retrieval / AI workspace | Homepage: "advanced retrieval, and an AI workspace tailored for the modern enterprise" | **Yes** — analogue of document grounding/RAG | P0 |
| 4 | Fraud AI — anomaly alerting | Real-time enterprise fraud alerts, accelerated anomaly detection | Partly — anomaly flagging over financials | P2 |
| 5 | Spend analytics reporting | "Spend reports provide data on invoice matching and tolerance, early payments and discounts" | Partly — reporting over extracted data | P1 |
| 6 | Straight-through processing / workflow automation | Approval workflows, invoice matching rules | No — Aivora is analysis, not AP processing | NOT RELEVANT |
| 7 | 500+ ERP connectors / integrations | Pre-built TechnologyOne and 3rd-party connectors | No — no ERP in scope | NOT RELEVANT |
| 8 | Vision AI / geospatial | Object tracking, predictive maintenance | No | NOT RELEVANT |
| 9 | Managed support, backup, migration | 24/7 support, SLAs, data backup/recovery | No — services, not product | NOT RELEVANT |
| 10 | Security & compliance posture | SOC 1 / SOC 2 Type 2, ISO 27001/27002/9001, IRAP, GDPR | Partly — as a hygiene standard, not a feature | P1 |
| 11 | Multilingual models | Homepage claim | Not verified in product | P2 |
| 12 | Collaboration | "Connect easily with all of your suppliers" (supplier comms, in AP context) | No — supplier workflow, not analysis | NOT RELEVANT |

### Not verified — explicitly not implemented as "Xaana-inspired"

Document Q&A interface · evidence panels · source citation UI · financial
dashboards · period comparison · report generation UI · saved analyses · chat
history · alerts the user can define in natural language · knowledge management.
The brief lists these; **the public site does not demonstrate them.** Where
Aivora builds them, they are Aivora's own product decisions.

## B. Mapping — only for capabilities verified above

**1. iOCR document capture → Aivora Document Intelligence**

Upload → process → extract → index → ready → query. Frontend: Documents
library + viewer. Backend: `/api/document/upload`, `/api/rag/*`. Data: financial
documents in `data/documents/`. AI: extraction with source spans. Test:
document upload + query tests. **Status: already implemented before this brief.**

**2. Insights AI → Aivora Financial Analyst**

Ask → route by intent → specialised handler → evidence → answer. Frontend:
Financial Analyst view. Backend: `/api/copilot/chat`, `/api/route`. AI:
calculator for arithmetic, retrieval for document questions, refusal when
unsupported. Test: intent routing (80/80), response pipeline (26/26).
**Status: already implemented.**

**3. Advanced retrieval → Aivora grounding and source traceability**

Every stated figure traceable to the document it came from. **Status:
implemented in the pipeline; surfacing it in the UI as an evidence panel is the
Aivora-native part.**

**4. Spend analytics reporting → Aivora Reports** (P1)

Report over verified financial data with explicit sources. Aivora-native in
form; the "reporting over extracted data" idea is the verified analogue.

**5. Fraud/anomaly → Aivora Risk** (already present as a risk engine)

**6. Security posture → Aivora security hygiene** — upload validation, no
secrets client-side, sanitised document rendering. Not a user-facing feature.

## C. Capabilities rejected, with reasons

ERP connectors, AP straight-through processing, supplier collaboration, vision
AI, managed services, backup/recovery, migrations — all are either services
rather than product, or belong to an accounts-payable/ERP problem Aivora does
not address. Implementing any of them would add navigation that does nothing for
a financial analyst.

## D. What this means for the build

Of the brief's eight product areas, five already exist and work. The three that
do not — Comparisons, Insights, Reports — are **Aivora-native** and are built
because they serve the financial-analysis workflow, not because Xaana was
observed to have them.

One hard data limit found while planning, which constrains §21 of the brief:
`Aivora_Enterprise_Client_FY2025_Report.txt` contains **only FY2024 revenue**
($8,000,000). It contains no FY2024 COGS, gross profit, operating expenses,
operating profit or net profit. A FY2024-vs-FY2025 comparison across all those
lines cannot be built from real data, so Comparisons shows the lines that exist
and marks the rest as not present in the source, per the brief's own rule that
comparisons are built "ONLY where real data exists".
