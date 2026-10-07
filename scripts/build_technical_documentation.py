"""Build Aivora_Technical_Documentation.docx.

Structured to match the reference document supplied by the owner (PropFlow):
a title block, numbered H1/H2 sections, dense specification tables, bullet
lists, embedded architecture figures, and a closing glossary.

Every figure, count and file reference is taken from the repository or from this
project's own measured evaluation reports. Where something is unverified or
absent it is written as such rather than filled in.

    python scripts/build_architecture_diagrams.py     # figures first
    python scripts/build_technical_documentation.py
"""

import os

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

OUT = "Aivora_Technical_Documentation.docx"
DIAGRAMS = os.path.join("docs", "diagrams")

INK = RGBColor(0x11, 0x18, 0x27)
MUTED = RGBColor(0x6B, 0x72, 0x80)
ACCENT = RGBColor(0x1D, 0x4E, 0xD8)
WARN = RGBColor(0xB9, 0x1C, 0x1C)


def shade(cell, hex_colour):
    element = OxmlElement("w:shd")
    element.set(qn("w:val"), "clear")
    element.set(qn("w:fill"), hex_colour)
    cell._tc.get_or_add_tcPr().append(element)


def table(document, headers, rows, widths=None):
    t = document.add_table(rows=1, cols=len(headers))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, header in enumerate(headers):
        cell = t.rows[0].cells[i]
        cell.text = ""
        run = cell.paragraphs[0].add_run(header)
        run.bold = True
        run.font.size = Pt(9)
        run.font.color.rgb = INK
        shade(cell, "E8EDF5")
    for row in rows:
        cells = t.add_row().cells
        for i, value in enumerate(row):
            cells[i].text = ""
            run = cells[i].paragraphs[0].add_run(str(value))
            run.font.size = Pt(9)
            if i == 0 and len(headers) > 2:
                run.bold = True
    if widths:
        for row in t.rows:
            for i, width in enumerate(widths):
                row.cells[i].width = Inches(width)
    document.add_paragraph()
    return t


def bullets(document, items):
    for item in items:
        paragraph = document.add_paragraph(item, style="List Bullet")
        paragraph.paragraph_format.space_after = Pt(3)
        for run in paragraph.runs:
            run.font.size = Pt(10)


def numbered(document, items):
    for item in items:
        paragraph = document.add_paragraph(item, style="List Number")
        paragraph.paragraph_format.space_after = Pt(3)
        for run in paragraph.runs:
            run.font.size = Pt(10)


def body(document, text, italic=False, colour=None, size=10):
    paragraph = document.add_paragraph()
    run = paragraph.add_run(text)
    run.font.size = Pt(size)
    run.italic = italic
    if colour is not None:
        run.font.color.rgb = colour
    paragraph.paragraph_format.space_after = Pt(8)
    return paragraph


def code(document, text):
    paragraph = document.add_paragraph()
    run = paragraph.add_run(text)
    run.font.name = "Consolas"
    run.font.size = Pt(8.5)
    paragraph.paragraph_format.left_indent = Inches(0.25)
    paragraph.paragraph_format.space_after = Pt(8)
    return paragraph


def callout(document, text):
    """A stated limitation or guarantee - the things a reader must not miss."""
    t = document.add_table(rows=1, cols=1)
    t.style = "Table Grid"
    cell = t.rows[0].cells[0]
    cell.text = ""
    run = cell.paragraphs[0].add_run(text)
    run.font.size = Pt(9.5)
    run.bold = True
    run.font.color.rgb = WARN
    shade(cell, "FDF2F2")
    document.add_paragraph()


def figure(document, filename, caption, width=6.6):
    path = os.path.join(DIAGRAMS, filename)
    if not os.path.exists(path):
        body(document, f"[figure missing: {filename}]", italic=True, colour=WARN)
        return
    document.add_picture(path, width=Inches(width))
    document.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph = document.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = paragraph.add_run(caption)
    run.font.size = Pt(8.5)
    run.italic = True
    run.font.color.rgb = MUTED
    paragraph.paragraph_format.space_after = Pt(12)


def h1(document, text):
    document.add_heading(text, level=1)


def h2(document, text):
    document.add_heading(text, level=2)


def build():
    document = Document()
    style = document.styles["Normal"]
    style.font.name = "Calibri"
    style.font.size = Pt(10)

    for section in document.sections:
        section.top_margin = Inches(0.8)
        section.bottom_margin = Inches(0.8)
        section.left_margin = Inches(0.85)
        section.right_margin = Inches(0.85)

    # ------------------------------------------------------------ title
    title = document.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = title.add_run("AIVORA")
    run.bold = True
    run.font.size = Pt(34)
    run.font.color.rgb = ACCENT

    for text, size, colour in (
            ("Financial Intelligence", 15, INK),
            ("Document Extraction, Grounded Analysis & Evidence-Backed Reporting",
             11, MUTED),
            ("Technical Documentation", 13, INK)):
        paragraph = document.add_paragraph()
        paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = paragraph.add_run(text)
        run.font.size = Pt(size)
        run.font.color.rgb = colour
        if size >= 13:
            run.bold = True

    paragraph = document.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = paragraph.add_run(
        "Version 1.0  ·  5 October 2026  ·  Aivora Enterprise FY2025 workspace")
    run.font.size = Pt(9)
    run.font.color.rgb = MUTED
    document.add_paragraph()

    callout(document,
            "Reading note. Every figure, count and measurement in this document "
            "is taken from the repository or from this project's own evaluation "
            "reports, and each is labelled measured, inferred or unverified. "
            "Known limitations are stated in section 12 rather than omitted.")

    # ------------------------------------------------- 1 executive overview
    h1(document, "1. Executive Overview")
    body(document,
         "Aivora is a financial intelligence application that reads company "
         "filings and answers questions about them. Its defining property is "
         "not that it answers, but that it refuses to answer without evidence: "
         "no figure is presented to a user unless it can be traced to a span of "
         "text in the source document, and no arithmetic is performed by a "
         "language model. Values are extracted with their source spans, "
         "validated in code, and every calculation is executed in Python.")
    body(document,
         "The system comprises a browser interface of eight product areas, an "
         "HTTP API of 99 method-path routes over 82 endpoints, a service "
         "layer of routing, "
         "extraction, grounding and reporting modules, a document retrieval "
         "store, and a tool pipeline that enforces the evidence rules. It runs "
         "on CPU with no external service dependency.")

    h2(document, "1.1 Purpose & Use Cases")
    bullets(document, [
        "Extraction of financial values from company filings, each bound to the "
        "span of source text it was read from.",
        "Deterministic calculation of financial ratios, performed in Python "
        "from extracted operands rather than generated by a model.",
        "Grounded document question answering, scoped so that a document can "
        "only answer for the company it belongs to.",
        "Explicit refusal when a filing does not contain the information asked "
        "for, including forward-looking questions no filing can answer.",
        "Period-over-period comparison that reports which lines have no "
        "prior-year figure instead of leaving them blank.",
        "Evidence-backed insights and exportable financial reports in which "
        "every stated number carries its formula, working and source document.",
        "Monitoring conditions on financial metrics, evaluated against the "
        "filing, with conditions the engine cannot compute refused at creation.",
    ])

    h2(document, "1.2 Technology Stack")
    table(document, ["Category", "Technology / Library"], [
        ("Runtime", "Python 3.14, CPU-only deployment"),
        ("HTTP Server", "ThreadingHTTPServer (standard library) - no web framework"),
        ("Frontend", "Single-page HTML, CSS and vanilla JavaScript - no build step"),
        ("Deep Learning", "PyTorch 2.13 (CPU build)"),
        ("Local Model", "DeepSeekV3-architecture model trained in this repository, "
                        "101.7M parameters"),
        ("Optional Model Backend", "Hugging Face Transformers 5.x - Qwen2.5-Instruct "
                                   "and SmolLM2 verified"),
        ("Tokeniser", "tiktoken, GPT-2 BPE, 50,257 tokens"),
        ("Retrieval", "TF-IDF chunk index with SQLite persistence"),
        ("Structured Storage", "SQLite (standard library sqlite3) for monitoring "
                               "rules and saved analyses"),
        ("Document Parsing", "pypdf, python-docx, plain text"),
        ("Diagrams & Docs", "matplotlib, python-docx - figures generated from source"),
        ("Testing", "Standalone Python test scripts, no test framework dependency"),
    ], widths=[2.0, 4.6])

    h2(document, "1.3 Model Specification")
    body(document, "The model trained in this repository, used as the local "
                   "generation backend. (measured, from models/config.py)")
    table(document, ["Parameter", "Value"], [
        ("Architecture", "DeepSeek-V3 inspired - MoE, Multi-head Latent Attention, "
                         "Multi-Token Prediction"),
        ("Total parameters", "101.7M"),
        ("Layers", "8"),
        ("Hidden size", "512"),
        ("Attention heads", "8"),
        ("Context length", "1,024 tokens"),
        ("Experts", "8, top-2 routing"),
        ("KV LoRA rank", "128"),
        ("Query LoRA rank", "192"),
        ("Vocabulary", "50,257 (GPT-2 BPE)"),
        ("Training tokens", "120.9M unique, 16.3 epochs"),
        ("Token / parameter ratio", "1.19 (compute-optimal is approximately 20)"),
    ], widths=[2.0, 4.6])

    callout(document,
            "The local model is under-trained by design constraint, not by "
            "defect: at 1.19 tokens per parameter it saw about 6% of the data a "
            "model this size needs. Section 11 reports what that costs, and why "
            "the architecture places no trust in the model's arithmetic.")

    # ------------------------------------------------ 2 system architecture
    h1(document, "2. System Architecture")
    body(document,
         "Aivora is a layered pipeline. A request from the browser enters a "
         "single-file HTTP layer, is classified by intent, dispatched to the "
         "service that can answer it correctly, validated against the source "
         "document, and returned with its evidence. The layering exists so that "
         "the component least able to be trusted - the language model - is "
         "given the smallest possible job.")
    figure(document, "01_system_architecture.png",
           "Figure 1. System architecture. The evidence band at the foot of the "
           "diagram is what every successful response carries.")

    h2(document, "2.1 High-Level Component Map")
    table(document, ["Layer", "Module", "Responsibility"], [
        ("HTTP API", "app/backend/server.py",
         "Route table, request parsing, JSON responses, static file serving"),
        ("Security", "app/backend/security.py",
         "CORS allow-list, body size limit, path confinement, extension allow-list"),
        ("Workspace", "services/workspace.py",
         "Company to document scope; what a question may be answered from"),
        ("Intent Routing", "services/financial_router.py",
         "Classifies a question as calculation, extraction, definition or refusal"),
        ("Analyst", "services/chat_service.py",
         "Conversational analysis, context carry-over, structured responses"),
        ("Analyst Bridge", "services/analyst_pipeline.py",
         "Routes the Analyst through the tool pipeline when a capable model is "
         "hosted; refuses rather than degrading"),
        ("Value Binding", "services/financial_values.py",
         "Binds a label to its value with the source span, nearest-label "
         "preference"),
        ("Product Views", "services/product_views.py",
         "Comparisons, insights and reports as structured data with sources"),
        ("Monitoring", "services/alerts.py",
         "13 computable metrics, condition evaluation, state-change detection"),
        ("Saved Analyses", "services/saved_analyses.py",
         "Named snapshots of a view, persisted server-side in SQLite"),
        ("Grounding", "services/grounding.py, services/abstention.py",
         "Determines whether an answer is supported; produces refusals"),
        ("Answer Quality", "services/quality.py, services/question_focus.py",
         "Checks that an answer addresses the question actually asked"),
        ("Calculation", "tools/financial_calculator.py",
         "Deterministic financial arithmetic"),
        ("Tool Pipeline", "pipeline/tool_pipeline.py, pipeline/schema.py",
         "Span validation, operand resolution, Python-only computation"),
        ("Retrieval", "rag/retriever.py, rag/persistent_store.py",
         "TF-IDF chunk index, SQLite-backed chunk persistence"),
        ("Generation", "services/generation.py",
         "Pluggable model backends - local DeepSeekV3 or Hugging Face"),
    ], widths=[1.3, 2.0, 3.3])

    h2(document, "2.2 Request Flow")
    numbered(document, [
        "The browser issues a JSON request to an endpoint on the same origin.",
        "security.py applies the CORS allow-list, body size limit and, for any "
        "path argument, confinement beneath the document root.",
        "The route table dispatches to a handler; handlers contain no model, "
        "scoring or financial logic of their own.",
        "financial_router.py classifies the question's intent.",
        "The responsible service answers: calculator for arithmetic, retrieval "
        "for document questions, knowledge layer for definitions, refusal rule "
        "for questions no filing can answer.",
        "Workspace scoping discards any retrieval hit from a document outside "
        "the company being viewed.",
        "The grounding layer confirms the answer is supported by its source; "
        "if not, an abstention is produced.",
        "The response is returned with its value, source span, formula, working "
        "and source document.",
    ])

    figure(document, "02_question_routing.png",
           "Figure 2. Question routing. Each lane is handled by the component "
           "that can answer it correctly; all converge on one validation gate.")

    # ---------------------------------------------------- 3 module detail
    h1(document, "3. Module-by-Module Explanation")

    h2(document, "3.1 Tool Pipeline (pipeline/tool_pipeline.py)")
    body(document,
         "The pipeline divides labour so that correctness does not depend on "
         "the model being good. The model reads the context and copies a value "
         "with the span it came from. Python validates that span, selects the "
         "formula and performs every piece of arithmetic. Rules abstain "
         "whenever validation fails or an operand is missing.")
    table(document, ["Element", "Count", "Purpose"], [
        ("Formulas", "21", "Question wording to operation and operand list"),
        ("Operations", "8", "difference, sum, ratio, percentage_of, margin, "
                            "difference_margin, percentage_change, enterprise_value"),
        ("Operand fields", "24", "The canonical operands formulas may request"),
        ("Captions", "147", "Statement wordings mapped to those operands"),
    ], widths=[1.5, 0.9, 4.2])
    body(document,
         "Operand resolution is the reason the pipeline reaches real filings. A "
         "formula asks for \"total debt\"; a statement says \"Borrowings\". "
         "Python resolves the operand to the caption the document actually "
         "uses before the model is asked for it, so the model is only ever "
         "asked to copy a label that is verbatim in front of it. Matching is "
         "exact or qualified-prefix, never substring, and a caption is assigned "
         "to whichever operand matches it most specifically - without that, "
         "\"total debt\" would claim \"Debtors and cash\". (measured: this "
         "raised calculation accuracy from 83.3% to 100% on the frozen "
         "evaluation set)")

    h2(document, "3.2 Validation Schema (pipeline/schema.py)")
    body(document, "An extraction is accepted only if all three conditions hold:")
    numbered(document, [
        "The model reports the value as found.",
        "The cited source span appears in the context that was supplied.",
        "The value occurs inside the span that was cited.",
    ])
    body(document,
         "Comparison is whitespace- and Unicode-normalised, because models "
         "reflow whitespace and substitute quotation marks when copying; the "
         "original span is what gets reported. Any condition failing produces "
         "an abstention. A fabricated span cannot satisfy condition two, so a "
         "value the document does not contain cannot reach a user.")
    figure(document, "03_tool_pipeline.png",
           "Figure 3. The four stages of the tool pipeline.")

    h2(document, "3.3 Workspace Scope (services/workspace.py)")
    body(document,
         "A document is answerable only if it belongs to the workspace being "
         "viewed. This is a product rule rather than a ranking adjustment: no "
         "relevance score should permit one company's filing to answer a "
         "question about another's. Uploaded documents join the workspace and "
         "become answerable; files present on disk but outside the workspace "
         "never answer for it.")
    callout(document,
            "This module exists because of a defect found in testing. "
            "\"What was revenue?\" returned Microsoft's quarterly revenue, "
            "labelled DOCUMENT GROUNDED with a genuine citation, because the "
            "retrieval index held third-party sample excerpts and not the "
            "client's filing. A correct-looking citation is not evidence of a "
            "correct source.")

    h2(document, "3.4 Value Binding (services/financial_values.py)")
    body(document,
         "Parses financial values from text into records carrying field, value, "
         "unit, currency, scale, source span and confidence. Label-to-value "
         "association uses nearest-label binding with a backward preference, so "
         "a value is attributed to the label that precedes it rather than to a "
         "distant heading. Thousands separators are parsed as whole tokens: a "
         "malformed separator is rejected rather than read as its leading "
         "digits.")

    h2(document, "3.5 Monitoring (services/alerts.py)")
    body(document,
         "Conditions on financial metrics, evaluated against the current "
         "filing. The API offers exactly the 13 metrics the engine can compute, "
         "so a condition that cannot be tested cannot be created. A metric that "
         "is computable in general but absent from the filing in hand evaluates "
         "to \"not evaluable\" with a reason, never to \"condition not met\" - "
         "reading an absence as satisfied would be a silent failure.")
    callout(document,
            "There is no scheduler. Conditions are evaluated when the view is "
            "opened, and the interface states this in place. A genuine "
            "threshold crossing between evaluations is detected and reported; "
            "nothing is described as a notification.")

    h2(document, "3.6 Retrieval (rag/)")
    body(document,
         "Documents are loaded, chunked, embedded with TF-IDF and indexed. "
         "Chunk text and metadata are persisted to SQLite; embedding vectors "
         "are deliberately not persisted, and the vector space is rebuilt in a "
         "single reindex on load so it stays internally consistent rather than "
         "mixing vectors from an earlier corpus and vocabulary.")

    h2(document, "3.7 Product Views (services/product_views.py)")
    body(document,
         "Comparisons, insights and reports, returned as data so the interface "
         "renders and the export serialises without either re-deriving a "
         "figure. Each value carries its source and, where derived, the "
         "arithmetic that produced it. A line with no prior-year figure is "
         "returned as unavailable with a reason, never as a blank that reads "
         "like a zero.")

    # -------------------------------------------------------- 4 lifecycle
    h1(document, "4. Document Lifecycle")
    body(document,
         "A filing passes through seven stages before it can answer a "
         "question. Status is reported from the actual stage reached; progress "
         "is never simulated.")
    figure(document, "04_document_lifecycle.png",
           "Figure 4. Document lifecycle, from upload to answerable record.",
           width=6.8)
    table(document, ["Stage", "Module", "Outcome"], [
        ("Upload", "POST /api/document/upload", "File received"),
        ("Validate", "security.py", "Extension, size and path confinement checked"),
        ("Store", "data/documents/", "Written beneath the document root"),
        ("Chunk", "rag/document_loader.py", "Split into overlapping chunks"),
        ("Index", "rag/retriever.py", "TF-IDF vectors built, chunks persisted"),
        ("Register", "services/workspace.py", "Added to the workspace manifest"),
        ("Answerable", "-", "Available to scoped Q&A and extraction"),
    ], widths=[1.1, 2.3, 3.2])

    # ---------------------------------------------------------- 5 api
    h1(document, "5. API Reference")
    body(document, "Selected endpoints. All accept and return JSON; all are "
                   "served from the same origin as the interface.")
    table(document, ["Method", "Endpoint", "Purpose"], [
        ("GET", "/api/health", "Service and checkpoint status"),
        ("GET", "/api/company", "Verified company profile and headline metrics"),
        ("POST", "/api/financial/calculate", "Deterministic metric calculation"),
        ("POST", "/api/financial/full-analysis",
         "Metrics, risks, recommendations, executive summary"),
        ("POST", "/api/document/upload", "Upload and index a filing"),
        ("POST", "/api/document/query", "Workspace-scoped grounded question answering"),
        ("POST", "/api/copilot/chat", "Financial Analyst conversation"),
        ("POST", "/api/risk/analyze", "Risk assessment with evidence"),
        ("GET", "/api/comparison", "Period comparison with per-line availability"),
        ("GET", "/api/insights", "Evidence-backed observations"),
        ("GET|POST", "/api/report/generate", "Financial report as structured sections"),
        ("GET", "/api/alerts", "Monitoring conditions evaluated against the filing"),
        ("GET", "/api/alerts/metrics", "The metrics a condition may name"),
        ("POST", "/api/alerts/create", "Create a condition; refuses unknown metrics"),
        ("GET", "/api/analyses", "Saved analyses, newest first"),
        ("POST", "/api/analyses/save", "Persist a named snapshot of a view"),
        ("GET", "/api/analyst/engine", "Which engine serves the Analyst, and why"),
    ], widths=[0.8, 2.3, 3.5])

    body(document, "Example - a grounded answer and its evidence:")
    code(document,
         'POST /api/document/query\n'
         '{ "query": "What was revenue?" }\n\n'
         '{ "grounded": true,\n'
         '  "status": "DOCUMENT GROUNDED",\n'
         '  "answer": "... total consolidated revenue of $10,000,000 (AUD) ...",\n'
         '  "citation": "Aivora_Enterprise_Client_FY2025_Report.txt" }')
    body(document, "Example - a refusal, which is a correct answer:")
    code(document,
         'POST /api/document/query\n'
         '{ "query": "What will revenue be in FY2030?" }\n\n'
         '{ "grounded": false,\n'
         '  "status": "INFORMATION NOT AVAILABLE",\n'
         '  "answer": "The provided document does not contain sufficient\n'
         '              information to determine FY2030 revenue.",\n'
         '  "explanation": "Corporate filings do not contain FY2030 forecasts." }')

    # ------------------------------------------------------ 6 data & schema
    h1(document, "6. Data & Persistence")
    table(document, ["Store", "Technology", "Contents"], [
        ("data/documents/", "Filesystem", "Source filings"),
        ("data/workspace.json", "JSON", "Workspace, company, document scope"),
        ("data/demo_company.json", "JSON", "Verified company profile and metrics"),
        ("Chunk store", "SQLite", "Document chunk text and metadata"),
        ("data/monitoring.db", "SQLite", "Monitoring rules, saved analyses"),
        ("checkpoints/", "PyTorch", "Model checkpoints"),
        ("data/eval_heldout/", "JSONL", "Frozen evaluation set and manifest"),
    ], widths=[1.6, 1.3, 3.7])
    body(document,
         "Runtime databases are excluded from version control: they hold a "
         "user's records, not source. The frozen evaluation set is verified on "
         "every scored run against a content hash computed over the parsed "
         "items in canonical order, so the check is identical on every platform "
         "and still changes if any item changes.")

    # ----------------------------------------------------- 7 grounding
    h1(document, "7. Grounding & Hallucination Prevention")
    body(document, "Four independent mechanisms, each of which can refuse on "
                   "its own:")
    table(document, ["Mechanism", "Enforced by", "Effect"], [
        ("Source-span validation", "pipeline/schema.py",
         "A value with no valid span never reaches the user"),
        ("Python-only arithmetic", "pipeline/tool_pipeline.py",
         "A wrong total from a model cannot become a wrong answer"),
        ("Workspace scope", "services/workspace.py",
         "Another company's filing cannot answer for this one"),
        ("Refusal rules", "services/abstention.py",
         "Forward-looking and unsupported questions are declined explicitly"),
    ], widths=[1.6, 1.8, 3.2])
    callout(document,
            "An abstention is a correct answer. It is never replaced by a guess "
            "from another path, because doing so would discard the guarantee "
            "that justifies the architecture.")

    # ---------------------------------------------------- 8 evaluation
    h1(document, "8. Evaluation & Measured Results")
    body(document,
         "Aivora is measured against a frozen held-out set of 313 items across "
         "five capability gates, built with 8-gram overlap removal against the "
         "training corpus and scored once per configuration by a single shared "
         "scorer. Prompts, decoding and scoring are identical across systems. "
         "(measured; see reports/phase4/PHASE4_V2_REPORT.md)")
    table(document, ["Gate", "Tool pipeline + Qwen2.5-1.5B", "95% CI"], [
        ("Copy", "68/69 - 98.6%", "[92.2, 99.7]"),
        ("Extraction", "67/69 - 97.1%", "[89.9, 99.2]"),
        ("Wording robustness", "72/75 - 96.0%", "[88.9, 98.6]"),
        ("Calculation", "48/48 - 100%", "[92.6, 100]"),
        ("Abstention", "35/36 - 97.2%", "[85.8, 99.5]"),
        ("Overall", "290/297 - 97.6%", "[95.2, 98.8]"),
        ("Invented values", "0 / 261 answerable items", "-"),
        ("Source-span validation", "273 / 273 stated values", "-"),
    ], widths=[1.9, 2.7, 2.0])

    h2(document, "8.1 What the Calculation Figure Means")
    body(document,
         "Calculation reads 100%, and that number is conditional in a way the "
         "figure alone does not convey. Operand resolution matches a formula's "
         "operand to the caption a statement actually uses, from a table of 147 "
         "captions. The same capability, measured three ways:")
    table(document, ["Measurement", "Score"], [
        ("Frozen set", "48/48 - 100%"),
        ("Fresh contexts, captions the vocabulary has", "44/44 - 100%"),
        ("Fresh contexts, captions the vocabulary lacks", "5/44 - 11.4%"),
    ], widths=[4.2, 2.4])
    body(document,
         "When the table knows the wording the pipeline is exact; when it does "
         "not, the model bridges the gap 5 times in 44. Coverage, not "
         "arithmetic, is the limit. The failure is the safe one - of 39 misses "
         "on unknown captions, 39 were refusals and none was a wrong value, "
         "with zero invented values and zero span violations. (measured)")
    callout(document,
            "A real filing will use wording outside the table, so it will meet "
            "refusals rather than errors. Calculation should be read as "
            "coverage multiplied by accuracy-within-coverage, and coverage "
            "against real filings has not yet been measured.")

    h2(document, "8.2 The Pipeline's Contribution, Isolated")
    body(document,
         "The same model, same scorer, same prompts and decoding, answering "
         "directly instead of through the pipeline, on the same 297 items:")
    table(document, ["Gate", "Model alone", "With the pipeline"], [
        ("Copy", "97.1%", "98.6%"),
        ("Extraction", "98.5%", "97.1%"),
        ("Wording robustness", "88.0%", "96.0%"),
        ("Calculation", "52.1%", "100%"),
        ("Abstention", "63.9%", "97.2%"),
        ("Overall", "249/297 - 83.8%", "290/297 - 97.6%"),
        ("Invented values", "0.77%", "0.00%"),
    ], widths=[1.9, 2.2, 2.5])
    body(document,
         "Unaided, the model gets roughly half of all arithmetic wrong and "
         "answers 13 of the 36 questions it should refuse. Those two gates are "
         "what the pipeline buys. Extraction is within a single item of the "
         "unaided model. (measured)")

    h2(document, "8.3 Comparison Against the Local Model")
    body(document,
         "On identical items, with an exact McNemar test for paired "
         "observations: the tool pipeline was better on 44 items and worse on "
         "none (p < 0.0001). The from-scratch model scores 13.0% on extraction "
         "and invents a value on 36.8% of answerable items. (measured)")
    body(document,
         "A parameter-matched comparison settles the cause. SmolLM2-135M - a "
         "pre-trained model of 135M parameters against Aivora's 101.7M - beats "
         "the from-scratch model on identical items (p = 0.0309). With size "
         "approximately held constant the pre-trained model still wins, so the "
         "deficit is training data, not capacity. (measured)")

    h2(document, "8.4 Test Suite")
    table(document, ["Suite", "Checks", "Covers"], [
        ("test_product_views.py", "81", "Comparisons, insights, reports, workspace scope"),
        ("test_alerts_and_analyses.py", "58", "Monitoring conditions, saved analyses"),
        ("test_tool_pipeline.py", "55", "Span rule, Python arithmetic, operand vocabulary"),
        ("test_intent_routing.py", "80", "Question classification"),
        ("test_benchmark.py", "47", "Benchmark construction and scoring"),
        ("test_analyst_pipeline.py", "28", "Analyst bridge and readiness gate"),
        ("test_response_pipeline.py", "26", "End-to-end response construction"),
        ("test_api_security.py", "25", "CORS, body limits, path confinement"),
        ("test_scorer_regressions.py", "23", "Scorer correctness"),
        ("test_numbers_in.py", "19", "Numeric parsing and token boundaries"),
        ("test_knowledge_retrieval.py", "11", "Glossary and definitions"),
    ], widths=[2.3, 0.8, 3.5])
    body(document, "All suites pass as of 5 October 2026. (measured)")

    # -------------------------------------------------- 9 installation
    h1(document, "9. Installation & Setup")
    h2(document, "9.1 Prerequisites")
    bullets(document, [
        "Python 3.11 or later (3.14 verified).",
        "Approximately 2 GB of disk space for the repository and checkpoints.",
        "4 GB RAM minimum for the local model; 8 GB free to host a 1.5B "
        "parameter Hugging Face backend.",
        "No GPU required. No external API key required.",
    ])

    h2(document, "9.2 Installation")
    code(document,
         "git clone https://github.com/Ankushk-aosc/Aivora-AI.git\n"
         "cd Aivora-AI\n"
         "python -m venv deepseek_env\n"
         "deepseek_env\\Scripts\\activate        # Windows\n"
         "source deepseek_env/bin/activate      # macOS / Linux\n"
         "pip install -r requirements.txt\n"
         "python -m app.backend.server --port 8001")
    body(document, "The interface is then served at http://127.0.0.1:8001/.")

    h2(document, "9.3 Environment Variables")
    table(document, ["Variable", "Default", "Purpose"], [
        ("AIVORA_ANALYST_PIPELINE", "1",
         "On by default. Set to 0 to use the rule-based engine instead; when "
         "on and no capable model can be hosted, the Analyst refuses rather "
         "than answering through an unguarded path"),
        ("AIVORA_ANALYST_MODEL", "Qwen/Qwen2.5-1.5B-Instruct",
         "Model backing the Analyst pipeline"),
        ("AIVORA_CORS_ORIGINS", "localhost allow-list",
         "Permitted browser origins"),
        ("AIVORA_DOCUMENT_ROOT", "data/documents",
         "Root beneath which document paths are confined"),
        ("AIVORA_MAX_BODY_BYTES", "built-in limit", "Maximum request body size"),
    ], widths=[2.1, 1.9, 2.6])

    h2(document, "9.4 Running the Tests")
    code(document,
         "deepseek_env/Scripts/python.exe tests/test_product_views.py\n"
         "deepseek_env/Scripts/python.exe tests/test_tool_pipeline.py\n"
         "deepseek_env/Scripts/python.exe tests/test_alerts_and_analyses.py\n"
         "deepseek_env/Scripts/python.exe tests/test_api_security.py")

    # ------------------------------------------------------ 10 security
    h1(document, "10. Security")
    table(document, ["Control", "Implementation"], [
        ("Origin control", "CORS allow-list; same-origin API calls from the interface"),
        ("Request size", "Body byte limit enforced before parsing"),
        ("Path confinement", "Document paths resolved and confined beneath the "
                             "document root"),
        ("Extension allow-list", "Only permitted document extensions accepted"),
        ("Content handling", "Uploaded document content is never executed; "
                             "rendered output is escaped"),
        ("Secrets", "No API keys, credentials or model secrets in frontend code"),
    ], widths=[1.8, 4.8])
    callout(document,
            "Limitation. The product endpoints are not individually "
            "authenticated. Monitoring rules and saved analyses are therefore "
            "per-installation rather than per-user. This is acceptable for a "
            "single-workspace deployment and must change before multi-user use.")

    # ----------------------------------------------------- 11 analyst
    h1(document, "11. The Analyst Engine and Its Gate")
    body(document,
         "The Analyst can be served either by the existing rule-based engine or "
         "by the tool pipeline. The pipeline is the better engine, and it is "
         "gated for a measured reason.")
    body(document,
         "Placed behind the local 101.7M model, the pipeline abstained on three "
         "of three test questions. The model emitted \"Revenue = $5,200,000\" "
         "for a context stating $10,000,000: it invented a figure, and the span "
         "rule stopped it. The guarantee held, but coverage was zero. Switching "
         "wholesale would have replaced an engine that answers with one that "
         "refuses everything. (measured, 5 October 2026)")
    body(document,
         "The bridge therefore refuses to serve unless the configured model can "
         "actually be hosted, and reports the shortfall at "
         "/api/analyst/engine. Enabling it on unsuitable hardware is not "
         "possible by accident.")
    table(document, ["Engine", "Extraction", "Invented values", "Hosting"], [
        ("Local 101.7M model", "13.0%", "36.8%", "Runs on 4 GB RAM"),
        ("Qwen2.5-1.5B alone, no pipeline", "98.5%", "0.77%",
         "Needs about 7 GB free"),
        ("Tool pipeline + Qwen2.5-1.5B", "97.1%", "0.0%", "Needs about 7 GB free"),
    ], widths=[2.3, 1.2, 1.5, 1.6])

    # -------------------------------------------------- 12 limitations
    h1(document, "12. Known Limitations")
    bullets(document, [
        "Citations are document-level, not passage-level. Source documents are "
        "plain text with no page structure, and page numbers are not invented.",
        "Period comparison is limited by the source. The FY2025 filing states "
        "only FY2024 revenue, so 1 of 9 comparison lines has a prior-year "
        "figure. UNDER REVIEW: a change not yet accepted supplies the other "
        "eight by applying FY2025 margins to FY2024 revenue. Those figures are "
        "modelled, not reported, and must not be presented beside audited ones "
        "without being labelled as such.",
        "Calculation accuracy is conditional on caption coverage: 100% on "
        "statement wordings the operand vocabulary holds, 11.4% on wordings it "
        "does not. Coverage against real filings is not measured.",
        "Two misses remain where a valid source span cites the wrong line - "
        "goodwill answered from an adjacent intangibles line, cash generated "
        "from operations from an adjacent revenue line. Span validation cannot "
        "catch this by construction: the span is real, the row is wrong.",
        "Multi-company and document-versus-document comparison are designed for "
        "but not demonstrated: the workspace contains one company and one "
        "period.",
        "Chat history and global search are not implemented. A search box over "
        "a single document would be decoration.",
        "Monitoring is evaluated on view, not watched continuously. There is no "
        "scheduler and no change feed.",
        "The product endpoints are unauthenticated; stores are "
        "per-installation.",
        "The local model is the least reliable component. It is used for "
        "generation only, never for arithmetic, and the pipeline is the "
        "recommended engine wherever it can be hosted.",
        "Evaluation set defects found and disclosed: two duplicated "
        "question-and-context pairs, and two abstention controls mislabelled "
        "answerable. The set is not rebuilt after freezing.",
    ])

    # ----------------------------------------------------- 13 glossary
    h1(document, "13. Glossary")
    table(document, ["Term", "Definition"], [
        ("Abstention", "An explicit refusal to answer, returned when the source "
                       "does not support an answer. A correct outcome, not a "
                       "failure."),
        ("Source span", "The exact passage of document text a value was copied "
                        "from, validated to appear in the context supplied."),
        ("Grounding", "Binding every stated figure to the document text that "
                      "supports it."),
        ("Workspace scope", "The rule that a document may only answer questions "
                            "about the company it belongs to."),
        ("Operand resolution", "Mapping a formula's operand to the caption the "
                               "statement actually uses, in code, before the "
                               "model is asked."),
        ("Frozen evaluation set", "313 held-out items, never trained on and "
                                  "never used for selection, scored once per "
                                  "configuration."),
        ("Selection split", "61 items used to choose a configuration; never "
                            "reported as a result."),
        ("Capability gate", "One of copy, extraction, wording, calculation and "
                            "abstention - measured separately so an average "
                            "cannot hide a failure."),
        ("McNemar test", "An exact paired significance test used when two "
                         "systems are measured on identical items."),
        ("Over-refusal", "Refusing a question that the source does in fact "
                         "answer. Tracked separately, because abstention "
                         "accuracy alone would reward a system that refuses "
                         "everything."),
        ("Invented value", "A number presented to the user that does not appear "
                           "in the source. The metric the architecture exists "
                           "to drive to zero."),
        ("MoE", "Mixture of Experts - 8 experts with top-2 routing in this "
                "model."),
        ("MLA", "Multi-head Latent Attention - compresses the key/value cache "
                "through a low-rank projection."),
        ("MTP", "Multi-Token Prediction - an auxiliary objective predicting "
                "more than one future token."),
    ], widths=[1.7, 4.9])

    document.save(OUT)
    size = os.path.getsize(OUT) / 1024
    print(f"wrote {OUT} ({size:.0f} KB)")


if __name__ == "__main__":
    build()
