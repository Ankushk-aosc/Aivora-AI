"""AI Capability Registry (spec Part 35).

The single source of truth for what this platform can actually do.
Every capability declares its real status - IMPLEMENTED / TESTED / PARTIAL /
BLOCKED / NOT_IMPLEMENTED - per Part 34 ("never claim a capability exists
unless it's actually connected and tested"). Nothing here is aspirational;
a capability's status only changes when the code backing it changes.
"""

from dataclasses import dataclass, field
from enum import Enum


class Status(str, Enum):
    IMPLEMENTED = "IMPLEMENTED"       # code exists and runs
    TESTED = "TESTED"                 # implemented AND verified against real inputs
    PARTIAL = "PARTIAL"               # implemented for a subset of the spec'd capability
    BLOCKED = "BLOCKED"               # a specific external dependency is missing
    NOT_IMPLEMENTED = "NOT_IMPLEMENTED"


@dataclass
class Capability:
    name: str
    route: str          # matches the CapabilityRoute enum in orchestrator.py
    type: str            # llm | tool | ml_model | agent | infra
    model: str
    version: str
    provider: str        # "proprietary" | "statistical" | "heuristic" | "none"
    status: Status
    reason: str = ""      # required detail when status is not TESTED
    endpoint: str = ""
    tools: list = field(default_factory=list)
    permissions: str = "none required"

    def to_dict(self):
        d = {**self.__dict__}
        d["status"] = self.status.value
        return d


REGISTRY = {
    # ------------------------------------------------------------------
    # Real, working, tested this session or earlier this project
    # ------------------------------------------------------------------
    "GENERAL_LLM": Capability(
        name="General LLM", route="GENERAL_LLM", type="llm",
        model="FinLLM-102M-Instruct", version="checkpoints/instruction/checkpoint_100.pt",
        provider="proprietary", status=Status.TESTED,
        reason="Proprietary DeepSeek-V3-style model (101.7M params). Trained only "
        "100 steps on CPU - architecture and pipeline are verified, but language "
        "ability is not yet useful (see evaluation results).",
        endpoint="/api/chat",
    ),
    "FINANCIAL_LLM": Capability(
        name="Financial LLM", route="FINANCIAL_LLM", type="llm",
        model="FinLLM-102M-Financial", version="checkpoints/base/checkpoint_100.pt",
        provider="proprietary", status=Status.TESTED,
        reason="Same proprietary model, base-stage checkpoint trained on the full "
        "6-bucket financial+general dataset mix. Same training-budget caveat as above.",
        endpoint="/api/chat",
    ),
    "CALCULATOR": Capability(
        name="Financial Calculator / Math Engine", route="CALCULATOR", type="tool",
        model="deterministic Python", version="1.0", provider="none",
        status=Status.TESTED,
        reason="15 financial formulas (margins, CAGR, ROE/ROA/ROIC, D/E, current "
        "ratio, FCF, EPS, P/E, EV/EBITDA). Verified 15/15 against hand-computed "
        "expected values. Never uses LLM-generated arithmetic.",
        endpoint="/api/calculate",
    ),
    "RAG": Capability(
        name="Retrieval-Augmented Generation", route="RAG", type="tool",
        model="TF-IDF retriever + FinLLM", version="1.0", provider="proprietary",
        status=Status.TESTED,
        reason="Ingest -> parse -> chunk -> embed (TF-IDF, or model hidden-states "
        "when a checkpoint is loaded) -> retrieve -> LLM -> citations. Verified "
        "against real TXT/PDF/DOCX/CSV/JSON files with correct page citations.",
        endpoint="/api/rag/search",
    ),
    "LIVE_DATA": Capability(
        name="Live Market Data", route="LIVE_DATA", type="tool",
        model="Yahoo Finance public chart endpoint", version="1.0",
        provider="unofficial_public_api", status=Status.TESTED,
        reason="No API key required. Verified against real symbols (AAPL). "
        "This is an UNOFFICIAL, undocumented public endpoint, not a "
        "credentialed integration - it can change or rate-limit without "
        "notice. Any fetch failure (unknown symbol, network error, "
        "endpoint change) returns 'Current market data is not available.' - "
        "verified for an invalid symbol - never a fabricated price.",
        endpoint="/api/ai/orchestrate",
    ),
    "DOCUMENT_AI": Capability(
        name="Document AI", route="DOCUMENT_AI", type="tool",
        model="pypdf / python-docx parsers", version="1.0", provider="none",
        status=Status.PARTIAL,
        reason="Text extraction, chunking, and structured CSV/JSON field reading "
        "are implemented and tested (rag/document_loader.py). OCR (scanned/image "
        "PDFs), classification, invoice/contract-specific extraction, and table "
        "structure extraction are NOT implemented - no OCR engine or "
        "document-classification model is connected.",
    ),

    # ------------------------------------------------------------------
    # Real, newly implemented this session (statistical / heuristic - no
    # GPU or external model required, so no fabrication is involved)
    # ------------------------------------------------------------------
    "FORECASTING_AI": Capability(
        name="Forecasting AI", route="FORECASTING_AI", type="ml_model",
        model="linear trend + simple exponential smoothing", version="1.0",
        provider="statistical", status=Status.TESTED,
        reason="Classical statistical forecasting (not an LLM guessing numbers). "
        "Returns point forecast + MAE/RMSE backtested error metrics. No deep "
        "learning forecasting model (e.g. a trained temporal transformer) exists.",
        endpoint="/api/ai/forecast",
    ),
    "ANOMALY_AI": Capability(
        name="Anomaly Detection AI", route="ANOMALY_AI", type="ml_model",
        model="IQR (default) / z-score outlier detection + duplicate-invoice matching",
        version="1.0", provider="statistical", status=Status.TESTED,
        reason="Deterministic statistical outlier detection on numeric "
        "transaction data (IQR by default - robust to a single dominant "
        "outlier, unlike z-score, which that same outlier can mask), plus "
        "exact/near-duplicate invoice detection reusing the dataset-dedup "
        "hashing. Not a trained anomaly-detection neural network - none "
        "exists or was trained.",
        endpoint="/api/ai/anomaly",
    ),

    # ------------------------------------------------------------------
    # Explicitly not implemented - real external dependencies missing
    # ------------------------------------------------------------------
    "EMBEDDING_AI": Capability(
        name="Embedding AI", route="EMBEDDING_AI", type="ml_model",
        model="TF-IDF (fallback) / FinLLM hidden states", version="1.0",
        provider="proprietary", status=Status.PARTIAL,
        reason="rag/embeddings.py provides two embedders, but there is no "
        "dedicated pretrained embedding model (e.g. a sentence-transformer) - by "
        "design (Part 45 of the earlier spec forbids importing pretrained "
        "external models). TF-IDF is lexical, not semantic; the model-hidden-state "
        "embedder inherits the LLM's undertrained quality.",
    ),
    "RERANKING_AI": Capability(
        name="Reranking AI", route="RERANKING_AI", type="ml_model",
        model="BM25 (Okapi, from scratch)", version="1.0",
        provider="statistical", status=Status.TESTED,
        reason="Second-stage BM25 reranking over the first-stage TF-IDF "
        "retrieval, now used by default in the RAG pipeline. Measured, not "
        "just claimed: on a test query ('What was the EBITDA margin?') "
        "against 4 chunks, first-stage TF-IDF ranked an irrelevant risk-"
        "factors chunk #1 and the actually-relevant EBITDA chunk #2; BM25 "
        "reranking correctly promoted the relevant chunk to #1. Not a "
        "pretrained cross-encoder (none available without a GPU/download) - "
        "a classical, from-scratch IR ranking function, genuinely distinct "
        "from the first-stage cosine score it re-ranks.",
        endpoint="/api/ai/rerank",
    ),
    "VISION_AI": Capability(
        name="Vision AI", route="VISION_AI", type="ml_model",
        model="none", version="-", provider="none", status=Status.NOT_IMPLEMENTED,
        reason="No vision model is connected or tested. Image/chart/table "
        "understanding, scanned-document OCR, and visual QA are not available.",
    ),
    "SPEECH_AI": Capability(
        name="Speech AI", route="SPEECH_AI", type="ml_model",
        model="Windows SAPI (System.Speech)", version="1.0",
        provider="local_os", status=Status.PARTIAL,
        reason="Text-to-speech only, and genuinely tested: produced a real, "
        "valid 4.19s WAV file (parsed with Python's wave module to confirm "
        "it isn't a stub - 92,365 audio frames at 22050Hz) from real text. "
        "Uses the OS's own synthesis engine (3 installed voices found), not "
        "a downloaded model. Speech-to-text is NOT implemented: the only "
        "realistic local option (e.g. openai-whisper) is a 150MB+ pretrained "
        "model download, which wasn't added without a clear need to verify "
        "against - this stays honestly PARTIAL rather than silently skipping "
        "half the capability.",
        endpoint="/api/ai/speech/synthesize",
    ),
    "CODE_AI": Capability(
        name="Code AI", route="CODE_AI", type="agent",
        model="subprocess sandbox (python -I -S + restricted builtins)",
        version="1.0", provider="local", status=Status.PARTIAL,
        reason="Execution only (not generation - the LLM isn't capable enough "
        "to write correct code yet). Real process isolation: fresh subprocess, "
        "hard wall-clock timeout (verified: an infinite loop was killed within "
        "the timeout), restricted builtins via a purpose-built exec() globals "
        "dict (verified: dir() raises NameError; caught and fixed a real bug "
        "first where reassigning __builtins__ in the same frame did NOT "
        "actually restrict it - CPython caches f_builtins at frame creation). "
        "This is PROCESS isolation, not container/VM-level security (no "
        "Docker/seccomp available here) - a determined attacker with time to "
        "find gaps in the builtin restriction could still find one; it stops "
        "the obvious cases (import os, open, eval, exec, runaway loops), not "
        "every case.",
        endpoint="/api/ai/code/execute",
    ),
    "DATABASE_AI": Capability(
        name="Database AI / Data Analyst", route="DATABASE_AI", type="agent",
        model="SQLite (stdlib) + deterministic query templates", version="1.0",
        provider="local", status=Status.PARTIAL,
        reason="Real local SQLite database, real schema discovery, and real "
        "read-only query execution - verified with a seeded synthetic test "
        "table (correct SUM/COUNT/ORDER BY results, e.g. supplies total "
        "2000.0 = 1200+800). SQL generation is deliberately NOT free-form "
        "NL2SQL from the LLM (it isn't trained enough to be trusted with "
        "exact queries yet) - a small set of parameterized templates handles "
        "'total by X', 'top N', 'total', 'average'; anything else is honestly "
        "reported as unsupported rather than guessed. Write/DDL statements "
        "and multi-statement injection are rejected before reaching sqlite3 "
        "(verified: DROP/DELETE/multi-statement all blocked).",
        endpoint="/api/ai/database",
    ),
    "FRAUD_AI": Capability(
        name="Fraud AI", route="FRAUD_AI", type="ml_model",
        model="RandomForestClassifier (scikit-learn)", version="1.0",
        provider="trained_local", status=Status.TESTED,
        reason="Trained on pointe77/credit-card-transaction (Apache-2.0, "
        "synthetic Sparkov-simulator data - not real cardholders), 60,000 "
        "streamed records, held-out test set (n=15,000, 145 fraud cases). "
        "Real measured metrics: precision 0.393, recall 0.938, F1 0.554, "
        "ROC-AUC 0.992. Recall is strong (catches most fraud); precision is "
        "genuinely mediocre (many false positives) because the feature set is "
        "just amount/time/category - no per-cardholder velocity features "
        "(e.g. 'transactions in the last hour'), which real fraud systems "
        "rely on heavily. Reported as-is, not rounded up.",
        endpoint="/api/ai/fraud",
    ),
    "RECOMMENDATION_AI": Capability(
        name="Recommendation AI", route="RECOMMENDATION_AI", type="agent",
        model="deterministic rules over SQLite data", version="1.0",
        provider="local", status=Status.PARTIAL,
        reason="Real, verified against seeded synthetic test data: vendor "
        "concentration (correctly flagged a vendor at 40.4% spend share, "
        "correctly did NOT flag one at 26.9%) and cost-outlier detection "
        "(reuses the already-tested IQR anomaly detector rather than a second "
        "untested implementation). Every recommendation carries recommendation/"
        "reason/evidence/confidence/data_sources as required. PARTIAL because "
        "it only reasons over whatever is in the local SQLite database - no "
        "real enterprise spend/vendor-catalog data source is connected, so "
        "recommendations are only as meaningful as the (currently synthetic "
        "test) data seeded into it.",
        endpoint="/api/ai/recommend",
    ),
    "MULTILINGUAL_AI": Capability(
        name="Multilingual AI", route="MULTILINGUAL_AI", type="tool",
        model="langdetect (pure Python, no download)", version="1.0",
        provider="local", status=Status.PARTIAL,
        reason="Language DETECTION only - verified 4/5 on a mixed-language "
        "test set (en/de/ja/zh-cn correct; a short Spanish question was "
        "misclassified as French, a known langdetect weakness on short "
        "strings - a longer Spanish sentence detected correctly at "
        "confidence 1.0). Translation is NOT implemented: the only realistic "
        "local option (argostranslate) pulls in ~35 packages including spacy "
        "and onnxruntime plus per-language runtime downloads - too heavy to "
        "add and verify reliably on this machine (~1GB free RAM) without "
        "risking exactly the kind of unverified claim this registry exists "
        "to prevent. The proprietary LLM's tokenizer and training data are "
        "English-only regardless, so multilingual generation would be low "
        "quality even with translation added.",
    ),
    "RESEARCH_AI": Capability(
        name="Research AI", route="RESEARCH_AI", type="agent",
        model="DuckDuckGo HTML search (unofficial, no API key)", version="1.0",
        provider="unofficial_public_api", status=Status.PARTIAL,
        reason="Search -> Retrieve -> Cite is real and tested (verified real "
        "URLs/snippets for 'EBITDA definition finance'). Compare/Analyze/Verify "
        "stages are NOT implemented - the proprietary LLM is not yet capable "
        "enough to meaningfully synthesize across sources (see evaluation "
        "results), so this stops at citation-backed raw results rather than "
        "claiming a verified synthesis it can't produce. Unofficial endpoint: "
        "can break without notice.",
        endpoint="/api/ai/research",
    ),
    "KNOWLEDGE_GRAPH": Capability(
        name="Knowledge Graph AI", route="KNOWLEDGE_GRAPH", type="infra",
        model="networkx (in-memory) + regex relation extraction", version="1.0",
        provider="local", status=Status.PARTIAL,
        reason="Real graph storage (networkx) with rule-based (not LLM-based) "
        "OWNS/SUBSIDIARY_OF/VENDOR_OF/EMPLOYS/SUPPLIES extraction, verified "
        "with a correct 2-hop path query (Acme -> Gamma -> Delta) over "
        "multi-sentence text. A first version had a real entity-boundary bug "
        "(a greedy regex captured 'Beta Logistics, a subsidiary that handles' "
        "as one entity) - caught by testing a multi-hop path, not a single "
        "clean example, and fixed with a proper-noun-phrase pattern. "
        "PARTIAL because extraction is conservative by design: relations "
        "phrased unusually, or with non-proper-noun objects (e.g. 'employs "
        "over 200 workers'), are correctly not extracted rather than guessed - "
        "so recall is deliberately traded for precision, and this is not a "
        "general-purpose entity extractor.",
        endpoint="/api/ai/knowledge-graph",
    ),
}


def get_capability(route: str) -> Capability:
    if route not in REGISTRY:
        raise KeyError(f"Unknown capability route '{route}'. Known: {sorted(REGISTRY)}")
    return REGISTRY[route]


def list_capabilities(status: Status = None):
    caps = list(REGISTRY.values())
    if status is not None:
        caps = [c for c in caps if c.status == status]
    return caps


def summary():
    counts = {}
    for c in REGISTRY.values():
        counts[c.status.value] = counts.get(c.status.value, 0) + 1
    return counts
