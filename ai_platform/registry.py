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
        model="none", version="-", provider="none", status=Status.BLOCKED,
        reason="No market-data provider (e.g. a stock price API) is configured. "
        "Deliberately returns 'Current market data is not available.' rather than "
        "fabricating a price - this refusal is itself the tested behavior.",
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
        model="none", version="-", provider="none", status=Status.NOT_IMPLEMENTED,
        reason="No dedicated reranker (cross-encoder or otherwise) is implemented. "
        "RAG currently ranks by a single retrieval pass only.",
    ),
    "VISION_AI": Capability(
        name="Vision AI", route="VISION_AI", type="ml_model",
        model="none", version="-", provider="none", status=Status.NOT_IMPLEMENTED,
        reason="No vision model is connected or tested. Image/chart/table "
        "understanding, scanned-document OCR, and visual QA are not available.",
    ),
    "SPEECH_AI": Capability(
        name="Speech AI", route="SPEECH_AI", type="ml_model",
        model="none", version="-", provider="none", status=Status.NOT_IMPLEMENTED,
        reason="No speech-to-text or text-to-speech model is connected or tested.",
    ),
    "CODE_AI": Capability(
        name="Code AI", route="CODE_AI", type="agent",
        model="none", version="-", provider="none", status=Status.NOT_IMPLEMENTED,
        reason="No sandboxed execution environment is configured. Per spec Part 11, "
        "generated code must never run on the production host directly, and no "
        "isolated sandbox exists here, so this capability is deliberately not wired up.",
    ),
    "DATABASE_AI": Capability(
        name="Database AI / Data Analyst", route="DATABASE_AI", type="agent",
        model="none", version="-", provider="none", status=Status.NOT_IMPLEMENTED,
        reason="No database connection is configured. Schema discovery, "
        "SQL generation/validation, and query execution have nothing to run against.",
    ),
    "FRAUD_AI": Capability(
        name="Fraud AI", route="FRAUD_AI", type="ml_model",
        model="none", version="-", provider="none", status=Status.NOT_IMPLEMENTED,
        reason="Distinct from ANOMALY_AI: a real fraud model needs labeled "
        "historical fraud/legitimate transaction data to train a classifier. No "
        "such labeled dataset exists in this project, so no fraud classifier was "
        "trained. Use ANOMALY_AI (statistical outliers) as an unsupervised proxy "
        "with the understanding it is not a trained fraud detector.",
    ),
    "RECOMMENDATION_AI": Capability(
        name="Recommendation AI", route="RECOMMENDATION_AI", type="agent",
        model="none", version="-", provider="none", status=Status.NOT_IMPLEMENTED,
        reason="No enterprise data source (spend history, vendor catalog, etc.) "
        "is connected to generate grounded recommendations from.",
    ),
    "MULTILINGUAL_AI": Capability(
        name="Multilingual AI", route="MULTILINGUAL_AI", type="tool",
        model="none", version="-", provider="none", status=Status.NOT_IMPLEMENTED,
        reason="No language-detection or translation model is connected. The "
        "proprietary LLM's tokenizer (GPT-2 BPE) and training data are "
        "English-only, so multilingual generation would be low quality even if "
        "input language detection were added.",
    ),
    "RESEARCH_AI": Capability(
        name="Research AI", route="RESEARCH_AI", type="agent",
        model="none", version="-", provider="none", status=Status.NOT_IMPLEMENTED,
        reason="No live web-search API is configured inside the platform "
        "(distinct from the developer's own tooling used to build this project). "
        "Without a search backend there is nothing for a research agent to "
        "retrieve, compare, or cite.",
    ),
    "KNOWLEDGE_GRAPH": Capability(
        name="Knowledge Graph AI", route="KNOWLEDGE_GRAPH", type="infra",
        model="none", version="-", provider="none", status=Status.NOT_IMPLEMENTED,
        reason="No graph store or entity-relationship extraction pipeline exists. "
        "Would require both an extraction step (LLM or rules to identify "
        "companies/vendors/transactions) and a graph database, neither present.",
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
