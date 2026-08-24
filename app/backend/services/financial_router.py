"""Financial query router (Part 20 / §36).

Classifies an incoming request so the chat layer knows which component
should answer it:

  GENERAL             -> ordinary language, answered by the model
  FINANCIAL_KNOWLEDGE -> financial concept/definition, answered by the model
  NUMERICAL           -> arithmetic, answered by the deterministic calculator
  DOCUMENT            -> question about an uploaded document, answered via RAG
  LIVE_DATA           -> current market data; never fabricated
  UNKNOWN             -> could not classify

This is a deterministic rule-based classifier, not a model: routing must
be predictable and inspectable.
"""

import re
from dataclasses import dataclass, field
from enum import Enum


class Route(str, Enum):
    GENERAL = "GENERAL"
    FINANCIAL_KNOWLEDGE = "FINANCIAL_KNOWLEDGE"
    NUMERICAL = "NUMERICAL"
    DOCUMENT = "DOCUMENT"
    LIVE_DATA = "LIVE_DATA"
    UNKNOWN = "UNKNOWN"


LIVE_DATA_UNAVAILABLE = "Current market data is not available."

# Terms that signal the finance domain.
FINANCIAL_TERMS = [
    "ebitda", "ebit", "revenue", "profit", "margin", "equity", "asset", "liability",
    "cash flow", "fcf", "eps", "p/e", "pe ratio", "roe", "roa", "roic", "capex",
    "balance sheet", "income statement", "cash flow statement", "dividend",
    "amortization", "amortisation", "depreciation", "working capital", "cagr",
    "gross profit", "net income", "operating income", "valuation", "ev/ebitda",
    "debt", "leverage", "current ratio", "shareholder", "yoy", "qoq", "nopat",
    "invested capital", "free cash flow", "net profit", "turnover", "solvency",
]

# Explicit requests to compute something.
CALC_VERBS = [
    "calculate", "compute", "what is the margin", "work out", "derive",
    "how much is", "find the", "determine the",
]

# Live/current market data.
LIVE_DATA_TERMS = [
    "current stock price", "stock price", "share price", "today's price",
    "current price", "market price", "latest price", "trading at", "quote for",
    "market cap right now", "current market", "live price", "real-time",
    "right now", "as of today", "latest quarter results",
]

# References to an uploaded document.
DOCUMENT_TERMS = [
    "this report", "the report", "this document", "the document", "attached",
    "uploaded", "this filing", "the filing", "annual report", "10-k", "10k",
    "10-q", "quarterly report", "this pdf", "the pdf", "in the document",
    "according to the", "this statement", "these financials",
]

# Numbers, currency amounts, percentages.
_NUMBER_RE = re.compile(r"\d")
_ASSIGNMENT_RE = re.compile(
    r"(?:revenue|ebitda|profit|income|equity|assets?|liabilit(?:y|ies)|sales|"
    r"capex|cash\s*flow|shares?|price|debt|expenses?)\s*(?:is|=|:|of|was|were)?\s*"
    r"[₹$€£]?\s*[\d,]+(?:\.\d+)?",
    re.IGNORECASE,
)
_CURRENCY_RE = re.compile(r"[₹$€£]\s*[\d,]+(?:\.\d+)?", re.IGNORECASE)


@dataclass
class RouteDecision:
    route: Route
    confidence: float
    reasons: list = field(default_factory=list)
    matched_terms: list = field(default_factory=list)

    def to_dict(self):
        return {
            "route": self.route.value,
            "confidence": round(self.confidence, 3),
            "reasons": self.reasons,
            "matched_terms": self.matched_terms,
        }


def _find(text: str, terms) -> list:
    return [t for t in terms if t in text]


def classify(query: str, has_document: bool = False) -> RouteDecision:
    """Classify a user query into a Route.

    has_document: whether a document is currently loaded in the session,
    which makes DOCUMENT routing possible.
    """
    if not query or not query.strip():
        return RouteDecision(Route.UNKNOWN, 0.0, ["empty query"])

    text = query.lower().strip()
    reasons = []

    live_hits = _find(text, LIVE_DATA_TERMS)
    doc_hits = _find(text, DOCUMENT_TERMS)
    fin_hits = _find(text, FINANCIAL_TERMS)
    calc_hits = _find(text, CALC_VERBS)
    assignments = _ASSIGNMENT_RE.findall(query)
    currency = _CURRENCY_RE.findall(query)
    has_digits = bool(_NUMBER_RE.search(query))

    # 1. Live market data wins: it must never be answered from model memory.
    if live_hits:
        reasons.append(f"matched live-data phrase(s): {live_hits}")
        return RouteDecision(Route.LIVE_DATA, 0.9, reasons, live_hits)

    # 2. Explicit reference to a document.
    if doc_hits:
        reasons.append(f"matched document reference(s): {doc_hits}")
        if not has_document:
            reasons.append("no document is loaded in this session")
        return RouteDecision(Route.DOCUMENT, 0.85 if has_document else 0.6, reasons, doc_hits)

    # 3. Numerical: needs both a computation intent (or supplied figures)
    #    and actual numbers to work with.
    numeric_signal = bool(assignments) or bool(currency) or (has_digits and calc_hits)
    if numeric_signal and (calc_hits or assignments):
        reasons.append("numeric inputs present with a calculation intent")
        if assignments:
            reasons.append(f"parsed value assignments: {assignments}")
        return RouteDecision(Route.NUMERICAL, 0.9, reasons, calc_hits + fin_hits)

    # 4. Financial concept question.
    if fin_hits:
        reasons.append(f"matched financial term(s): {fin_hits}")
        return RouteDecision(Route.FINANCIAL_KNOWLEDGE, 0.8, reasons, fin_hits)

    # 5. Otherwise general language.
    reasons.append("no financial, numeric, document, or live-data signal")
    return RouteDecision(Route.GENERAL, 0.5, reasons)


def extract_financial_values(query: str) -> dict:
    """Pull named numeric values out of a query, e.g.
    "Revenue is ₹500 crore and EBITDA is ₹100 crore" ->
    {"revenue": 500.0, "ebitda": 100.0}

    Returns only what was actually found; never guesses a missing value.
    Crore/lakh/million/billion multipliers are applied when present.
    """
    values = {}
    pattern = re.compile(
        r"(revenue|ebitda|gross profit|net income|operating income|profit|equity|"
        r"total assets|assets|liabilities|shares outstanding|shares|price|debt|"
        r"capex|capital expenditure|operating cash flow|cash flow)"
        r"\s*(?:is|=|:|of|was|were)?\s*"
        r"[₹$€£]?\s*([\d,]+(?:\.\d+)?)\s*"
        r"(crore|cr|lakh|million|mn|bn|billion|thousand|k)?",
        re.IGNORECASE,
    )
    multipliers = {
        "crore": 1e7, "cr": 1e7, "lakh": 1e5,
        "million": 1e6, "mn": 1e6, "bn": 1e9, "billion": 1e9,
        "thousand": 1e3, "k": 1e3,
    }
    for label, number, scale in pattern.findall(query):
        key = label.lower().strip().replace(" ", "_")
        amount = float(number.replace(",", ""))
        if scale:
            amount *= multipliers[scale.lower()]
        values.setdefault(key, amount)
    return values
