"""Financial query router (Part 20 / §36).

Classifies an incoming request so the chat layer knows which component
should answer it:

  GENERAL             -> ordinary language, answered by the model
  FINANCIAL_KNOWLEDGE -> financial concept/definition (CONCEPT intent)
  EXTRACTION          -> the figure asked for is stated in the question; read
                         it back rather than computing something else
  NUMERICAL           -> arithmetic, answered by the deterministic calculator
                         (CALCULATION and multi-step REASONING intents)
  DOCUMENT            -> question about an uploaded document, answered via RAG
                         (RETRIEVAL intent over uploaded text)
  INTERPRETATION      -> an observation the asker wants explained. Decided
                         BEFORE any figure is noticed, because "Debt/EBITDA rose
                         from 2x to 5x - what does that indicate?" contains
                         numbers and is not a calculation
  CURRENT_DATA        -> a value that changes with time and is not held here;
                         abstained, never fabricated (was LIVE_DATA)
  UNKNOWN             -> could not classify

ABSTENTION is not a route: it is the outcome when a route's handler finds the
information absent, so an honest refusal comes from the component that knows
what was missing.

Classification order is deliberate: current data, then document, then
EXTRACTION, then calculation, then concept. EXTRACTION must precede calculation
because the baseline measured 77 questions where a figure was stated in the text
and the calculator answered with an unrelated ratio computed from the other
figures - "What is Total debt?" answered with an EBITDA margin.

This is a deterministic rule-based classifier, not a model: routing must
be predictable and inspectable.
"""

import re
from dataclasses import dataclass, field
from enum import Enum

from app.backend.services.financial_values import asked_field, parse_financial_values
from app.backend.services.question_focus import is_interpretive


class Route(str, Enum):
    GENERAL = "GENERAL"
    EXTRACTION = "EXTRACTION"
    INTERPRETATION = "INTERPRETATION"
    CURRENT_DATA = "CURRENT_DATA"
    FINANCIAL_KNOWLEDGE = "FINANCIAL_KNOWLEDGE"
    NUMERICAL = "NUMERICAL"
    DOCUMENT = "DOCUMENT"
    # The previous name for CURRENT_DATA. Kept so stored evaluation results, the
    # frontend and older callers keep resolving; classify() returns CURRENT_DATA.
    LIVE_DATA = "LIVE_DATA"
    UNKNOWN = "UNKNOWN"


LIVE_DATA_UNAVAILABLE = (
    "I cannot verify the current value from the available data. This question "
    "asks for a figure that changes over time, and no verified live source is "
    "connected, so no number is given here - a definition of the measure would "
    "not be an answer to what was asked."
)

# Terms that signal the finance domain.
FINANCIAL_TERMS = [
    "ebitda", "ebit", "revenue", "profit", "margin", "equity", "asset", "liability",
    "cash flow", "fcf", "eps", "p/e", "pe ratio", "roe", "roa", "roic", "capex",
    "liabilities", "current assets", "current liabilities",
    "balance sheet", "income statement", "cash flow statement", "dividend",
    "amortization", "amortisation", "depreciation", "working capital", "cagr",
    "gross profit", "net income", "operating income", "valuation", "ev/ebitda",
    "debt", "leverage", "current ratio", "shareholder", "yoy", "qoq", "nopat",
    "invested capital", "free cash flow", "net profit", "turnover", "solvency",
    "expenses", "expense", "costs", "stock", "shares",
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
    # Added after the baseline: "What is the current Federal Reserve interest
    # rate today?" matched none of the above and was answered with a glossary
    # DEFINITION of interest rates, which is not an answer to a question about
    # today's value. A definition of X must never stand in for the current
    # value of X.
    "today", "tomorrow", "yesterday", "currently", "at the moment",
    "this quarter", "last quarter", "this month", "last month", "this week",
    "last week", "current rate", "current interest rate", "current fed",
    "latest figures", "up to date", "so far this year", "year to date",
    "at present", "these days",
]

# Words that make a time reference part of a definition rather than a request
# for a current value ("What is a stock split?" ... "over time").
_TIMELESS_CONTEXT = ["over time", "at a point in time", "point in time"]

# Asking for the present value of a named measure. A list of literal phrases
# cannot cover this - "the current inflation rate" and "today's CPI" were both
# answered from the glossary - so the shape of the request is matched instead.
_CURRENT_VALUE_PATTERNS = [
    re.compile(r"\bcurrent\s+(?:[a-z]+\s+){0,2}(rate|price|value|level|yield|"
               r"figure|reading|number|inflation|cpi)\b", re.IGNORECASE),
    re.compile(r"\btoday'?s\s+[a-z]+", re.IGNORECASE),
    re.compile(r"\b(?:what|where)\s+(?:is|are)\b[^?]*\b(now|today|currently|"
               r"at present|right now|this week|this month)\b", re.IGNORECASE),
    re.compile(r"\blatest\s+(?:[a-z]+\s+){0,2}(rate|price|figure|number|result|"
               r"reading)\b", re.IGNORECASE),
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
    r"(?:enterprise\s+value|invested\s+capital|nopat|earnings\s+per\s+share|eps|"
    r"revenue|ebitda|profit|income|equity|assets?|liabilit(?:y|ies)|sales|"
    r"capex|cash\s*flow|shares?|price|debt|expenses?|costs?)\s*(?:is|are|=|:|of|was|were)?\s*"
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

    # A live-data phrase followed by a figure ("share price 80") supplies the
    # value rather than asking for today's market data, so it is an input to
    # a calculation, not a live-data request.
    live_hits = [t for t in _find(text, LIVE_DATA_TERMS)
                 if not re.search(re.escape(t) + r"\s*(?:is|of|=|:|was)?\s*[₹$€£]?\s*\d", text)]
    if not live_hits and any(p.search(query) for p in _CURRENT_VALUE_PATTERNS):
        matched = [p.pattern[:28] for p in _CURRENT_VALUE_PATTERNS if p.search(query)]
        live_hits = ["asks for a present value"]
        reasons.append(f"matches a current-value request: {matched}")

    if any(phrase in text for phrase in _TIMELESS_CONTEXT):
        live_hits = [t for t in live_hits if t not in ("today", "currently",
                                                      "at the moment", "these days")]
    doc_hits = _find(text, DOCUMENT_TERMS)
    fin_hits = _find(text, FINANCIAL_TERMS)
    calc_hits = _find(text, CALC_VERBS)
    assignments = _ASSIGNMENT_RE.findall(query)
    currency = _CURRENCY_RE.findall(query)
    has_digits = bool(_NUMBER_RE.search(query))
    rich_values = parse_financial_values(query)

    # 1. Live market data wins: it must never be answered from model memory.
    if live_hits:
        reasons.append(f"matched live-data phrase(s): {live_hits}")
        return RouteDecision(Route.CURRENT_DATA, 0.9, reasons, live_hits)

    # 2. Explicit reference to a document.
    if doc_hits:
        definitional = any(p in text for p in ("what is a", "what is an", "what is the purpose",
                                               "what are the", "define", "what does a",
                                               # "What is the MD&A section of an annual
                                               # report?" asks what a section IS, while
                                               # "What is the revenue in this report?"
                                               # asks to read one, so the test is the
                                               # word "section", not "what is the".
                                               "section of", "section in",
                                               "what is the md&a", "section is"))
        # An interpretive question that merely mentions a report type ("Why do
        # regulators require quarterly reporting?") was answered "no document is
        # currently loaded", which answers nothing. is_interpretive is imported
        # at module level; a local import here made it a local name for the whole
        # function and broke the interpretation check below.
        if not has_document and (definitional or is_interpretive(query)):
            # "What is a 10-K?" names a filing type but asks for a definition.
            # Answering "no document is loaded" is a non-answer, which is what
            # the baseline measured for the reporting category.
            reasons.append(f"filing term {doc_hits} in a definitional question, "
                           "and no document is loaded")
            return RouteDecision(Route.FINANCIAL_KNOWLEDGE, 0.75, reasons, doc_hits)
        reasons.append(f"matched document reference(s): {doc_hits}")
        if not has_document:
            reasons.append("no document is loaded in this session")
        return RouteDecision(Route.DOCUMENT, 0.85 if has_document else 0.6, reasons, doc_hits)

    # 3. Interpretation, BEFORE anything looks at the figures. This ordering is
    #    the point: a rule of the form "two numbers -> calculator" sends
    #    "Debt/EBITDA rose from 2x to 5x. What does that indicate?" to the
    #    calculator, which then reports Debt/EBITDA = 5.00 from the 2 and the 5.
    #    What the asker requested outranks what the text happens to contain.
    #    An explicit instruction to compute ("calculate", "work out") is not an
    #    interpretation request even when phrased with "why".
    explicit_compute_verbs = [v for v in calc_hits
                              if v in ("calculate", "compute", "work out", "derive")]
    if is_interpretive(query) and not explicit_compute_verbs:
        reasons.append("the question asks for an explanation of a stated "
                       "observation, so the figures in it are context, not inputs")
        return RouteDecision(Route.INTERPRETATION, 0.85, reasons, fin_hits)

    # 4. Extraction: the question names a field whose value is stated right
    #    there. Reading it back is deterministic and cannot be improved on by
    #    computing something else from the neighbouring figures.
    wanted = asked_field(query)
    if wanted and wanted in {v.field for v in rich_values}:
        reasons.append(f"question asks for '{wanted}', which is stated in the text")
        return RouteDecision(Route.EXTRACTION, 0.95, reasons, [wanted])

    # 5. Numerical: needs both a computation intent (or supplied figures)
    #    and actual numbers to work with.
    # Two or more labelled figures ARE the numeric signal, even when no
    # calculation verb appears: "An investment grew from 400.00 to 644.20 over
    # 3 years. What is the CAGR?" matched none of the old assignment patterns
    # and was answered with a definition of CAGR instead of a number.
    named_figures = [v for v in rich_values if v.confidence >= 0.7]
    numeric_signal = (bool(assignments) or bool(currency)
                      or (has_digits and calc_hits) or len(named_figures) >= 2)
    if numeric_signal and (calc_hits or assignments or len(named_figures) >= 2):
        if len(named_figures) >= 2:
            reasons.append("two or more labelled figures were parsed: "
                           + ", ".join(f"{v.field}={v.value:g}" for v in named_figures[:6]))
        reasons.append("numeric inputs present with a calculation intent")
        if assignments:
            reasons.append(f"parsed value assignments: {assignments}")
        return RouteDecision(Route.NUMERICAL, 0.9, reasons, calc_hits + fin_hits)

    # 5b. An explicit instruction to compute, with no figures supplied. This
    #     belongs to the calculator so that it can say what is missing:
    #     "Calculate ROE." was being answered with the DEFINITION of ROE, which
    #     is not a refusal and not an answer.
    explicit_compute = [v for v in calc_hits
                        if v in ("calculate", "compute", "work out", "derive")]
    if explicit_compute and not has_digits:
        reasons.append(f"explicit calculation request {explicit_compute} with no figures")
        return RouteDecision(Route.NUMERICAL, 0.8, reasons, explicit_compute)

    # 6. Financial concept question.
    if fin_hits:
        reasons.append(f"matched financial term(s): {fin_hits}")
        return RouteDecision(Route.FINANCIAL_KNOWLEDGE, 0.8, reasons, fin_hits)

    # 7. Otherwise general language.
    reasons.append("no financial, numeric, document, or live-data signal")
    return RouteDecision(Route.GENERAL, 0.5, reasons)


def extract_financial_values(query: str) -> dict:
    """Pull named numeric values out of a query, e.g.
    "Revenue is ₹500 crore and EBITDA is ₹100 crore" ->
    {"revenue": 5000000000.0, "ebitda": 1000000000.0}

    Returns only what was actually found; never guesses a missing value.
    Crore/lakh/million/billion multipliers are applied when present.
    """
    values = {}
    multipliers = {
        "crore": 1e7, "cr": 1e7, "lakh": 1e5,
        "million": 1e6, "mn": 1e6, "m": 1e6,
        "billion": 1e9, "bn": 1e9, "b": 1e9,
        "thousand": 1e3, "k": 1e3,
    }
    label_aliases = {
        "costs": "expenses",
        "cost_of_goods_sold": "cogs",
        "opex": "operating_expenses",
        "shareholder_equity": "equity",
        "shareholders_equity": "equity",
        "average_shareholder_equity": "equity",
        "total_debt": "debt",
        "total_assets": "total_assets",
        "capital_expenditure": "capex",
        "beginning_value": "beginning_value",
        "initial_value": "beginning_value",
        "ending_value": "ending_value",
        "final_value": "ending_value",
        "share_price": "price_per_share",
        "eps": "earnings_per_share",
    }

    # Pattern A: <Label> ... <Number> <Scale>
    pattern_label_first = re.compile(
        r"(enterprise value|price per share|share price|earnings per share|eps|nopat|invested capital|"
        r"depreciation|amortisation|amortization|"
        r"revenue|ebitda|gross profit|net income|operating income|operating expenses|expenses|costs|cogs|cost of goods sold|interest expense|profit|shareholder equity|shareholders equity|average shareholder equity|equity|"
        r"total assets|assets|liabilities|current assets|current liabilities|shares outstanding|shares|price|total debt|debt|"
        r"capex|capital expenditure|operating cash flow|cash flow|initial value|beginning value|final value|ending value|prior revenue|current revenue)"
        r"\s*(?:is|are|=|:|of|was|were|has)?\s*"
        r"[₹$€£]?\s*([\d,]+(?:\.\d+)?)\s*"
        r"(crore|cr|lakh|million|mn|billion|bn|thousand|k)?",
        re.IGNORECASE,
    )

    for label, number, scale in pattern_label_first.findall(query):
        number_str = number.replace(",", "").strip()
        if not number_str:
            continue
        key = label.lower().strip().replace(" ", "_")
        key = label_aliases.get(key, key)
        amount = float(number_str)
        if scale and scale.lower() in multipliers:
            amount *= multipliers[scale.lower()]
        values.setdefault(key, amount)

    # Pattern B: <Number> <Scale> in/of <Label> (e.g., "15 million in net income")
    pattern_number_first = re.compile(
        r"[₹$€£]?\s*([\d,]+(?:\.\d+)?)\s*"
        r"(crore|cr|lakh|million|mn|billion|bn|thousand|k)?\s*"
        r"(?:in|of|for|as)?\s*"
        r"(revenue|ebitda|gross profit|net income|net profit|operating income|operating expenses|expenses|costs|cogs|cost of goods sold|interest expense|profit|shareholder equity|shareholders equity|equity|total debt|debt|capex|assets|liabilities)",
        re.IGNORECASE,
    )

    for number, scale, label in pattern_number_first.findall(query):
        number_str = number.replace(",", "").strip()
        if not number_str:
            continue
        key = label.lower().strip().replace(" ", "_")
        key = label_aliases.get(key, key)
        amount = float(number_str)
        if scale and scale.lower() in multipliers:
            amount *= multipliers[scale.lower()]
        values.setdefault(key, amount)

    # Pattern C: Historical revenue growth (e.g., "revenue was 80 million last year and 100 million this year")
    growth_match = re.search(
        r"revenue\s+was\s+[₹$€£]?\s*([\d,]+(?:\.\d+)?)\s*(crore|cr|lakh|million|mn|bn|billion|k)?\s*(?:last year|prior year|previously|in year 1).*?(?:and|to)\s+[₹$€£]?\s*([\d,]+(?:\.\d+)?)\s*(crore|cr|lakh|million|mn|bn|billion|k)?\s*(?:this year|current year|now|in year 2)",
        query, re.IGNORECASE,
    )
    if growth_match:
        p_num, p_scale, c_num, c_scale = growth_match.groups()
        p_amt = float(p_num.replace(",", ""))
        if p_scale and p_scale.lower() in multipliers:
            p_amt *= multipliers[p_scale.lower()]
        c_amt = float(c_num.replace(",", ""))
        if c_scale and c_scale.lower() in multipliers:
            c_amt *= multipliers[c_scale.lower()]
        values["prior_revenue"] = p_amt
        values["current_revenue"] = c_amt
        values["revenue"] = c_amt

    # Pattern D: Years / periods for CAGR (e.g., "over 5 years")
    years_match = re.search(r"(?:over|in|for)\s*([\d]+(?:\.\d+)?)\s*(?:years|yrs|periods)", query, re.IGNORECASE)
    if years_match:
        values["years"] = float(years_match.group(1))

    # Anything the patterns above missed, from the normalized parser in
    # financial_values.py. The baseline measured 116 questions declined with
    # the needed figure present in the text: "dividends paid are 362.00",
    # "360.00 shares outstanding", "Equity is 70% of capital at a cost of 8%".
    # The patterns above are kept and take precedence, so no behaviour that
    # already worked changes; this only fills gaps.
    from app.backend.services.financial_values import (
        parse_financial_values, values_dict,
    )

    for key, amount in values_dict(parse_financial_values(query)).items():
        values.setdefault(key, amount)
        # Both spellings, because the calculator's intent table predates the
        # canonical names used by financial_values.FIELDS.
        for alias in _LEGACY_ALIASES.get(key, ()):
            values.setdefault(alias, amount)

    return values


# canonical name in financial_values.FIELDS -> names the calculator already uses
_LEGACY_ALIASES = {
    "total_liabilities": ("liabilities",),
    "shares_outstanding": ("shares",),
    "total_debt": ("debt",),
    "capex": ("capital_expenditure",),
    "operating_income": ("ebit",),
    "equity": ("shareholders_equity",),
}
