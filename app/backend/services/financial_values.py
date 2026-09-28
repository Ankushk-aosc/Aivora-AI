"""Normalized extraction of financial figures from a question.

The baseline measured 116 questions where the needed figure was present in the
text and the pipeline still declined, because the old label-first/number-first
regex pairs in financial_router.py only matched a fixed set of phrasings. The
misses were things like "dividends paid are 362.00", "360.00 shares
outstanding", "Equity is 70% of capital at a cost of 8%".

Rather than adding more regexes per phrasing, this module separates the two
things that were tangled together:

1. WHAT the labels are - a data table, `FIELDS`, mapping a canonical field name
   to its surface spellings. Adding a synonym is a one-line data change.
2. HOW a figure binds to a label - one generic pass: find every label
   occurrence and every number in a line, then bind each number to its nearest
   label. Nearest-label binding handles both orders without separate patterns,
   so "net income is 220" and "360 shares outstanding" both work.

Two phrasings do not reduce to nearest-label binding and get their own small,
named parsers: a capital-structure sentence ("Equity is 70% of capital at a
cost of 8%, debt is the rest at 5%") and a growth span ("grew from 100 to 200
over 5 years"). They are explicit and documented rather than hidden inside a
general pattern.

Every figure keeps its provenance, so a caller can show its work and a test can
assert on where a value came from:

    FinancialValue(field="total_debt", value=3735.0, unit="currency",
                   currency="USD", scale=None, source_span="Total debt: 3,735.00",
                   confidence=0.95)
"""

import re
from dataclasses import dataclass, field as dataclass_field

# Canonical field -> surface spellings. Order inside a list does not matter;
# the matcher tries longer spellings first so "total debt" beats "debt" and
# "shareholders equity" beats "equity".
FIELDS = {
    "revenue": ["revenue", "revenues", "total revenue", "net sales", "sales",
                "turnover", "top line"],
    "cogs": ["cost of goods sold", "cogs", "cost of sales", "cost of revenue"],
    "gross_profit": ["gross profit", "gross income"],
    "operating_expenses": ["operating expenses", "opex", "operating expense",
                           "sg&a", "overheads"],
    "expenses": ["expenses", "total expenses", "costs", "total costs"],
    "depreciation": ["depreciation"],
    "amortization": ["amortization", "amortisation"],
    "depreciation_amortization": ["depreciation & amortization",
                                  "depreciation and amortization",
                                  "depreciation & amortisation",
                                  "depreciation and amortisation", "d&a"],
    "operating_income": ["operating income", "operating profit", "ebit"],
    "ebitda": ["ebitda"],
    "interest_expense": ["interest expense", "interest cost", "interest paid",
                         "interest"],
    "tax_rate": ["tax rate", "effective tax rate"],
    "tax": ["tax expense", "income tax", "taxes"],
    "net_income": ["net income", "net profit", "net earnings", "profit after tax",
                   "earnings"],
    "total_assets": ["total assets", "assets"],
    "total_liabilities": ["total liabilities", "liabilities"],
    "current_assets": ["current assets"],
    "current_liabilities": ["current liabilities"],
    "inventory": ["inventory", "inventories", "stock on hand"],
    "cash": ["cash", "cash and equivalents", "cash & equivalents"],
    "equity": ["shareholders equity", "shareholders' equity", "shareholder equity",
               "stockholders equity", "stockholders' equity", "total equity",
               "book value", "equity"],
    "total_debt": ["total debt", "debt", "borrowings"],
    "capex": ["capital expenditure", "capital expenditures", "capex"],
    "operating_cash_flow": ["operating cash flow", "cash flow from operations",
                            "cash from operations"],
    "free_cash_flow": ["free cash flow", "fcf"],
    "shares_outstanding": ["shares outstanding", "outstanding shares",
                           "shares in issue", "share count", "shares"],
    "earnings_per_share": ["earnings per share", "eps"],
    "price_per_share": ["share price", "price per share", "stock price",
                        "price of the share"],
    "market_cap": ["market capitalisation", "market capitalization", "market cap"],
    "enterprise_value": ["enterprise value", "ev"],
    "dividends": ["dividends paid", "dividend paid", "dividends", "dividend"],
    "nopat": ["nopat", "net operating profit after tax"],
    "invested_capital": ["invested capital"],
    "beginning_value": ["beginning value", "initial value", "starting value",
                        "opening value"],
    "ending_value": ["ending value", "final value", "closing value"],
    "years": ["years", "yrs"],
    "cost_of_equity": ["cost of equity"],
    "cost_of_debt": ["cost of debt"],
}

# Ratio and metric NAMES. A question asking for one of these wants a
# calculation, never a field lookup, so the router must not treat them as
# extractable fields even though they parse like labels.
DERIVED_NAMES = {
    "gross margin", "net margin", "net profit margin", "operating margin",
    "ebitda margin", "current ratio", "quick ratio", "debt-to-equity",
    "debt to equity", "debt/equity", "debt-to-ebitda", "debt to ebitda",
    "roe", "return on equity", "roa", "return on assets", "roic",
    "return on invested capital", "asset turnover", "interest coverage",
    "inventory turnover", "dividend payout ratio", "payout ratio", "wacc",
    "weighted average cost of capital", "cagr", "compound annual growth",
    "p/e ratio", "pe ratio", "price-to-earnings", "price to earnings",
    "ev/ebitda", "ev to ebitda", "net present value", "npv", "working capital",
    "revenue growth", "growth rate", "book value per share",
}

SCALES = {
    "crore": 1e7, "cr": 1e7, "lakh": 1e5, "thousand": 1e3, "k": 1e3,
    "million": 1e6, "mn": 1e6, "m": 1e6, "billion": 1e9, "bn": 1e9, "b": 1e9,
    "trillion": 1e12, "tn": 1e12,
}

CURRENCY_SYMBOLS = {"$": "USD", "₹": "INR", "€": "EUR", "£": "GBP", "¥": "JPY"}

_NUMBER = re.compile(
    r"(?P<currency>[₹$€£¥])?\s*"
    r"(?P<number>-?\d{1,3}(?:,\d{3})+(?:\.\d+)?|-?\d+(?:\.\d+)?)"
    r"\s*(?P<percent>%)?"
    # The scale word carries its own boundary. With a trailing \b outside the
    # optional group, "20%." backtracked and dropped the percent sign, so a tax
    # rate of 20% parsed as 20 instead of 0.20.
    r"(?:\s*(?P<scale>crore|cr|lakh|thousand|million|mn|billion|bn|trillion|tn)\b)?",
    re.IGNORECASE,
)


def _label_pattern():
    """One alternation over every spelling, longest first so the longest wins."""
    spellings = sorted({s for group in FIELDS.values() for s in group},
                       key=len, reverse=True)
    escaped = [re.escape(s) for s in spellings]
    return re.compile(r"(?<![a-z])(" + "|".join(escaped) + r")(?![a-z])",
                      re.IGNORECASE)


# How much further a label after the number is treated as being.
_BACKWARD_PREFERENCE = 6

_LABELS = _label_pattern()
_SPELLING_TO_FIELD = {spelling.lower(): canonical
                      for canonical, group in FIELDS.items()
                      for spelling in group}


@dataclass
class FinancialValue:
    field: str
    value: float
    unit: str = "currency"          # currency | percent | count | years
    currency: str = None
    scale: str = None
    source_span: str = ""
    confidence: float = 0.9
    notes: list = dataclass_field(default_factory=list)

    def as_fraction(self) -> float:
        """A percentage as a fraction; other units unchanged."""
        return self.value / 100.0 if self.unit == "percent" else self.value

    def to_dict(self):
        return {"field": self.field, "value": self.value, "unit": self.unit,
                "currency": self.currency, "scale": self.scale,
                "source_span": self.source_span,
                "confidence": round(self.confidence, 2)}


def _numbers_in(line: str):
    for match in _NUMBER.finditer(line):
        raw = match.group("number").replace(",", "")
        try:
            value = float(raw)
        except ValueError:
            continue
        scale = (match.group("scale") or "").lower() or None
        if scale:
            value *= SCALES[scale]
        yield {
            "start": match.start(), "end": match.end(), "value": value,
            "percent": bool(match.group("percent")),
            "currency": CURRENCY_SYMBOLS.get(match.group("currency") or ""),
            "scale": scale, "span": match.group(0).strip(),
        }


def _labels_in(line: str):
    for match in _LABELS.finditer(line):
        yield {"start": match.start(), "end": match.end(),
               "field": _SPELLING_TO_FIELD[match.group(1).lower()],
               "text": match.group(1)}


def _bind_line(line: str):
    """Bind each number in one line to its nearest label.

    Nearest-label binding is what removes the need for a pattern per phrasing:
    "net income is 220" binds backwards, "360 shares outstanding" binds
    forwards, and a statement line "  Total debt: 3,735.00" binds backwards
    too. Pairs are taken in order of increasing distance, and a label and a
    number are each used once, so "Revenue 100 COGS 60" cannot cross-bind.
    """
    numbers = list(_numbers_in(line))
    labels = list(_labels_in(line))
    if not numbers or not labels:
        return []

    pairs = []
    for n_index, number in enumerate(numbers):
        for l_index, label in enumerate(labels):
            if label["end"] <= number["start"]:
                distance = number["start"] - label["end"]
                direction = "label_before"
            elif number["end"] <= label["start"]:
                # A label after the number is the less common order, so it is
                # penalised: in "Current assets are 1,060.00, inventory is
                # 200.00" the comma puts "inventory" two characters after the
                # figure and it would otherwise steal it from "current assets".
                # Adjacent forward labels ("360.00 shares outstanding") still
                # win, because nothing else is closer.
                distance = label["start"] - number["end"] + _BACKWARD_PREFERENCE
                direction = "label_after"
            else:
                continue                      # label and number overlap
            # Text between them that is neither connector nor punctuation
            # means they are probably unrelated ("revenue rose while 5 plants").
            between = (line[label["end"]:number["start"]] if direction == "label_before"
                       else line[number["end"]:label["start"]])
            if len(between) > 24:
                continue
            pairs.append((distance, n_index, l_index, direction, between))

    pairs.sort(key=lambda p: (p[0], p[1]))
    used_numbers, used_labels, bound = set(), set(), []
    for distance, n_index, l_index, direction, between in pairs:
        if n_index in used_numbers or l_index in used_labels:
            continue
        used_numbers.add(n_index)
        used_labels.add(l_index)
        number, label = numbers[n_index], labels[l_index]
        if number["percent"]:
            unit = "percent"
        elif label["field"] == "years":
            unit = "years"
        elif label["field"] == "shares_outstanding":
            unit = "count"
        else:
            unit = "currency"
        # Confidence drops with distance and with unusual word order.
        confidence = 0.95 if distance <= 3 else 0.85 if distance <= 12 else 0.7
        if direction == "label_after":
            confidence -= 0.05
        span_start = min(label["start"], number["start"])
        span_end = max(label["end"], number["end"])
        bound.append(FinancialValue(
            field=label["field"], value=number["value"], unit=unit,
            currency=number["currency"], scale=number["scale"],
            source_span=line[span_start:span_end].strip(),
            confidence=round(confidence, 2),
        ))
    return bound


# --- the two phrasings that nearest-label binding cannot express -----------

_CAPITAL_STRUCTURE = re.compile(
    r"equity\s+is\s+(?P<equity_weight>\d+(?:\.\d+)?)\s*%\s*of\s*capital"
    r".{0,40}?cost\s+of\s+(?P<cost_of_equity>\d+(?:\.\d+)?)\s*%"
    r".{0,60}?debt\s+is\s+the\s+rest\s+at\s+(?P<cost_of_debt>\d+(?:\.\d+)?)\s*%",
    re.IGNORECASE | re.DOTALL,
)

_GROWTH_SPAN = re.compile(
    r"(?:grew|rose|increased|went)\s+from\s+[₹$€£]?\s*"
    r"(?P<begin>[\d,]+(?:\.\d+)?)\s*(?:to|up\s+to)\s+[₹$€£]?\s*"
    r"(?P<end>[\d,]+(?:\.\d+)?)"
    r"(?:.{0,30}?over\s+(?P<years>\d+(?:\.\d+)?)\s*(?:years|yrs|periods))?",
    re.IGNORECASE | re.DOTALL,
)


def _parse_capital_structure(text: str):
    """Returns (values, matched_span) so the caller can blank the span out."""
    match = _CAPITAL_STRUCTURE.search(text)
    if not match:
        return [], None
    equity_weight = float(match.group("equity_weight"))
    out = [
        FinancialValue("equity_weight", equity_weight, "percent",
                       source_span=match.group(0)[:60], confidence=0.9),
        FinancialValue("debt_weight", round(100.0 - equity_weight, 6), "percent",
                       source_span="debt is the rest", confidence=0.85,
                       notes=["derived as 100% - equity weight"]),
        FinancialValue("cost_of_equity", float(match.group("cost_of_equity")),
                       "percent", source_span=match.group(0)[:60], confidence=0.9),
        FinancialValue("cost_of_debt", float(match.group("cost_of_debt")),
                       "percent", source_span=match.group(0)[-40:], confidence=0.9),
    ]
    return out, match.span()


def _parse_growth_span(text: str):
    """Returns (values, matched_span); see _parse_capital_structure."""
    match = _GROWTH_SPAN.search(text)
    if not match:
        return [], None
    out = [
        FinancialValue("beginning_value", float(match.group("begin").replace(",", "")),
                       source_span=match.group(0)[:50], confidence=0.9),
        FinancialValue("ending_value", float(match.group("end").replace(",", "")),
                       source_span=match.group(0)[:60], confidence=0.9),
    ]
    if match.group("years"):
        out.append(FinancialValue("years", float(match.group("years")), "years",
                                  source_span=match.group(0)[-30:], confidence=0.9))
    return out, match.span()


def parse_financial_values(text: str):
    """All figures found in `text`, as FinancialValue records.

    Lines are parsed independently so a labelled statement block cannot bind a
    label on one line to a number on another.
    """
    if not text:
        return []

    # The two special phrasings are parsed first and their spans blanked out,
    # because the generic pass would otherwise also bind pieces of them and
    # produce nonsense: "debt is the rest at 5%" became total_debt = 5.
    found, remaining = [], text
    for parser in (_parse_capital_structure, _parse_growth_span):
        values, span = parser(remaining)
        if span:
            found.extend(values)
            start, end = span
            blanked = re.sub(r"[^\n]", " ", remaining[start:end])
            remaining = remaining[:start] + blanked + remaining[end:]

    for line in remaining.splitlines():
        found.extend(_bind_line(line))

    # First occurrence of a field wins, matching the old parser's setdefault
    # behaviour, but a higher-confidence later record replaces a weak one.
    best = {}
    for value in found:
        current = best.get(value.field)
        if current is None or value.confidence > current.confidence + 0.1:
            best[value.field] = value
    return list(best.values())


def values_dict(values):
    """FinancialValue list -> {field: number}, percentages as fractions where
    the field is a rate or a weight (what a calculator wants)."""
    as_fraction = {"tax_rate", "equity_weight", "debt_weight", "cost_of_equity",
                   "cost_of_debt", "discount_rate", "growth_rate"}
    out = {}
    for value in values:
        out[value.field] = (value.as_fraction() if value.field in as_fraction
                            else value.value)
    return out


def asked_field(question: str):
    """The field a question asks for, or None if it asks for something derived.

    "What is Total debt?" -> "total_debt"
    "What is the gross margin?" -> None (a ratio: that is a calculation)
    """
    if not question:
        return None
    # Only the asking clause matters, not the figures supplied with it. Take
    # the last sentence that actually asks something: for a one-line question
    # like "Market capitalisation is 12,000, total debt is 3,000 ... What is
    # enterprise value?" the data and the question share a line, and reading
    # the whole line makes "market capitalisation" look like the subject.
    lines = [line for line in question.strip().splitlines() if line.strip()]
    clause = lines[-1] if lines else question
    sentences = [s for s in re.split(r"(?<=[.?!])\s+", clause) if s.strip()]
    asking = [s for s in sentences if "?" in s] or sentences[-1:]
    clause = asking[-1] if asking else clause
    lowered = clause.lower()

    # A derived metric name anywhere in the clause means this is not a lookup.
    for name in DERIVED_NAMES:
        if name in lowered:
            return None

    matches = list(_LABELS.finditer(clause))
    if not matches:
        return None
    # The longest label in the clause is the subject ("total debt", not "debt").
    best = max(matches, key=lambda m: len(m.group(1)))
    # A figure attached to the label means it is an input, not the question.
    if re.search(re.escape(best.group(1)) + r"\s*(?:is|are|was|were|=|:|of)?\s*"
                 r"[₹$€£]?\s*\d", lowered):
        return None
    return _SPELLING_TO_FIELD[best.group(1).lower()]
