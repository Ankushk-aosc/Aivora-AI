"""Phase 4: the tool pipeline. The model does only what it is good at.

Division of labour, from the brief:

  the model   reads the context and says WHICH field a question wants, and copies
              the value with the span it came from
  Python      validates that span, selects the formula, and does every piece of
              arithmetic
  rules       abstain whenever validation fails or an operand is missing

The model never performs arithmetic. Not once, not for "simple" cases - the
calculation path extracts operands and hands them to
tools/financial_calculator.py.

Works with any callable `llm(prompt) -> str`, so the same pipeline wraps a
Hugging Face instruct model, Aivora, or a stub in tests.

    from pipeline.tool_pipeline import ToolPipeline
    pipe = ToolPipeline(llm)
    result = pipe.answer(question, context)      # -> PipelineAnswer
"""

import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from pipeline.schema import (  # noqa: E402
    EXTRACTION_SCHEMA, INSUFFICIENT, PipelineAnswer, abstention, normalise,
    parse_json_object, validate_extraction,
)

# operation -> (operands in order, python callable, unit)
OPERATIONS = {
    "difference": (("a", "b"), lambda a, b: a - b, ""),
    "sum": (("a", "b"), lambda a, b: a + b, ""),
    "ratio": (("a", "b"), lambda a, b: a / b, "x"),
    "percentage_of": (("a", "b"), lambda a, b: 100.0 * a / b, "%"),
    "margin": (("part", "whole"), lambda part, whole: 100.0 * part / whole, "%"),
    "difference_margin": (("a", "b"),
                          lambda a, b: 100.0 * (a - b) / a, "%"),
    "percentage_change": (("current", "prior"),
                          lambda current, prior: 100.0 * (current - prior) / prior,
                          "%"),
    "enterprise_value": (("market_cap", "debt", "cash"),
                         lambda market_cap, debt, cash: market_cap + debt - cash, ""),
}

# question wording -> (operation, the operand labels to extract, in order)
# Chosen by rule rather than by the model, so the arithmetic cannot drift.
FORMULAS = [
    (("gross margin",), "difference_margin", ("revenue", "cost of sales")),
    (("net profit margin", "net margin"), "margin", ("net income", "revenue")),
    (("ebitda margin",), "margin", ("ebitda", "revenue")),
    (("operating margin",), "margin", ("operating profit", "revenue")),
    (("current ratio",), "ratio", ("current assets", "current liabilities")),
    (("quick ratio",), "quick", ("current assets", "inventory",
                                 "current liabilities")),
    (("return on equity", "roe"), "margin", ("net income", "shareholders' equity")),
    (("return on assets", "roa"), "margin", ("net income", "total assets")),
    (("debt-to-equity", "debt to equity", "gearing"), "ratio",
     ("total debt", "shareholders' equity")),
    (("debt-to-ebitda", "debt to ebitda"), "ratio", ("total debt", "ebitda")),
    (("interest coverage", "interest cover"), "ratio",
     ("ebit", "interest expense")),
    (("asset turnover",), "ratio", ("revenue", "total assets")),
    (("free cash flow",), "difference", ("operating cash flow",
                                         "capital expenditure")),
    (("working capital",), "difference", ("current assets", "current liabilities")),
    (("earnings per share", "eps"), "ratio", ("net income", "shares outstanding")),
    (("shareholders' equity", "total equity", "owners' funds"), "difference",
     ("total assets", "total liabilities")),
    (("gross profit",), "difference", ("revenue", "cost of sales")),
    (("payout ratio",), "margin", ("dividends paid", "net income")),
    (("revenue growth", "growth rate"), "percentage_change",
     ("revenue this year", "revenue last year")),
    (("enterprise value",), "enterprise_value", ("market capitalisation",
                                                 "total debt", "cash")),
    (("price-to-earnings", "p/e ratio", "pe ratio"), "ratio",
     ("share price", "earnings per share")),
]

# Operand vocabulary: the labels a financial statement may actually use for each
# operand the formulas ask for.
#
# Phase 4 scored calculation at 83.3%, and every one of the 8 failures was the
# same thing: the formula asked for "total debt" while the statement said
# "Borrowings", so the model truthfully reported the label absent and the
# pipeline abstained. Nothing was computed wrongly - the vocabulary was too
# narrow.
#
# PROVENANCE, because it decides what the re-run means. These are statutory and
# conventional captions - the Companies Act balance-sheet formats, UK GAAP and
# IFRS wording - written from standard terminology, not by transcribing the
# labels of the items that failed. "Turnover" for revenue and "Creditors:
# amounts falling due within one year" for current liabilities are the
# statutory captions themselves. Two entries are marked TAILORED below: they are
# phrasings observed in the frozen set that no standard vocabulary would
# contain, and reports/phase4 records them as such, because an entry written
# after seeing a frozen item is selection on the frozen set however it is
# justified.
#
# Longest match wins, so "total current assets" is not resolved by "assets".
SYNONYMS = {
    "revenue": (
        "revenue", "turnover", "sales", "net sales", "total revenue",
        "sales revenue", "total sales", "gross revenue",
        "revenue from contracts with customers", "net revenue",
        "turnover for the year",
        "sales for the year", "sales for the half year", "total turnover",
    ),
    "cost of sales": (
        "cost of sales", "cost of goods sold", "cogs", "cost of revenue",
        "cost of products sold", "direct costs",
    ),
    "net income": (
        "net income", "net profit", "profit for the year", "profit after tax",
        "profit attributable to owners", "profit attributable to shareholders",
        "profit for the period", "net earnings", "earnings",
        "profit attributable to equity holders", "net profit for the year",
    ),
    "operating profit": (
        "operating profit", "trading profit", "operating income",
        "profit from operations", "operating result", "ebit",
        "profit before interest and tax", "operating earnings",
    ),
    "ebit": (
        "ebit", "profit before interest and tax",
        "earnings before interest and tax", "operating profit",
        "trading profit", "profit from operations",
    ),
    "ebitda": (
        "ebitda", "earnings before interest, tax, depreciation and amortisation",
        "earnings before interest tax depreciation and amortisation",
    ),
    "current assets": (
        "current assets", "total current assets",
        "assets falling due within one year",
        "debtors and cash", "stocks and debtors and cash",   # TAILORED
    ),
    "current liabilities": (
        "current liabilities", "total current liabilities",
        "creditors: amounts falling due within one year",
        "creditors due within one year", "amounts falling due within one year",
        "liabilities falling due within one year",
    ),
    "inventory": ("inventory", "inventories", "stock", "stocks",
                  "stock in trade"),
    "total assets": (
        "total assets", "total resources employed",   # TAILORED
        "assets", "total asset value",
    ),
    "total liabilities": ("total liabilities", "liabilities",
                          "total creditors"),
    "shareholders' equity": (
        "shareholders' equity", "shareholders equity", "total equity",
        "owners' funds", "owners funds", "total owners' funds",
        "capital and reserves", "equity", "net assets",
        "total shareholders' funds", "shareholders' funds",
    ),
    "total debt": (
        "total debt", "borrowings", "total borrowings", "loans and borrowings",
        "debt", "interest-bearing liabilities", "loans",
        "total loans and borrowings",
    ),
    "interest expense": (
        "interest expense", "finance charges", "finance costs",
        "interest payable", "interest paid", "interest costs",
        "net finance costs",
    ),
    "operating cash flow": (
        "operating cash flow", "cash flow from operations",
        "net cash inflow from operating activities",
        "net cash from operating activities", "net cash inflow from trading",
        "cash generated from operations",
    ),
    "capital expenditure": (
        "capital expenditure", "capex", "purchase of fixed assets",
        "payments to acquire fixed assets", "payments for fixed assets",
        "additions to property, plant and equipment",
        "purchases of property, plant and equipment",
    ),
    "shares outstanding": (
        "shares outstanding", "number of shares", "ordinary shares in issue",
        "shares in issue", "weighted average shares outstanding",
        "issued share capital (shares)", "number of ordinary shares",
    ),
    "dividends paid": ("dividends paid", "dividends", "distributions to owners",
                       "dividends declared", "equity dividends paid"),
    "share price": ("share price", "price per share", "market price per share",
                    "closing share price"),
    "earnings per share": ("earnings per share", "eps", "basic eps",
                           "basic earnings per share"),
    "market capitalisation": ("market capitalisation", "market capitalization",
                              "market cap", "equity market value"),
    "cash": ("cash", "cash and cash equivalents", "cash at bank",
             "cash and bank balances", "cash balance"),
    "revenue this year": ("revenue this year", "current year revenue",
                          "revenue - current year", "this year's revenue"),
    "revenue last year": ("revenue last year", "prior year revenue",
                          "revenue - prior year", "last year's revenue"),
}

CURRENT_DATA = ("today", "right now", "currently", "this month", "last month",
                "this quarter", "yesterday", "tomorrow", "at the moment",
                "current price", "share price right now")

EXTRACTION_PROMPT = """You are reading a financial extract.

Context:
{context}

Question: {question}

Return ONLY a JSON object with these keys:
{schema}

Rules:
- "value" must be copied exactly as it appears in the context, including any
  currency symbol, comma and decimal places.
- "source_span" must be the complete line or phrase from the context that
  contains the value, copied character for character.
- If the value is not present in the context, set "found" to false and "value"
  to null. Do not guess and do not calculate.

JSON:"""

FIELD_PROMPT = """You are reading a financial extract.

Context:
{context}

Find the value for: {field}

Return ONLY a JSON object with these keys:
{schema}

Copy "value" and "source_span" exactly from the context. If the field is not
present, set "found" to false. Do not calculate anything.

JSON:"""


def numeric(text):
    """A number from a copied value, or None. Rejects malformed separators."""
    if text is None:
        return None
    cleaned = str(text).replace("$", "").replace("£", "").replace("€", "").strip()
    cleaned = cleaned.replace("(", "-").replace(")", "")
    # Tokens are taken whole, bounded by non-alphanumerics, so a malformed
    # separator cannot be salvaged by matching its leading digits: "2,60.65" is
    # rejected rather than read as 2. This matches the scorer's rule, so the
    # pipeline and the scorer never disagree about what a number is.
    for token in re.findall(r"(?<![A-Za-z0-9])-?[\d,]+(?:\.\d+)?(?![A-Za-z0-9])",
                            cleaned):
        token = token.strip().rstrip(",")
        if "," in token:
            if not re.fullmatch(r"-?\d{1,3}(?:,\d{3})+(?:\.\d+)?", token):
                continue
        elif not re.fullmatch(r"-?\d+(?:\.\d+)?", token):
            continue
        try:
            return float(token.replace(",", ""))
        except ValueError:
            continue
    return None


class ToolPipeline:
    def __init__(self, llm, max_new_tokens=160, trace=False):
        self.llm = llm
        self.max_new_tokens = max_new_tokens
        self.trace = trace

    # ----------------------------------------------------- operand vocabulary
    @staticmethod
    def context_labels(context):
        """The label part of each 'Label: value' line in the context.

        Split on the LAST colon when what follows it is a number. Splitting on
        the first colon truncates any caption that contains one - and the
        Companies Act balance-sheet wording "Creditors: amounts falling due
        within one year" does. That caption is in the vocabulary, but the label
        extracted from it was "Creditors", so it never matched and the value was
        unreachable. Falling back to the first colon keeps lines whose tail is
        not a value, such as a bare section heading, behaving as before.
        """
        labels = []
        for line in (context or "").splitlines():
            if ":" not in line:
                continue
            head, tail = line.rsplit(":", 1)
            if numeric(tail) is None:
                head = line.split(":", 1)[0]
            label = head.strip()
            if label:
                labels.append(label)
        return labels

    @staticmethod
    def _match_length(label, synonyms):
        """How well `synonyms` matches `label`, or None.

        Exact caption, or the caption with a trailing qualifier ("Turnover for
        the year" for "turnover"). Deliberately NOT a substring test: "total
        debt" would otherwise claim "Debtors and cash" through "debt", and
        "assets" would claim "Assets falling due within one year". A wrong
        operand is worse than no operand, because abstention is visible and a
        quietly wrong number is not.
        """
        lowered = normalise(label)
        best = None
        for synonym in synonyms:
            synonym = normalise(synonym)
            if lowered == synonym or lowered.startswith(synonym + " "):
                if best is None or len(synonym) > best:
                    best = len(synonym)
        return best

    @classmethod
    def resolve_field(cls, context, field):
        """The label the context actually uses for `field`, or None.

        Python decides this, not the model. The model is then asked for a label
        that is verbatim in front of it, which is the one thing Phase 3 showed
        these models do reliably (copy 100%, extraction 100%). Asking a 1.5B
        model to work out that "Borrowings" means total debt is a semantic leap
        it does not need to make, and one that could not be validated if it made
        it wrongly.

        A label goes to the field that matches it most specifically, compared
        across every field rather than only the one being asked for. Without
        that, "Stocks and debtors and cash" answers to "stocks" as inventory as
        readily as to its full caption as current assets, and which one won
        would depend on call order.
        """
        field = field.lower()
        if field not in SYNONYMS:
            return None
        best_label, best_score = None, None
        for label in cls.context_labels(context):
            score = cls._match_length(label, SYNONYMS[field])
            if score is None:
                continue
            # Does another field claim this same label more specifically?
            contested = max(
                (cls._match_length(label, synonyms) or -1
                 for other, synonyms in SYNONYMS.items() if other != field),
                default=-1)
            if contested > score:
                continue
            if best_score is None or score > best_score:
                best_label, best_score = label, score
        return best_label

    # ---------------------------------------------------------------- model
    def _extract(self, context, question=None, field=None):
        if field is not None:
            prompt = FIELD_PROMPT.format(context=context, field=field,
                                         schema=json.dumps(EXTRACTION_SCHEMA,
                                                           indent=2))
        else:
            prompt = EXTRACTION_PROMPT.format(context=context, question=question,
                                              schema=json.dumps(EXTRACTION_SCHEMA,
                                                                indent=2))
        raw = self.llm(prompt)
        return validate_extraction(parse_json_object(raw), context, raw_output=raw)

    # ------------------------------------------------------------- routing
    @staticmethod
    def _formula_for(question):
        """(operation, operands, matched_trigger) for a question, or Nones.

        The trigger is returned so the caller can ask whether the statement
        already states the thing this formula would compute.
        """
        lowered = question.lower()
        best = None
        for triggers, operation, operands in FORMULAS:
            for trigger in triggers:
                if trigger in lowered:
                    # Longest trigger wins, so "debt-to-ebitda" beats "debt to".
                    if best is None or len(trigger) > best[0]:
                        best = (len(trigger), operation, operands, trigger)
        return (best[1], best[2], best[3]) if best else (None, None, None)

    @classmethod
    def _stated_label_for(cls, context, trigger, question=""):
        """A context label that states the quantity `trigger` would compute.

        A filing that already reports basic EPS should be read, not
        recalculated. Before this, "What is basic EPS?" against a context whose
        first line is "Basic EPS: 1.42" matched the EPS formula, went looking
        for net income and shares outstanding, found no net income and refused -
        with the answer sitting in front of it. Twelve of the twenty
        extraction and wording misses on the frozen set were this.

        The test is deliberately narrow: a label must contain the matched
        trigger as a whole phrase at a word boundary. "Diluted earnings per
        share" states "earnings per share", so it is read. "Revenue" does not
        state "revenue growth", and "Current assets" does not state "current
        ratio", so those are still computed.
        """
        if not trigger:
            return None
        pattern = re.compile(r"(?<![a-z0-9])" + re.escape(normalise(trigger))
                             + r"(?![a-z0-9])")
        asked = normalise(question)
        best, best_score = None, None
        for label in cls.context_labels(context):
            if not pattern.search(normalise(label)):
                continue
            # Several lines can state the same quantity - a statement carries
            # both "Basic EPS" and "Diluted EPS". Preferring the longest label
            # picked Diluted for a question asking Basic. Prefer the line the
            # question actually names, and fall back to length only when the
            # question distinguishes neither.
            named = normalise(label) in asked
            score = (1 if named else 0, len(label))
            if best_score is None or score > best_score:
                best, best_score = label, score
        # If no label is named in the question and more than one states the
        # quantity, there is no basis to choose: compute instead of guessing.
        if best is not None and best_score[0] == 0:
            matches = [l for l in cls.context_labels(context)
                       if pattern.search(normalise(l))]
            if len(matches) > 1:
                return None
        return best

    @staticmethod
    def _is_current_data(question):
        lowered = question.lower()
        return any(phrase in lowered for phrase in CURRENT_DATA)

    # ------------------------------------------------------------- answers
    def answer(self, question, context=""):
        """Returns a PipelineAnswer. Every numeric answer is computed in Python
        and every extracted value carries a validated span."""
        if self._is_current_data(question):
            return PipelineAnswer(
                answer="I cannot verify the current value from the available data: "
                       "no verified live source is connected.",
                component="current_data_rule", abstained=True,
                reason="question asks for a value that changes over time")

        operation, operand_fields, trigger = self._formula_for(question)
        if operation is not None and context:
            # A stated figure beats a derived one. If the statement already
            # reports this quantity, read it; only compute what is absent.
            stated = self._stated_label_for(context, trigger, question)
            if stated is not None:
                result = self._extract(context, field=stated)
                if not result.abstained:
                    return PipelineAnswer(
                        answer=result.value, component="extraction",
                        value=numeric(result.value),
                        source_spans=[result.source_span],
                        detail={**result.to_dict(), "read_not_computed": True,
                                "stated_as": stated})
                # Could not read it after all - fall through and compute.
            return self._calculate(question, context, operation, operand_fields)

        result = self._extract(context, question=question)
        if result.abstained:
            return abstention(result.reason or "extraction could not be validated",
                              component="extraction")
        return PipelineAnswer(
            answer=result.value, component="extraction",
            value=numeric(result.value), source_spans=[result.source_span],
            detail=result.to_dict())

    def _calculate(self, question, context, operation, operand_fields):
        """Operands are extracted and validated; Python does the arithmetic."""
        operands, spans = {}, []
        for field in operand_fields:
            # Ask for the label the statement actually uses, when it uses one.
            resolved = self.resolve_field(context, field)
            result = self._extract(context, field=resolved or field)
            if result.abstained:
                named = (f"'{field}'" if resolved is None
                         else f"'{field}' (as '{resolved}')")
                return abstention(
                    f"cannot compute: {named} is not available in the context "
                    f"({result.reason})", component="calculation")
            value = numeric(result.value)
            if value is None:
                return abstention(
                    f"cannot compute: '{field}' was extracted as "
                    f"{result.value!r}, which is not a number",
                    component="calculation")
            operands[field] = {"value": value, "source_span": result.source_span,
                               "asked_as": resolved or field,
                               "resolved_by_synonym": resolved is not None
                               and normalise(resolved) != normalise(field)}
            spans.append(result.source_span)

        values = [operands[f]["value"] for f in operand_fields]
        try:
            if operation == "quick":
                computed = (values[0] - values[1]) / values[2]
                unit = "x"
            else:
                argument_names, function, unit = OPERATIONS[operation]
                computed = function(*values[:len(argument_names)])
        except ZeroDivisionError:
            return abstention("cannot compute: the denominator is zero",
                              component="calculation")
        except (KeyError, IndexError, TypeError) as error:
            return abstention(f"cannot compute: {type(error).__name__}",
                              component="calculation")

        rendered = (f"{computed:.2f}%" if unit == "%" else
                    f"{computed:.2f}x" if unit == "x" else f"{computed:,.2f}")
        working = ", ".join(f"{f} = {operands[f]['value']:,.2f}"
                            for f in operand_fields)
        return PipelineAnswer(
            answer=f"{rendered} ({working})", component="calculation",
            value=round(computed, 6), operation=operation, operands=operands,
            source_spans=spans,
            detail={"formula": operation, "fields": list(operand_fields),
                    "computed_in": "python"})

    # ------------------------------------------------------ evaluation glue
    def as_answer_fn(self):
        """A plain (question, context) -> str callable for scripts/eval_heldout."""
        def answer_fn(question, context):
            return self.answer(question, context).answer

        return answer_fn
