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
        lowered = question.lower()
        best = None
        for triggers, operation, operands in FORMULAS:
            for trigger in triggers:
                if trigger in lowered:
                    # Longest trigger wins, so "debt-to-ebitda" beats "debt to".
                    if best is None or len(trigger) > best[0]:
                        best = (len(trigger), operation, operands)
        return (best[1], best[2]) if best else (None, None)

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

        operation, operand_fields = self._formula_for(question)
        if operation is not None and context:
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
            result = self._extract(context, field=field)
            if result.abstained:
                return abstention(
                    f"cannot compute: '{field}' is not available in the context "
                    f"({result.reason})", component="calculation")
            value = numeric(result.value)
            if value is None:
                return abstention(
                    f"cannot compute: '{field}' was extracted as "
                    f"{result.value!r}, which is not a number",
                    component="calculation")
            operands[field] = {"value": value, "source_span": result.source_span}
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
