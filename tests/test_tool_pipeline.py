"""The tool pipeline's guarantees, proven with a stub model.

The point of the pipeline is that correctness does not depend on the model being
good. These tests use a scripted stub, so they check the GUARANTEES:

  * a value whose span is absent from the context never reaches the user
  * a value that is not inside its own span never reaches the user
  * unparseable or dishonest model output becomes an abstention
  * arithmetic is done in Python - a model that returns a wrong total cannot
    make the pipeline wrong
  * a missing operand abstains instead of guessing
  * answerable questions are not refused

Run: deepseek_env/Scripts/python.exe tests/test_tool_pipeline.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.schema import parse_json_object, validate_extraction  # noqa: E402
from pipeline.tool_pipeline import SYNONYMS, ToolPipeline, numeric  # noqa: E402

PASSED, FAILED = [], []

CONTEXT = ("Financial summary:\n"
           "Revenue: 12,000.00\n"
           "Cost of sales: 7,400.00\n"
           "Cash balance: 1,204.50\n"
           "Total debt: 3,735.00")


def check(name, ok, detail=""):
    (PASSED if ok else FAILED).append(name)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" - {detail}" if detail else ""))


def stub(responses):
    """A model that returns scripted responses in order, then repeats the last."""
    state = {"i": 0}

    def llm(prompt):
        index = min(state["i"], len(responses) - 1)
        state["i"] += 1
        response = responses[index]
        return response(prompt) if callable(response) else response

    return llm


def test_parsing():
    check("parses plain JSON",
          parse_json_object('{"found": true, "value": "5"}') == {"found": True,
                                                                 "value": "5"})
    check("parses fenced JSON",
          parse_json_object('```json\n{"found": true}\n```') == {"found": True})
    check("parses JSON after prose",
          parse_json_object('Here it is: {"found": false}') == {"found": False})
    check("tolerates a trailing comma",
          parse_json_object('{"found": true, "value": "5",}') is not None)
    check("returns None on nonsense", parse_json_object("no json here") is None)
    check("returns None on empty", parse_json_object("") is None)


def test_span_rule():
    """The binding rule: no value without a span that is really in the context."""
    good = {"field": "cash balance", "value": "1,204.50",
            "source_span": "Cash balance: 1,204.50", "found": True}
    result = validate_extraction(good, CONTEXT)
    check("valid extraction passes", not result.abstained and result.span_validated,
          result.reason or "")

    invented_span = {**good, "source_span": "Cash balance: 9,999.00"}
    result = validate_extraction(invented_span, CONTEXT)
    check("invented span is rejected", result.abstained
          and "does not appear in the context" in (result.reason or ""),
          result.reason or "")

    value_not_in_span = {**good, "value": "9,999.00"}
    result = validate_extraction(value_not_in_span, CONTEXT)
    check("value outside its own span is rejected", result.abstained,
          result.reason or "")

    no_span = {"field": "cash", "value": "1,204.50", "found": True}
    check("missing span is rejected",
          validate_extraction(no_span, CONTEXT).abstained)

    check("unparseable output abstains",
          validate_extraction(None, CONTEXT).abstained)

    check("found=false abstains",
          validate_extraction({"found": False, "value": None}, CONTEXT).abstained)

    # Whitespace and punctuation differences must NOT fail a correct span.
    reflowed = {**good, "source_span": "Cash   balance:  1,204.50"}
    check("whitespace differences still validate",
          not validate_extraction(reflowed, CONTEXT).abstained)


def test_extraction_path():
    pipe = ToolPipeline(stub([json.dumps(
        {"field": "cash balance", "value": "1,204.50",
         "source_span": "Cash balance: 1,204.50", "found": True})]))
    answer = pipe.answer("What is the cash balance?", CONTEXT)
    check("extraction returns the value", answer.answer == "1,204.50", answer.answer)
    check("extraction reports its span",
          answer.source_spans == ["Cash balance: 1,204.50"], str(answer.source_spans))
    check("extraction is not an abstention", not answer.abstained)

    # A model that hallucinates a value with a fabricated span.
    pipe = ToolPipeline(stub([json.dumps(
        {"field": "cash balance", "value": "8,888.00",
         "source_span": "Cash balance: 8,888.00", "found": True})]))
    answer = pipe.answer("What is the cash balance?", CONTEXT)
    check("hallucinated value is blocked by the span rule",
          answer.abstained and "Insufficient information" in answer.answer,
          answer.answer[:60])


def test_arithmetic_is_python():
    """The model supplies operands; Python computes. A model that returns a
    wrong total cannot make the pipeline wrong, because it is never asked."""
    responses = [
        json.dumps({"field": "revenue", "value": "12,000.00",
                    "source_span": "Revenue: 12,000.00", "found": True}),
        json.dumps({"field": "cost of sales", "value": "7,400.00",
                    "source_span": "Cost of sales: 7,400.00", "found": True}),
    ]
    pipe = ToolPipeline(stub(responses))
    answer = pipe.answer("What is the gross margin?", CONTEXT)
    # (12000 - 7400) / 12000 * 100 = 38.333...
    check("gross margin computed in Python",
          abs(answer.value - 38.3333) < 0.01, str(answer.value))
    check("calculation cites both spans", len(answer.source_spans) == 2,
          str(answer.source_spans))
    check("calculation records it computed in python",
          answer.detail.get("computed_in") == "python")

    # The model is never asked for the answer, so even if it tried to give one
    # the pipeline ignores it: here the stub returns a nonsense "value" for the
    # operand, and the pipeline must reject rather than use it.
    bad = [json.dumps({"field": "revenue", "value": "lots",
                       "source_span": "Revenue: 12,000.00", "found": True})]
    pipe = ToolPipeline(stub(bad))
    answer = pipe.answer("What is the gross margin?", CONTEXT)
    check("non-numeric operand abstains", answer.abstained, answer.answer[:70])


def test_missing_operand_abstains():
    responses = [
        json.dumps({"field": "revenue", "value": "12,000.00",
                    "source_span": "Revenue: 12,000.00", "found": True}),
        json.dumps({"field": "shares outstanding", "value": None, "found": False}),
    ]
    pipe = ToolPipeline(stub(responses))
    answer = pipe.answer("What is earnings per share?", CONTEXT)
    check("missing operand abstains instead of guessing", answer.abstained,
          answer.answer[:70])
    check("abstention names the missing field",
          "shares outstanding" in (answer.reason or ""), answer.reason or "")


def test_current_data_rule():
    pipe = ToolPipeline(stub(["{}"]))
    for question in ("What is the share price right now?",
                     "What is the inflation rate today?"):
        answer = pipe.answer(question, "")
        check(f"current data refused: {question[:34]}",
              answer.abstained and answer.component == "current_data_rule",
              answer.answer[:50])


def test_answerable_not_refused():
    """Over-refusal is a failure too: a present value must come back."""
    pipe = ToolPipeline(stub([json.dumps(
        {"field": "total debt", "value": "3,735.00",
         "source_span": "Total debt: 3,735.00", "found": True})]))
    answer = pipe.answer("What is total debt?", CONTEXT)
    check("answerable question is answered, not refused",
          not answer.abstained and "3,735.00" in answer.answer, answer.answer)


def test_numeric_parsing():
    cases = [("1,204.50", 1204.5), ("$3,735.00", 3735.0), ("(2,110.00)", -2110.0),
             ("38.33%", 38.33), ("lots", None), ("2,60.65", None), ("", None)]
    for text, expected in cases:
        got = numeric(text)
        ok = (got is None and expected is None) or (
            got is not None and expected is not None and abs(got - expected) < 0.01)
        check(f"numeric({text!r}) -> {expected}", ok, f"got {got}")



def test_operand_vocabulary():
    """A formula's operand is found under the caption the statement uses.

    Phase 4 lost 8 calculation items to this: the formula asked for "total
    debt" and the statement said "Borrowings". The danger in fixing it is the
    opposite error - a synonym claiming a label that is not its own - because a
    quietly wrong operand is worse than an abstention.
    """
    resolve = ToolPipeline.resolve_field

    # Statutory and conventional captions resolve.
    for context, field, expected in [
        ("Turnover for the year: 1,000.00", "revenue", "Turnover for the year"),
        ("Borrowings: 3,528.00", "total debt", "Borrowings"),
        ("Owners' funds: 8,820.00", "shareholders' equity", "Owners' funds"),
        ("Finance charges: 297.00", "interest expense", "Finance charges"),
        ("Profit before interest and tax: 2,376.00", "ebit",
         "Profit before interest and tax"),
        ("Trading profit: 400.00", "operating profit", "Trading profit"),
        ("Creditors due within one year: 50.00", "current liabilities",
         "Creditors due within one year"),
        ("Cash and cash equivalents: 42.00", "cash", "Cash and cash equivalents"),
        ("Stocks: 120.00", "inventory", "Stocks"),
    ]:
        check(f"caption resolves: {field} <- {expected!r}",
              resolve(context, field) == expected, repr(resolve(context, field)))

    # A synonym must NOT claim a label belonging to something else.
    for context, field, why in [
        ("Debtors and cash: 500.00", "total debt", "'debt' vs 'Debtors'"),
        ("Assets falling due within one year: 100.00", "total assets",
         "'assets' vs a current-asset caption"),
        ("Liabilities falling due within one year: 50.00", "total liabilities",
         "'liabilities' vs a current-liability caption"),
        ("Stocks and debtors and cash: 900.00", "inventory",
         "composite caption belongs to current assets"),
        ("Stocks and debtors and cash: 900.00", "cash",
         "caption does not start with cash"),
        ("Total current assets: 10.00", "total assets", "total vs current"),
        ("Headcount: 88", "revenue", "unrelated label"),
    ]:
        check(f"no false claim: {field} ({why})",
              resolve(context, field) is None, repr(resolve(context, field)))

    check("an unknown field resolves to nothing",
          resolve("Revenue: 10.00", "flux capacitance") is None)
    check("no context resolves to nothing", resolve("", "revenue") is None)
    check("the vocabulary covers every operand the formulas ask for",
          all(f in SYNONYMS for _, _, fields in
              __import__("pipeline.tool_pipeline", fromlist=["FORMULAS"]).FORMULAS
              for f in fields),
          "a formula asks for an operand with no vocabulary entry")


def test_synonym_cannot_invent_an_answer():
    """Resolution only renames what to look for; the span rule still decides.

    A model that answers with a value absent from the context must still be
    blocked, even when the operand label resolved perfectly.
    """
    context = "Borrowings: 3,528.00\nOwners' funds: 8,820.00"
    responses = [
        json.dumps({"field": "Borrowings", "value": "9,999.00",
                    "source_span": "Borrowings: 9,999.00", "found": True}),
    ]
    pipe = ToolPipeline(stub(responses))
    answer = pipe.answer("What is the gearing ratio?", context)
    check("a fabricated operand span still abstains", answer.abstained,
          answer.answer[:70])

    # And the honest path computes in Python from the resolved captions.
    good = [
        json.dumps({"field": "Borrowings", "value": "3,528.00",
                    "source_span": "Borrowings: 3,528.00", "found": True}),
        json.dumps({"field": "Owners' funds", "value": "8,820.00",
                    "source_span": "Owners' funds: 8,820.00", "found": True}),
    ]
    pipe = ToolPipeline(stub(good))
    answer = pipe.answer("What is the gearing ratio?", context)
    check("gearing computed from resolved captions",
          answer.value is not None and abs(answer.value - 0.4) < 0.01,
          str(answer.value))
    check("the resolution is recorded for diagnosis",
          any(o.get("resolved_by_synonym") for o in answer.operands.values()),
          str(answer.operands))


def main():
    test_parsing()
    test_span_rule()
    test_extraction_path()
    test_arithmetic_is_python()
    test_missing_operand_abstains()
    test_current_data_rule()
    test_answerable_not_refused()
    test_numeric_parsing()
    test_operand_vocabulary()
    test_synonym_cannot_invent_an_answer()
    print(f"\n{len(PASSED)}/{len(PASSED) + len(FAILED)} passed")
    sys.exit(1 if FAILED else 0)


if __name__ == "__main__":
    main()
