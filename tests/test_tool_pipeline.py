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
from pipeline.tool_pipeline import ToolPipeline, numeric  # noqa: E402

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


def main():
    test_parsing()
    test_span_rule()
    test_extraction_path()
    test_arithmetic_is_python()
    test_missing_operand_abstains()
    test_current_data_rule()
    test_answerable_not_refused()
    test_numeric_parsing()
    print(f"\n{len(PASSED)}/{len(PASSED) + len(FAILED)} passed")
    sys.exit(1 if FAILED else 0)


if __name__ == "__main__":
    main()
