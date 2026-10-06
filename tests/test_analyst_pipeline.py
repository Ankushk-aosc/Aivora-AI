"""The Analyst served by the Phase 4 tool pipeline.

Two things are tested, and the second is the one that matters on hardware that
cannot host the model:

1. The guarantees, driven by an injected model so no download or 7 GB of memory
   is needed. A value reaches the user only with a validated span, arithmetic is
   Python's, and a model that lies is blocked.
2. The readiness gate. The pipeline must refuse to serve when it cannot serve
   well, because an Analyst that abstains on every question is a regression, and
   silently becoming one would be worse than not switching at all. That is not
   hypothetical: measured on the from-scratch checkpoint, the pipeline abstained
   on 3 of 3 questions, and the model had invented $5.2M for a context stating
   $10,000,000.

Run: deepseek_env/Scripts/python.exe tests/test_analyst_pipeline.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.backend.services import analyst_pipeline  # noqa: E402

PASSED, FAILED = [], []

CONTEXT = ("Revenue: $10,000,000\n"
           "Cost of Goods Sold: $7,000,000\n"
           "Operating Expenses: $2,100,000\n"
           "Net Profit: $500,000")


def check(name, ok, detail=""):
    (PASSED if ok else FAILED).append(name)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" - {detail}" if detail else ""))


def competent_llm(prompt):
    """A model that follows the extraction format, as Qwen2.5-1.5B does."""
    field = ""
    if "Find the value for:" in prompt:
        field = prompt.split("Find the value for:")[1].split("\n")[0].strip().lower()
    elif "Question:" in prompt:
        question = prompt.split("Question:")[1].split("\n")[0].lower()
        for candidate in ("net profit", "revenue", "operating expenses"):
            if candidate in question:
                field = candidate
                break
    # The pipeline resolves an operand to the caption the document actually
    # uses, so it asks for "Cost of Goods Sold", not "cost of sales". A stub
    # that only knew the formula's own wording would fail here for the wrong
    # reason - match the captions, as a real extractor reading the page would.
    labels = {
        "revenue": ("Revenue", "$10,000,000"),
        "cost of goods sold": ("Cost of Goods Sold", "$7,000,000"),
        "cost of sales": ("Cost of Goods Sold", "$7,000,000"),
        "operating expenses": ("Operating Expenses", "$2,100,000"),
        "net profit": ("Net Profit", "$500,000"),
        "net income": ("Net Profit", "$500,000"),
    }
    # Longest caption first, so "cost of goods sold" is not claimed by a
    # shorter key that happens to be a substring of the request.
    for key in sorted(labels, key=len, reverse=True):
        if key in field:
            label, value = labels[key]
            return json.dumps({"field": label, "value": value,
                               "source_span": f"{label}: {value}", "found": True})
    return json.dumps({"found": False, "value": None})


def lying_llm(_prompt):
    """A model that reports a value absent from the context, with a fabricated
    span - the failure the span rule exists to stop."""
    return json.dumps({"field": "Revenue", "value": "$5,200,000",
                       "source_span": "Revenue: $5,200,000", "found": True})


def unparseable_llm(_prompt):
    """What the from-scratch model actually produced, measured 2026-10-05."""
    return "Revenue = $5,200,000 - the EBIT entry."


def test_enabled_by_default():
    """The guarded path is the default. A default install must not answer
    financial questions through the unguarded one."""
    analyst_pipeline.reset()
    os.environ.pop("AIVORA_ANALYST_PIPELINE", None)
    state = analyst_pipeline.readiness()
    check("the pipeline is on unless switched off", state["enabled"] is True)


def test_refuses_rather_than_falling_back():
    """Enabled but unable to serve must REFUSE, never hand the question to the
    unguarded engine. Silent fallback would mean a user receives a figure with
    none of the product's guarantees and is not told."""
    analyst_pipeline.reset()
    os.environ.pop("AIVORA_ANALYST_PIPELINE", None)      # default: on
    original = analyst_pipeline.free_memory_gb
    try:
        analyst_pipeline.free_memory_gb = lambda: 0.5    # cannot host the model
        result = analyst_pipeline.answer("What is revenue?", CONTEXT)
        check("it does not yield to the legacy engine", result is not None)
        check("it refuses", result["abstained"] is True)
        check("the refusal is marked unavailable", result.get("unavailable") is True)
        check("no value is returned", result["value"] is None)
        check("no arithmetic is claimed", result["arithmetic"] == "not performed")
        check("it explains why", bool(result["reason"]))
        check("it states the remedy", "AIVORA_ANALYST_PIPELINE=0" in result["remedy"])
    finally:
        analyst_pipeline.free_memory_gb = original
        analyst_pipeline.reset()


def test_explicit_opt_out_yields_to_the_legacy_engine():
    """Opting out is still allowed - but it must be explicit."""
    analyst_pipeline.reset()
    os.environ["AIVORA_ANALYST_PIPELINE"] = "0"
    try:
        state = analyst_pipeline.readiness()
        check("opting out switches it off", state["enabled"] is False)
        check("the Analyst keeps its existing engine",
              state["engine"] == "existing analyst")
        check("answer() yields to the existing engine",
              analyst_pipeline.answer("What is revenue?", CONTEXT) is None)
    finally:
        os.environ.pop("AIVORA_ANALYST_PIPELINE", None)
        analyst_pipeline.reset()


def test_refuses_when_the_model_cannot_be_hosted():
    """The gate that stops a silent regression."""
    analyst_pipeline.reset()
    os.environ["AIVORA_ANALYST_PIPELINE"] = "1"
    os.environ["AIVORA_ANALYST_MODEL"] = "Qwen/Qwen2.5-1.5B-Instruct"
    original = analyst_pipeline.free_memory_gb
    try:
        analyst_pipeline.free_memory_gb = lambda: 1.2
        state = analyst_pipeline.readiness()
        check("it refuses when memory is short", state["serving"] is False)
        check("it names the model and the shortfall",
              "7.0 GB" in state["reason"] and "1.2 GB" in state["reason"],
              state["reason"])
        check("it explains why refusing beats degrading",
              "refuses rather than answering" in state["reason"], state["reason"])
        check("the status does not claim the legacy engine is serving",
              state["engine"] == "none - refusing", state["engine"])
        check("it reports what it needs",
              state["needed_memory_gb"] == 7.0 and state["free_memory_gb"] == 1.2)

        analyst_pipeline.free_memory_gb = lambda: 12.0
        ready = analyst_pipeline.readiness()
        check("it serves when memory allows", ready["serving"] is True, str(ready))
        check("it names the engine that will serve",
              "tool pipeline" in ready["engine"] and "Qwen" in ready["engine"],
              ready["engine"])
    finally:
        analyst_pipeline.free_memory_gb = original
        os.environ.pop("AIVORA_ANALYST_PIPELINE", None)
        os.environ.pop("AIVORA_ANALYST_MODEL", None)
        analyst_pipeline.reset()


def test_extraction_carries_a_validated_span():
    analyst_pipeline.reset()
    result = analyst_pipeline.answer("What is revenue?", CONTEXT, llm=competent_llm)
    check("an answer is produced", result is not None)
    check("the value is the one in the document",
          "$10,000,000" in str(result["answer"]), str(result["answer"]))
    check("it is not an abstention", result["abstained"] is False)
    check("it cites the span it came from",
          result["source_spans"] == ["Revenue: $10,000,000"],
          str(result["source_spans"]))
    check("it reports arithmetic is Python's", result["arithmetic"] == "python")
    check("it states the guarantee", "validated against the document"
          in result["guarantee"])


def test_arithmetic_is_pythons():
    analyst_pipeline.reset()
    result = analyst_pipeline.answer("What is the gross margin?", CONTEXT,
                                     llm=competent_llm)
    # (10,000,000 - 7,000,000) / 10,000,000 x 100 = 30%
    check("the margin is computed, not generated",
          result["value"] is not None and abs(result["value"] - 30.0) < 0.01,
          str(result["value"]))
    check("the calculation cites both operand spans",
          len(result["source_spans"]) == 2, str(result["source_spans"]))
    check("it is attributed to the calculation component",
          result["component"] == "calculation", result["component"])


def test_a_lying_model_is_blocked():
    """The whole reason for the switch: the user never sees the invented value."""
    analyst_pipeline.reset()
    result = analyst_pipeline.answer("What is revenue?", CONTEXT, llm=lying_llm)
    check("a fabricated value is blocked", result["abstained"] is True)
    check("the invented figure never reaches the user",
          "5,200,000" not in str(result["answer"]), str(result["answer"]))
    check("the refusal is explained", bool(result["reason"]))


def test_the_measured_from_scratch_failure():
    """Exactly what the from-scratch model emitted, replayed as a regression."""
    analyst_pipeline.reset()
    result = analyst_pipeline.answer("What is revenue?", CONTEXT,
                                     llm=unparseable_llm)
    check("unparseable output becomes an abstention", result["abstained"] is True)
    check("its invented $5.2M is not shown",
          "5,200,000" not in str(result["answer"]), str(result["answer"]))
    check("the reason names the parse failure",
          "parseable" in str(result["reason"]).lower(), str(result["reason"]))


def test_an_abstention_is_an_answer_not_a_fallthrough():
    """An abstention must be returned, not treated as 'engine not serving'.
    Returning None would send the question to a path that may guess, discarding
    the guarantee that justified switching."""
    analyst_pipeline.reset()
    result = analyst_pipeline.answer("What is the headcount?", CONTEXT,
                                     llm=competent_llm)
    check("an abstention is a real answer, not None", result is not None)
    check("and it is marked as an abstention", result["abstained"] is True)


def main():
    test_enabled_by_default()
    test_refuses_rather_than_falling_back()
    test_explicit_opt_out_yields_to_the_legacy_engine()
    test_refuses_when_the_model_cannot_be_hosted()
    test_extraction_carries_a_validated_span()
    test_arithmetic_is_pythons()
    test_a_lying_model_is_blocked()
    test_the_measured_from_scratch_failure()
    test_an_abstention_is_an_answer_not_a_fallthrough()
    analyst_pipeline.reset()
    print(f"\n{len(PASSED)}/{len(PASSED) + len(FAILED)} passed")
    sys.exit(1 if FAILED else 0)


if __name__ == "__main__":
    main()
