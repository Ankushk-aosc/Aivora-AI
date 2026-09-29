"""Evaluate any answering system against an externally supplied benchmark.

Step 7 of the brief: the evaluator must not need code changes when a new
benchmark arrives. The benchmarks built in this project are now spent - the
development split is saturated and the hidden split has been run twice - so the
next honest measurement has to come from content this repository did not author.

A benchmark is a JSONL file. Each line needs:

    id              unique string
    category        free text, used only for grouping
    intent          extraction | calculation | concept | interpretation |
                    retrieval | current_data | abstention  (free text is
                    accepted and grouped as-is)
    difficulty      integer 1-5, or null
    question        the prompt given to the system, verbatim
    expected_answer the reference answer (may be null for abstention items)
    grading_type    numeric | exact | rubric | abstention
    source          where the question came from
    provenance      licence or permission under which it may be used

and, depending on grading_type:

    numeric     -> numeric_answer (float), optional tolerance (default 0.01)
    exact       -> expected_answer, optional acceptable (list of alternatives)
    rubric      -> required_any: list of groups of interchangeable wordings
    abstention  -> nothing further; a correct answer declines

Nothing here knows anything about finance, about this project's glossary, or
about which system is being measured. `grade()` dispatches on grading_type only,
so a new benchmark drops in as a file.

    from evaluation.external import load_external, evaluate_external
    items = load_external("benchmarks/vendor_2026.jsonl")
    result = evaluate_external(items, lambda q: my_system(q))
"""

import json
import os

from evaluation.financial_metrics import (
    abstained, normalized_match, numeric_match, rubric_match,
)

REQUIRED_FIELDS = ("id", "category", "intent", "difficulty", "question",
                   "expected_answer", "grading_type", "source", "provenance")

GRADING_TYPES = ("numeric", "exact", "rubric", "abstention")


class BenchmarkFormatError(ValueError):
    """The benchmark file does not meet the documented contract."""


def validate(items):
    """Check the contract before anything is measured.

    A benchmark that is silently malformed produces a number that looks fine and
    means nothing, so this refuses rather than guesses.
    """
    problems, seen = [], set()
    for index, item in enumerate(items):
        where = item.get("id") or f"line {index + 1}"
        missing = [f for f in REQUIRED_FIELDS if f not in item]
        if missing:
            problems.append(f"{where}: missing {missing}")
            continue
        if item["id"] in seen:
            problems.append(f"{where}: duplicate id")
        seen.add(item["id"])
        grading = item["grading_type"]
        if grading not in GRADING_TYPES:
            problems.append(f"{where}: grading_type {grading!r} not in {GRADING_TYPES}")
            continue
        if grading == "numeric" and item.get("numeric_answer") is None:
            problems.append(f"{where}: numeric grading needs numeric_answer")
        if grading == "rubric" and not item.get("required_any"):
            problems.append(f"{where}: rubric grading needs required_any")
        if grading == "rubric":
            groups = item.get("required_any") or []
            if not all(isinstance(g, (list, tuple)) for g in groups):
                problems.append(f"{where}: required_any must be groups of alternatives")
        if grading == "exact" and not item.get("expected_answer"):
            problems.append(f"{where}: exact grading needs expected_answer")
        if not str(item.get("question", "")).strip():
            problems.append(f"{where}: empty question")
        if not item.get("provenance"):
            problems.append(f"{where}: provenance is required - an unlicensed "
                            "benchmark cannot be published with results")
    return problems


def load_external(path, strict=True):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, encoding="utf-8") as handle:
        items = [json.loads(line) for line in handle if line.strip()]
    problems = validate(items)
    if problems and strict:
        raise BenchmarkFormatError(
            f"{len(problems)} problem(s) in {path}:\n  " + "\n  ".join(problems[:20]))
    return items


def grade(item, prediction):
    """Grade one answer. Returns (correct, outcome, graded_by).

    outcome is one of correct_answer, correct_abstention, hallucination,
    over_abstention - the same four the project's other scorers use, so results
    are comparable across benchmarks.
    """
    prediction = prediction or ""
    declined = abstained(prediction)
    grading = item["grading_type"]

    if grading == "abstention":
        return declined, ("correct_abstention" if declined else "hallucination"), grading

    if grading == "numeric":
        correct = numeric_match(prediction, float(item["numeric_answer"]),
                                float(item.get("tolerance", 0.01)))
    elif grading == "exact":
        correct = normalized_match(prediction, item["expected_answer"],
                                   item.get("acceptable"))
    elif grading == "rubric":
        correct = rubric_match(prediction, item["required_any"])
    else:
        raise BenchmarkFormatError(f"unknown grading_type {grading!r}")

    if correct:
        outcome = "correct_answer"
    elif declined:
        outcome = "over_abstention"
    else:
        outcome = "hallucination"
    return correct, outcome, grading


def _bucket(records):
    total = len(records)
    if not total:
        return {"total": 0, "correct": 0, "accuracy": None}
    correct = sum(1 for r in records if r["correct"])
    return {"total": total, "correct": correct,
            "accuracy": round(100.0 * correct / total, 2)}


def evaluate_external(items, answer_fn, label="external benchmark", verbose=False):
    """answer_fn(question) -> answer string."""
    records = []
    for index, item in enumerate(items, 1):
        prediction = answer_fn(item["question"])
        correct, outcome, graded_by = grade(item, prediction)
        records.append({
            "id": item["id"], "category": item["category"],
            "intent": item["intent"], "difficulty": item.get("difficulty"),
            "grading_type": graded_by, "correct": correct, "outcome": outcome,
            "prediction": prediction, "source": item.get("source"),
        })
        if verbose:
            print(f"  {'OK ' if correct else 'XX '}{item['id']}: {prediction[:60]!r}")
        elif index % 100 == 0:
            print(f"  {index}/{len(items)}")

    answerable = [r for r in records if r["grading_type"] != "abstention"]
    abstention = [r for r in records if r["grading_type"] == "abstention"]

    def group(key):
        out = {}
        for record in records:
            out.setdefault(str(record.get(key)), []).append(record)
        return {k: _bucket(v) for k, v in sorted(out.items())}

    hallucinated = sum(1 for r in answerable if r["outcome"] == "hallucination")
    over = sum(1 for r in answerable if r["outcome"] == "over_abstention")
    return {
        "label": label,
        "examples": len(records),
        "summary": {
            "overall": _bucket(records),
            "by_category": group("category"),
            "by_intent": group("intent"),
            "by_difficulty": group("difficulty"),
            "by_grading_type": group("grading_type"),
            "hallucination_rate": round(100.0 * hallucinated / len(answerable), 2)
            if answerable else None,
            "over_abstention_rate": round(100.0 * over / len(answerable), 2)
            if answerable else None,
            "abstention_accuracy": _bucket(abstention)["accuracy"],
            "abstention_items": len(abstention),
        },
        "details": records,
    }
