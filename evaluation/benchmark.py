"""Score any answer function against data/benchmark/{dev,test}.jsonl.

evaluation/evaluator.py scores the 45-question category sets and
evaluation/generic.py lets any model be scored on them. This adds what the
benchmark needs and those do not have:

* abstention accuracy - on items whose correct answer is "insufficient
  information", did the system decline?
* hallucination rate - on answerable items, how often did it assert a wrong
  answer rather than decline? A wrong assertion and a refusal are not the same
  failure and must not be averaged together.
* over-abstention rate - refusing questions that WERE answerable, which is the
  cheap way to fake a low hallucination rate.
* per-level accuracy - the benchmark tags difficulty 1-5, so a single headline
  number cannot hide collapse on multi-step items.
* extraction accuracy - field-level reading of a figure out of a statement.

The hidden split is test.jsonl. Loading it here is for MEASUREMENT only; no
training, tuning, prompt or glossary change may be driven by its contents.

    from evaluation.benchmark import evaluate_benchmark, print_benchmark_report
    results = evaluate_benchmark(lambda q: my_answer(q), split="dev")
    print_benchmark_report(results)
"""

import json
import os

from evaluation.financial_metrics import abstained, numeric_match, rubric_match

BENCHMARK_DIR = os.path.join("data", "benchmark")


def load_benchmark(split: str = "dev", path: str = None):
    """split: "dev" or "test". Returns a list of item dicts."""
    path = path or os.path.join(BENCHMARK_DIR, f"{split}.jsonl")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found - run python scripts/build_benchmark.py")
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def score_item(item: dict, prediction: str) -> dict:
    """Grade one item. Returns correct plus WHY it was correct or not.

    The outcome is one of:
        correct_answer      - answerable, answered right
        correct_abstention  - unanswerable, declined (the right call)
        hallucination       - answerable or not, asserted a wrong answer
        over_abstention     - answerable, but declined
    """
    prediction = prediction or ""
    declined = abstained(prediction)
    must_abstain = bool(item.get("must_abstain"))

    if must_abstain:
        outcome = "correct_abstention" if declined else "hallucination"
        return {"id": item["id"], "category": item["category"],
                "level": item.get("level"), "kind": item.get("kind"),
                "correct": declined, "outcome": outcome,
                "graded_by": "abstention", "prediction": prediction}

    if item.get("numeric_answer") is not None:
        correct = numeric_match(prediction, item["numeric_answer"],
                                item.get("tolerance", 0.01))
        graded_by = "numeric"
    elif item.get("required_any"):
        correct = rubric_match(prediction, item["required_any"])
        graded_by = "rubric"
    else:
        raise ValueError(f"{item['id']} has no gradable target")

    if correct:
        outcome = "correct_answer"
    elif declined:
        outcome = "over_abstention"
    else:
        outcome = "hallucination"

    record = {"id": item["id"], "category": item["category"],
              "level": item.get("level"), "kind": item.get("kind"),
              "correct": correct, "outcome": outcome, "graded_by": graded_by,
              "prediction": prediction}
    if item.get("field"):
        record["field"] = item["field"]
    return record


def _bucket(records):
    total = len(records)
    if not total:
        return {"total": 0, "correct": 0, "accuracy": None}
    correct = sum(1 for r in records if r["correct"])
    return {"total": total, "correct": correct,
            "accuracy": round(100.0 * correct / total, 2)}


def summarise(records: list) -> dict:
    """Aggregate scored records into the benchmark's metric set."""
    answerable = [r for r in records if r["graded_by"] != "abstention"]
    abstention = [r for r in records if r["graded_by"] == "abstention"]

    by_category, by_level, by_kind = {}, {}, {}
    for record in records:
        by_category.setdefault(record["category"], []).append(record)
        by_level.setdefault(str(record["level"]), []).append(record)
        by_kind.setdefault(record["kind"] or "unknown", []).append(record)

    hallucinated = sum(1 for r in answerable if r["outcome"] == "hallucination")
    over_abstained = sum(1 for r in answerable if r["outcome"] == "over_abstention")
    extraction = [r for r in records if r["category"] == "extraction"]

    summary = {
        "overall": _bucket(records),
        "answerable": _bucket(answerable),
        "by_category": {k: _bucket(v) for k, v in sorted(by_category.items())},
        "by_level": {k: _bucket(v) for k, v in sorted(by_level.items())},
        "by_kind": {k: _bucket(v) for k, v in sorted(by_kind.items())},
        # Rate of asserting something wrong, over the questions that HAD an answer.
        "hallucination_rate": round(100.0 * hallucinated / len(answerable), 2)
        if answerable else None,
        "over_abstention_rate": round(100.0 * over_abstained / len(answerable), 2)
        if answerable else None,
        "abstention_accuracy": _bucket(abstention)["accuracy"],
        "abstention_items": len(abstention),
        "extraction_accuracy": _bucket(extraction)["accuracy"],
    }
    return summary


def evaluate_benchmark(generate_fn, split: str = "dev", limit: int = None,
                       categories=None, verbose: bool = False,
                       label: str = None) -> dict:
    """generate_fn(question: str) -> answer: str."""
    items = load_benchmark(split)
    if categories:
        items = [i for i in items if i["category"] in categories]
    if limit:
        items = items[:limit]

    records = []
    for index, item in enumerate(items, 1):
        prediction = generate_fn(item["question"])
        record = score_item(item, prediction)
        records.append(record)
        if verbose:
            mark = "OK " if record["correct"] else "XX "
            print(f"  {mark}{item['id']:<24} {record['outcome']:<18} "
                  f"{prediction[:60]!r}")
        elif index % 100 == 0:
            print(f"  scored {index}/{len(items)}")

    return {"label": label or f"Aivora benchmark ({split})", "split": split,
            "examples": len(records), "summary": summarise(records),
            "details": records}


def print_benchmark_report(results: dict) -> None:
    summary = results["summary"]
    overall = summary["overall"]
    print(f"\n{results['label']}")
    print("=" * len(results["label"]))
    print(f"overall              {overall['correct']}/{overall['total']} "
          f"= {overall['accuracy']}%")
    print(f"hallucination rate   {summary['hallucination_rate']}%  "
          f"(wrong assertion on an answerable question)")
    print(f"over-abstention      {summary['over_abstention_rate']}%  "
          f"(declined an answerable question)")
    print(f"abstention accuracy  {summary['abstention_accuracy']}%  "
          f"over {summary['abstention_items']} unanswerable items")
    print(f"extraction accuracy  {summary['extraction_accuracy']}%")

    print("\nby category")
    for name, bucket in summary["by_category"].items():
        print(f"  {name:<20} {bucket['correct']:>4}/{bucket['total']:<4} "
              f"{bucket['accuracy']}%")
    print("by difficulty level")
    for name, bucket in summary["by_level"].items():
        print(f"  level {name:<14} {bucket['correct']:>4}/{bucket['total']:<4} "
              f"{bucket['accuracy']}%")
    print("by item kind")
    for name, bucket in summary["by_kind"].items():
        print(f"  {name:<20} {bucket['correct']:>4}/{bucket['total']:<4} "
              f"{bucket['accuracy']}%")
