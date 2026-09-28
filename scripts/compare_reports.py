"""Before/after comparison of two baseline_report.json runs.

Both sides are re-graded with the CURRENT scorer from their stored predictions,
which matters: the abstention marker list grew while these fixes were made (the
current-data handler began refusing with "Insufficient current data available"),
and grading the old run with the old list and the new run with the new one would
credit the fixes with a scoring change. Re-grading both removes that.

usage:
    python scripts/compare_reports.py reports/baseline_report.json reports/after/baseline_report.json
"""

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from evaluation.benchmark import load_benchmark, score_item, summarise  # noqa: E402

# The failure groups reports/baseline_analysis.md named, so the claim that they
# were fixed is checked against the items themselves rather than the headline.
TRACKED_GROUPS = {
    "extraction (77 failures)":
        lambda item: item["category"] == "extraction",
    "statement derivations (66 failures)":
        lambda item: item["category"] in ("statements", "accounting")
        and item.get("numeric_answer") is not None,
    "dividend payout / WACC / EPS / EV parser gaps":
        lambda item: item["category"] in ("corporate_finance", "valuation")
        and item.get("numeric_answer") is not None,
    "quick ratio / asset turnover / interest coverage":
        lambda item: item.get("formula") in ("quick_ratio", "asset_turnover",
                                             "interest_coverage"),
    "current-data and missing-figure abstention":
        lambda item: bool(item.get("must_abstain")),
}


def regrade(report, items):
    """Re-score every configuration's stored predictions with today's scorer."""
    out = {}
    for name, entry in report["configurations"].items():
        records = []
        for record in entry["details"]:
            item = items.get(record["id"])
            if item is None:
                continue
            fresh = score_item(item, record["prediction"])
            records.append(fresh)
        out[name] = {"summary": summarise(records), "records": records,
                     "seconds_per_question": entry.get("seconds_per_question"),
                     "description": entry.get("description", "")}
    return out


def group_accuracy(records, items, predicate):
    subset = [r for r in records if predicate(items[r["id"]])]
    if not subset:
        return None
    correct = sum(1 for r in subset if r["correct"])
    return {"total": len(subset), "correct": correct,
            "accuracy": round(100.0 * correct / len(subset), 2)}


def delta(before, after):
    if before is None or after is None:
        return "-"
    change = after - before
    relative = f" ({change / before * 100:+.0f}% rel)" if before else ""
    return f"{change:+.2f}{relative}"


def markdown(before, after, items, before_path, after_path):
    lines, add = [], None
    add = lines.append
    add("# Before / after: deterministic pipeline fixes")
    add("")
    add(f"Both runs are the 795-item dev split, re-graded with the same current "
        f"scorer from their stored predictions.")
    add("")
    add(f"* before: `{before_path}`")
    add(f"* after:  `{after_path}`")
    add("")
    add("No model weights were changed. The checkpoint is the same file in both runs.")
    add("")

    configs = [n for n in ("model_only", "plus_calculator", "plus_retrieval",
                           "full_pipeline") if n in before and n in after]

    add("## Headline accuracy")
    add("")
    add("| configuration | before | after | absolute | relative |")
    add("| --- | --- | --- | --- | --- |")
    for name in configs:
        b = before[name]["summary"]["overall"]
        a = after[name]["summary"]["overall"]
        rel = f"{(a['accuracy'] - b['accuracy']) / b['accuracy'] * 100:+.0f}%" \
            if b["accuracy"] else "-"
        add(f"| {name} | {b['accuracy']}% ({b['correct']}/{b['total']}) | "
            f"**{a['accuracy']}%** ({a['correct']}/{a['total']}) | "
            f"{a['accuracy'] - b['accuracy']:+.2f} pts | {rel} |")
    add("")

    metrics = [
        ("overall accuracy", lambda s: s["overall"]["accuracy"], "higher"),
        ("extraction accuracy", lambda s: s["extraction_accuracy"], "higher"),
        ("abstention accuracy", lambda s: s["abstention_accuracy"], "higher"),
        ("over-abstention rate", lambda s: s["over_abstention_rate"], "lower"),
        ("hallucination rate", lambda s: s["hallucination_rate"], "lower"),
    ]
    add("## The served pipeline (full_pipeline), metric by metric")
    add("")
    add("| metric | before | after | change | wanted |")
    add("| --- | --- | --- | --- | --- |")
    b_summary = before["full_pipeline"]["summary"]
    a_summary = after["full_pipeline"]["summary"]
    for label, getter, direction in metrics:
        b_value, a_value = getter(b_summary), getter(a_summary)
        add(f"| {label} | {b_value}% | **{a_value}%** | {delta(b_value, a_value)} | "
            f"{direction} |")
    add(f"| seconds per question | {before['full_pipeline']['seconds_per_question']} | "
        f"{after['full_pipeline']['seconds_per_question']} | | lower |")
    add("")

    add("## The failure groups the baseline named")
    add("")
    add("| group | before | after | change |")
    add("| --- | --- | --- | --- |")
    for label, predicate in TRACKED_GROUPS.items():
        b_group = group_accuracy(before["full_pipeline"]["records"], items, predicate)
        a_group = group_accuracy(after["full_pipeline"]["records"], items, predicate)
        if not b_group:
            continue
        add(f"| {label} | {b_group['accuracy']}% ({b_group['correct']}/{b_group['total']}) "
            f"| **{a_group['accuracy']}%** ({a_group['correct']}/{a_group['total']}) | "
            f"{delta(b_group['accuracy'], a_group['accuracy'])} |")
    add("")

    add("## Per category (full_pipeline)")
    add("")
    add("| category | before | after | change |")
    add("| --- | --- | --- | --- |")
    for category in sorted(a_summary["by_category"]):
        b_bucket = b_summary["by_category"].get(category)
        a_bucket = a_summary["by_category"][category]
        add(f"| {category} | "
            f"{b_bucket['accuracy'] if b_bucket else '-'}% | "
            f"**{a_bucket['accuracy']}%** | "
            f"{delta(b_bucket['accuracy'] if b_bucket else None, a_bucket['accuracy'])} |")
    add("")

    add("## Per difficulty level (full_pipeline)")
    add("")
    add("| level | before | after | change |")
    add("| --- | --- | --- | --- |")
    for level in sorted(a_summary["by_level"]):
        b_bucket = b_summary["by_level"].get(level)
        a_bucket = a_summary["by_level"][level]
        add(f"| {level} | {b_bucket['accuracy'] if b_bucket else '-'}% | "
            f"**{a_bucket['accuracy']}%** | "
            f"{delta(b_bucket['accuracy'] if b_bucket else None, a_bucket['accuracy'])} |")
    add("")

    add("## Regressions")
    add("")
    before_correct = {r["id"] for r in before["full_pipeline"]["records"] if r["correct"]}
    after_records = {r["id"]: r for r in after["full_pipeline"]["records"]}
    regressed = [id_ for id_ in sorted(before_correct)
                 if id_ in after_records and not after_records[id_]["correct"]]
    if not regressed:
        add("None: every item the served pipeline answered correctly before is still correct.")
    else:
        add(f"{len(regressed)} items were correct before and are not now:")
        add("")
        for id_ in regressed[:15]:
            record = after_records[id_]
            add(f"* `{id_}` ({record['category']}, {record['outcome']}): "
                f"{record['prediction'][:120].strip()!r}")
    add("")

    add("## Remaining failures (full_pipeline, after)")
    add("")
    remaining = {}
    for record in after["full_pipeline"]["records"]:
        if record["correct"]:
            continue
        remaining.setdefault((record["category"], record["outcome"]), []).append(record)
    for (category, outcome), records in sorted(remaining.items(),
                                               key=lambda kv: -len(kv[1])):
        add(f"* **{len(records)}** {category} / {outcome} - e.g. `{records[0]['id']}`: "
            f"{records[0]['prediction'][:110].strip()!r}")
    add("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("before")
    parser.add_argument("after")
    parser.add_argument("--split", default="dev")
    parser.add_argument("--out", default=os.path.join("reports", "improvement_report.md"))
    args = parser.parse_args()

    items = {i["id"]: i for i in load_benchmark(args.split)}
    before = regrade(json.load(open(args.before, encoding="utf-8")), items)
    after = regrade(json.load(open(args.after, encoding="utf-8")), items)

    text = markdown(before, after, items, args.before, args.after)
    with open(args.out, "w", encoding="utf-8") as handle:
        handle.write(text)
    print(text)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
