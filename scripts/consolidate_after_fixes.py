"""Merge the per-split benchmark runs into reports/system_after_fixes.{json,md}.

scripts/run_benchmarks.py writes one file per split. The brief asks for a single
consolidated pair, so this collects whichever split runs exist and produces the
combined view, including the metric table Part 20 lists and the component
invocation shares.

usage:
    python scripts/consolidate_after_fixes.py
"""

import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

REPORT_DIR = "reports"

SPLIT_NOTES = {
    "dev": ("development", "iterated against while building the pipeline: it "
                           "describes this work rather than measuring it"),
    "test": ("hidden", "measurement only, spent once; its prose items were "
                       "authored by Claude earlier in this session"),
    "blind": ("authored prose", "144 authored questions, 13.7% of the 1,000 the "
                                "brief asked for; a fair test of the code, an "
                                "upper bound for the glossary"),
}

METRICS = [
    ("overall accuracy", lambda s: s["overall"]["accuracy"]),
    ("extraction accuracy", lambda s: s["extraction_accuracy"]),
    ("abstention accuracy", lambda s: s["abstention_accuracy"]),
    ("hallucination rate", lambda s: s["hallucination_rate"]),
    ("over-abstention rate", lambda s: s["over_abstention_rate"]),
]


def load_runs():
    runs = {}
    for path in sorted(glob.glob(os.path.join(REPORT_DIR, "system_after_fixes_*.json"))):
        with open(path, encoding="utf-8") as handle:
            report = json.load(handle)
        runs[report["split"]] = report
    return runs


def category_accuracy(summary, names):
    """Accuracy over a group of categories, for the brief's metric names."""
    total = correct = 0
    for name in names:
        bucket = summary["by_category"].get(name)
        if bucket:
            total += bucket["total"]
            correct += bucket["correct"]
    return round(100.0 * correct / total, 2) if total else None


def markdown(runs):
    lines, add = [], None
    add = lines.append
    add("# System state after the pre-training fixes")
    add("")
    add("Produced by `scripts/consolidate_after_fixes.py` from the per-split runs "
        "of `scripts/run_benchmarks.py`. No model weights were modified anywhere "
        "in this work.")
    add("")

    add("## Splits")
    add("")
    add("| split | items | what it measures |")
    add("| --- | --- | --- |")
    for split, report in runs.items():
        label, note = SPLIT_NOTES.get(split, (split, ""))
        add(f"| `{split}` ({label}) | {report['examples']} | {note} |")
    add("")

    add("## SYSTEM vs MODEL")
    add("")
    add("| split | benchmark | accuracy | hallucination | over-abstention | abstention acc. |")
    add("| --- | --- | --- | --- | --- | --- |")
    for split, report in runs.items():
        for name, entry in report["benchmarks"].items():
            s = entry["summary"]
            add(f"| {split} | {name} | **{s['overall']['accuracy']}%** "
                f"({s['overall']['correct']}/{s['overall']['total']}) | "
                f"{s['hallucination_rate']}% | {s['over_abstention_rate']}% | "
                f"{s['abstention_accuracy']}% |")
    add("")

    add("## Part 20 metric table")
    add("")
    header = "| metric |" + "".join(f" {split}/{b} |" for split, r in runs.items()
                                    for b in r["benchmarks"])
    add(header)
    add("| --- |" + " --- |" * (len(header.split("|")) - 3))
    rows = [(split, name, entry["summary"])
            for split, report in runs.items()
            for name, entry in report["benchmarks"].items()]
    for label, getter in METRICS:
        add(f"| {label} |" + "".join(f" {getter(s)}% |" for _, _, s in rows))
    for label, names in (("calculation accuracy",
                          ("ratios", "valuation", "corporate_finance", "statements")),
                         ("reasoning accuracy", ("reasoning",)),
                         ("interpretation accuracy", ("interpretation",)),
                         ("concept accuracy", ("concepts", "accounting", "reporting"))):
        add(f"| {label} |" + "".join(f" {category_accuracy(s, names)}% |"
                                    for _, _, s in rows))
    add("")

    add("## Which component answers, and how well")
    add("")
    for split, report in runs.items():
        for name, entry in report["benchmarks"].items():
            s = entry["summary"]
            add(f"**{split} / {name}**")
            add("")
            add("| component | share | answered | correct | accuracy |")
            add("| --- | --- | --- | --- | --- |")
            for component, pct in s["invocation_pct"].items():
                stats = s["correct_by_component"][component]
                add(f"| {component} | {pct}% | {stats['answered']} | "
                    f"{stats['correct']} | {stats['accuracy']}% |")
            add("")
    return "\n".join(lines)


def main():
    runs = load_runs()
    if not runs:
        sys.exit("no system_after_fixes_*.json runs found - run scripts/run_benchmarks.py")

    combined = {"splits": {}, "note": ("consolidated from per-split runs; no model "
                                       "weights were modified")}
    for split, report in runs.items():
        combined["splits"][split] = {
            "examples": report["examples"],
            "checkpoint_sha256": report["checkpoint_sha256"],
            "benchmarks": {name: {"description": entry["description"],
                                  "flags": entry["flags"],
                                  "summary": entry["summary"],
                                  "seconds_per_question": entry["seconds_per_question"]}
                           for name, entry in report["benchmarks"].items()},
        }
    with open(os.path.join(REPORT_DIR, "system_after_fixes.json"), "w",
              encoding="utf-8") as handle:
        json.dump(combined, handle, indent=2)
    text = markdown(runs)
    with open(os.path.join(REPORT_DIR, "system_after_fixes.md"), "w",
              encoding="utf-8") as handle:
        handle.write(text)
    print(text)


if __name__ == "__main__":
    main()
