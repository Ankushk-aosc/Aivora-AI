"""Parts 19-20: the SYSTEM benchmark and the MODEL benchmark, kept apart.

Two questions get confused by a single number, so they get separate runs:

  SYSTEM  - the served pipeline: routing, extraction, calculator, glossary,
            retrieval, diagnostic patterns, guard, and Aivora as the fallback.
            This is what a user experiences.
  MODEL   - Aivora alone. Every deterministic layer is switched off, so nothing
            but the 101M checkpoint answers. This is the only number that says
            anything about the model, and it is the one that decides whether
            training is justified.

It also records WHICH component answered each question, because "the system
scores X" means something different when the model answered 2% of the questions
than when it answered half of them.

usage:
    python scripts/run_benchmarks.py --split dev
    python scripts/run_benchmarks.py --split test --tag hidden
    python scripts/run_benchmarks.py --split blind --benchmarks system model
"""

import argparse
import json
import os
import sys
import time
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

CHECKPOINT = os.path.join("checkpoints", "final", "checkpoint_247850.pt")
SEED = 1234

# source string reported by the pipeline -> the component that produced it
COMPONENT_BY_SOURCE = {
    "FINANCIAL CALCULATOR": "calculator",
    "STATEMENT EXTRACTION": "extraction",
    "FINANCIAL KNOWLEDGE BASE (DOMAIN GLOSSARY)": "glossary",
    "FINANCIAL KNOWLEDGE BASE (RETRIEVED)": "retrieval",
    "FINANCIAL ANALYSIS PATTERN (CURATED)": "analysis_pattern",
    "MODEL": "model",
    "MODEL KNOWLEDGE": "model",
    "NOT AVAILABLE": "abstention",
}

BENCHMARKS = {
    # name: (flags for FinancialChat, description)
    "system": ({}, "the served pipeline: every layer enabled, Aivora as fallback"),
    "model": ({"use_calculator": False, "use_knowledge": False, "use_guard": False},
              "Aivora alone: calculator, glossary, retrieval, patterns and guard all off"),
}


def component_of(response):
    source = response.source or ""
    if source in COMPONENT_BY_SOURCE:
        return COMPONENT_BY_SOURCE[source]
    for prefix, component in COMPONENT_BY_SOURCE.items():
        if source.startswith(prefix):
            return component
    if "withheld" in source or "NOT AVAILABLE" in source:
        return "abstention"
    return f"other:{source[:28]}"


def run(name, items, model, device="cpu"):
    import torch

    from app.backend.services.chat_service import FinancialChat
    from evaluation.benchmark import score_item, summarise

    flags, description = BENCHMARKS[name]
    chat = FinancialChat(model=model, device=device, max_new_tokens=40, **flags)

    records, components, started = [], Counter(), time.time()
    for index, item in enumerate(items, 1):
        torch.manual_seed(SEED)
        response = chat.ask(item["question"])
        record = score_item(item, response.answer)
        record["component"] = component_of(response)
        record["route"] = response.route
        components[record["component"]] += 1
        records.append(record)
        if index % 100 == 0:
            print(f"    {index}/{len(items)}")
    elapsed = time.time() - started

    summary = summarise(records)
    total = max(len(records), 1)
    summary["invocation_pct"] = {component: round(100.0 * count / total, 2)
                                 for component, count in components.most_common()}
    # Accuracy attributable to each component, which is what "the calculator
    # carries the system" has to be measured against rather than asserted.
    summary["correct_by_component"] = {}
    for component in components:
        subset = [r for r in records if r["component"] == component]
        correct = sum(1 for r in subset if r["correct"])
        summary["correct_by_component"][component] = {
            "answered": len(subset), "correct": correct,
            "accuracy": round(100.0 * correct / len(subset), 2) if subset else None,
        }
    return {"benchmark": name, "description": description, "flags": flags,
            "examples": len(records), "seconds": round(elapsed, 1),
            "seconds_per_question": round(elapsed / total, 3),
            "summary": summary, "details": records}


def markdown(report):
    lines, add = [], None
    add = lines.append
    add(f"# Benchmark run: {report['split']} split")
    add("")
    add(f"{report['examples']} items. Checkpoint `{os.path.basename(report['checkpoint'])}`, "
        f"sha256 `{report['checkpoint_sha256'][:12]}...`, seed {SEED}. No weights were "
        "modified to produce these numbers.")
    add("")
    add("| benchmark | accuracy | hallucination | over-abstention | abstention acc. | s/question |")
    add("| --- | --- | --- | --- | --- | --- |")
    for name, entry in report["benchmarks"].items():
        s = entry["summary"]
        add(f"| **{name}** | {s['overall']['accuracy']}% "
            f"({s['overall']['correct']}/{s['overall']['total']}) | "
            f"{s['hallucination_rate']}% | {s['over_abstention_rate']}% | "
            f"{s['abstention_accuracy']}% | {entry['seconds_per_question']} |")
    add("")
    for name, entry in report["benchmarks"].items():
        s = entry["summary"]
        add(f"## {name} - {entry['description']}")
        add("")
        add("| metric | value |")
        add("| --- | --- |")
        for label, key in (("extraction accuracy", "extraction_accuracy"),
                           ("abstention accuracy", "abstention_accuracy"),
                           ("hallucination rate", "hallucination_rate"),
                           ("over-abstention rate", "over_abstention_rate")):
            add(f"| {label} | {s[key]}% |")
        add("")
        add("| category | accuracy |")
        add("| --- | --- |")
        for category, bucket in sorted(s["by_category"].items()):
            add(f"| {category} | {bucket['accuracy']}% ({bucket['correct']}/{bucket['total']}) |")
        add("")
        add("| item kind | accuracy |")
        add("| --- | --- |")
        for kind, bucket in sorted(s["by_kind"].items()):
            add(f"| {kind} | {bucket['accuracy']}% ({bucket['correct']}/{bucket['total']}) |")
        add("")
        add("Which component answered, and how well:")
        add("")
        add("| component | share of questions | answered | correct | accuracy |")
        add("| --- | --- | --- | --- | --- |")
        for component, pct in s["invocation_pct"].items():
            stats = s["correct_by_component"][component]
            add(f"| {component} | {pct}% | {stats['answered']} | {stats['correct']} | "
                f"{stats['accuracy']}% |")
        add("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="dev",
                        help="dev, test (the hidden split) or blind")
    parser.add_argument("--benchmarks", nargs="*", default=list(BENCHMARKS))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--checkpoint", default=CHECKPOINT)
    parser.add_argument("--out", default="reports")
    parser.add_argument("--tag", default="")
    args = parser.parse_args()

    from evaluation.benchmark import load_benchmark
    from inference import load_model_for_inference
    from scripts.baseline_report import sha256

    items = load_benchmark(args.split)
    if args.limit:
        items = items[:args.limit]
    print(f"{args.split}: {len(items)} items")

    model, _ = load_model_for_inference(args.checkpoint, device="cpu")
    report = {
        "split": args.split,
        "examples": len(items),
        "checkpoint": args.checkpoint,
        "checkpoint_sha256": sha256(args.checkpoint),
        "seed": SEED,
        "benchmarks": {},
    }
    for name in args.benchmarks:
        print(f"\n{name}: {BENCHMARKS[name][1]}")
        entry = run(name, items, model)
        report["benchmarks"][name] = entry
        s = entry["summary"]
        print(f"  accuracy {s['overall']['accuracy']}%  hallucination "
              f"{s['hallucination_rate']}%  answered by: {s['invocation_pct']}")

    os.makedirs(args.out, exist_ok=True)
    tag = args.tag or args.split
    json_path = os.path.join(args.out, f"system_after_fixes_{tag}.json")
    md_path = os.path.join(args.out, f"system_after_fixes_{tag}.md")
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    with open(md_path, "w", encoding="utf-8") as handle:
        handle.write(markdown(report))
    print(f"\nwrote {json_path}\nwrote {md_path}")


if __name__ == "__main__":
    main()
