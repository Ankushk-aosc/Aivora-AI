"""Phase 3: the measured baseline for Aivora v0.1-base, with ablations.

Runs the benchmark's dev split through four configurations of the SAME serving
pipeline, with components switched off via the flags on FinancialChat, so the
numbers describe the real system rather than a re-implementation of it:

    model_only        the 101M checkpoint's own output, nothing else
    plus_calculator   + the deterministic calculator and derived identities
    plus_retrieval    + the curated glossary and the retrieval layer
    full_pipeline     + the output-quality guard and live-data handling

Determinism: the model samples at the production settings (temperature 0.7,
top_k 40, top_p 0.9, repetition penalty 1.3), and torch is re-seeded per
question, so a rerun reproduces the same answers. Greedy decoding was tried
and rejected: it collapses to two or three tokens ("EBIT"), which would measure
the decoder rather than the model.

The hidden test split is NOT touched: --split defaults to dev, and the hidden
set is reserved for a final measurement once the improvement work is done.

usage:
    python scripts/baseline_report.py                     # full dev split
    python scripts/baseline_report.py --limit 150         # quick pass
    python scripts/baseline_report.py --configs model_only full_pipeline
"""

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

DEFAULT_CHECKPOINT = os.path.join("checkpoints", "final", "checkpoint_247850.pt")
REPORT_DIR = "reports"
SEED = 1234

# name -> (flags, one-line description for the report)
CONFIGS = {
    "model_only": (
        {"use_calculator": False, "use_knowledge": False, "use_guard": False},
        "the 101M model alone - no calculator, no glossary, no retrieval, no guard"),
    "plus_calculator": (
        {"use_calculator": True, "use_knowledge": False, "use_guard": False},
        "model + deterministic calculator and derived identities"),
    "plus_retrieval": (
        {"use_calculator": False, "use_knowledge": True, "use_guard": False},
        "model + curated glossary and retrieval layer"),
    "full_pipeline": (
        {"use_calculator": True, "use_knowledge": True, "use_guard": True},
        "the served pipeline: calculator + knowledge + quality guard"),
}


def sha256(path, chunk=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def git_revision():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"],
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


def environment(checkpoint):
    import torch

    return {
        "git_revision": git_revision(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "platform": platform.platform(),
        "cuda_available": torch.cuda.is_available(),
        "checkpoint": checkpoint,
        "checkpoint_sha256": sha256(checkpoint),
        "seed": SEED,
        "sampling": {"temperature": 0.7, "top_k": 40, "top_p": 0.9,
                     "repetition_penalty": 1.3, "max_new_tokens": 40},
    }


def build_answer_fn(chat):
    """Re-seed per question so the run is reproducible regardless of order."""
    import torch

    def answer(question):
        torch.manual_seed(SEED)
        return chat.ask(question).answer

    return answer


def run(split, limit, config_names, checkpoint):
    from app.backend.services.chat_service import FinancialChat
    from evaluation.benchmark import evaluate_benchmark
    from inference import load_model_for_inference

    print(f"loading {checkpoint} ...")
    started = time.time()
    model, model_config = load_model_for_inference(checkpoint, device="cpu")
    print(f"loaded in {time.time() - started:.1f}s")

    report = {
        "label": "Aivora v0.1-base baseline",
        "split": split,
        "limit": limit,
        "environment": environment(checkpoint),
        "model_parameters": sum(p.numel() for p in model.parameters()),
        "configurations": {},
    }

    for name in config_names:
        flags, description = CONFIGS[name]
        chat = FinancialChat(model=model, device="cpu", max_new_tokens=40, **flags)
        print(f"\n=== {name}: {description}")
        started = time.time()
        results = evaluate_benchmark(build_answer_fn(chat), split=split, limit=limit,
                                    label=f"{name} ({split})")
        elapsed = time.time() - started
        report["configurations"][name] = {
            "description": description,
            "flags": flags,
            "examples": results["examples"],
            "summary": results["summary"],
            "seconds": round(elapsed, 1),
            "seconds_per_question": round(elapsed / max(results["examples"], 1), 3),
            "details": results["details"],
        }
        summary = results["summary"]
        print(f"  overall {summary['overall']['accuracy']}%  "
              f"hallucination {summary['hallucination_rate']}%  "
              f"abstention {summary['abstention_accuracy']}%  "
              f"{elapsed / max(results['examples'], 1):.2f}s/question")

    return report


def markdown(report):
    """A report a person can read, with the caveats attached to the numbers."""
    env = report["environment"]
    configs = report["configurations"]
    order = [n for n in CONFIGS if n in configs]
    lines = []
    add = lines.append

    add("# Aivora v0.1-base - measured baseline")
    add("")
    add(f"Benchmark: `data/benchmark/{report['split']}.jsonl`, "
        f"{configs[order[0]]['examples']} items. Produced by "
        "`scripts/baseline_report.py`; every number below is measured, not estimated.")
    add("")
    add(f"* checkpoint `{os.path.basename(env['checkpoint'])}`, sha256 "
        f"`{env['checkpoint_sha256'][:12]}...`")
    add(f"* {report['model_parameters']:,} parameters, CPU, torch {env['torch']}")
    add(f"* sampling {env['sampling']}, torch re-seeded to {env['seed']} per question")
    add(f"* git revision `{(env['git_revision'] or 'unknown')[:12]}`")
    add("")
    add("The hidden test split was not used.")
    add("")

    add("## Headline")
    add("")
    add("| configuration | accuracy | hallucination rate | over-abstention | "
        "abstention accuracy | s/question |")
    add("| --- | --- | --- | --- | --- | --- |")
    for name in order:
        entry = configs[name]
        summary = entry["summary"]
        add(f"| {name} | **{summary['overall']['accuracy']}%** "
            f"({summary['overall']['correct']}/{summary['overall']['total']}) | "
            f"{summary['hallucination_rate']}% | {summary['over_abstention_rate']}% | "
            f"{summary['abstention_accuracy']}% | {entry['seconds_per_question']} |")
    add("")
    add("*hallucination rate* = asserted a wrong answer to a question that had one. "
        "*over-abstention* = declined a question that had an answer. "
        "*abstention accuracy* = correctly declined the unanswerable items. "
        "A system can only look good on all three at once by actually knowing "
        "which questions it can answer.")
    add("")

    add("## By category")
    add("")
    categories = sorted({c for name in order
                         for c in configs[name]["summary"]["by_category"]})
    add("| category | " + " | ".join(order) + " |")
    add("| --- |" + " --- |" * len(order))
    for category in categories:
        row = [category]
        for name in order:
            bucket = configs[name]["summary"]["by_category"].get(category)
            row.append(f"{bucket['accuracy']}% ({bucket['correct']}/{bucket['total']})"
                       if bucket else "-")
        add("| " + " | ".join(row) + " |")
    add("")

    add("## By difficulty level")
    add("")
    levels = sorted({l for name in order for l in configs[name]["summary"]["by_level"]})
    add("| level | " + " | ".join(order) + " |")
    add("| --- |" + " --- |" * len(order))
    for level in levels:
        row = [f"level {level}"]
        for name in order:
            bucket = configs[name]["summary"]["by_level"].get(level)
            row.append(f"{bucket['accuracy']}%" if bucket else "-")
        add("| " + " | ".join(row) + " |")
    add("")

    add("## By item kind")
    add("")
    kinds = sorted({k for name in order for k in configs[name]["summary"]["by_kind"]})
    add("| kind | " + " | ".join(order) + " |")
    add("| --- |" + " --- |" * len(order))
    for kind in kinds:
        row = [kind]
        for name in order:
            bucket = configs[name]["summary"]["by_kind"].get(kind)
            row.append(f"{bucket['accuracy']}% ({bucket['correct']}/{bucket['total']})"
                       if bucket else "-")
        add("| " + " | ".join(row) + " |")
    add("")

    add("## What each component is worth")
    add("")
    base = configs[order[0]]["summary"]["overall"]["accuracy"]
    for name in order[1:]:
        delta = configs[name]["summary"]["overall"]["accuracy"] - base
        add(f"* **{name}**: {delta:+.2f} points against model_only "
            f"({configs[name]['description']})")
    add("")

    add("## Sample failures")
    add("")
    worst = configs[order[-1]]
    shown = 0
    for record in worst["details"]:
        if record["correct"] or shown >= 8:
            continue
        add(f"* `{record['id']}` ({record['category']}, {record['outcome']}): "
            f"{record['prediction'][:160].strip()!r}")
        shown += 1
    add("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="dev",
                        help="dev (default). The hidden test split is reserved.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--configs", nargs="*", default=list(CONFIGS))
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--out", default=REPORT_DIR)
    args = parser.parse_args()

    unknown = [c for c in args.configs if c not in CONFIGS]
    if unknown:
        sys.exit(f"unknown configs {unknown}; choose from {list(CONFIGS)}")
    if not os.path.exists(args.checkpoint):
        sys.exit(f"checkpoint not found: {args.checkpoint}")

    report = run(args.split, args.limit, args.configs, args.checkpoint)

    os.makedirs(args.out, exist_ok=True)
    suffix = "" if args.split == "dev" else f"_{args.split}"
    json_path = os.path.join(args.out, f"baseline_report{suffix}.json")
    md_path = os.path.join(args.out, f"baseline_report{suffix}.md")
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    with open(md_path, "w", encoding="utf-8") as handle:
        handle.write(markdown(report))
    print(f"\nwrote {json_path}\nwrote {md_path}")


if __name__ == "__main__":
    main()
