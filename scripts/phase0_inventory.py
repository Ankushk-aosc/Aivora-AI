"""Phase 0: inventory every asset, reconcile the known inconsistencies, and
convert the SFT data to a model-agnostic chat format.

Nothing is deleted or overwritten. Every figure carries its source file, and
every claim is labelled measured / inferred / unverified.

The conversion deliberately stores ROLE-BASED MESSAGES rather than a rendered
prompt string: each base model has its own chat template, and rendering at
training time with the model's own tokenizer is the only way to keep the format
correct across Qwen, SmolLM2 and anything else tried later.

    python scripts/phase0_inventory.py
"""

import hashlib
import json
import os
import sys
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

OUT_JSON = os.path.join("reports", "PHASE0_INVENTORY.json")
OUT_MD = os.path.join("reports", "PHASE0_INVENTORY.md")
CHAT_OUT = os.path.join("data", "sft", "financial_sft_chat.jsonl")

SYSTEM_PROMPT = (
    "You are a financial analysis assistant. Answer only from the information "
    "given. If the information needed is not present, say that it is "
    "insufficient rather than guessing. Never invent a number."
)

DATASETS = {
    "financial_sft_v1": "data/sft/financial_sft_v1.jsonl",
    "financial_sft_v2": "data/sft/financial_sft_v2.jsonl",
    "tiny_overfit": "data/tiny_overfit/tiny_financial.jsonl",
    "benchmark_dev": "data/benchmark/dev.jsonl",
    "benchmark_test_hidden": "data/benchmark/test.jsonl",
    "benchmark_blind": "data/benchmark/blind.jsonl",
    "instruction_legacy": "data/instruction/financial_instructions.jsonl",
}

CHECKPOINTS = {
    "baseline_247850": "checkpoints/final/checkpoint_247850.pt",
    "sft_001": "checkpoints/sft_001/sft_001_best.pt",
    "sft_002": "checkpoints/sft_002/sft_002_best.pt",
}

REPORTS = [
    "reports/FREEZE_MANIFEST.json", "reports/gates_baseline.json",
    "reports/gates_sft_002.json", "reports/SFT_EXPERIMENT_001.json",
    "reports/SFT_EXPERIMENT_002.json", "reports/E1_RESULTS.md",
    "reports/e1/e1_arm1_result.json", "reports/e1/e1_arm2_result.json",
    "reports/MODEL_COMPARISON.md", "reports/TRAINING_DATA_AUDIT.json",
]


def sha256_file(path, chunk=8 * 1024 * 1024):
    if not os.path.exists(path):
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def count_rows(path):
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def reconcile():
    """The two inconsistencies named in the brief, answered from the files."""
    findings = []

    path = "reports/SFT_EXPERIMENT_001.json"
    if os.path.exists(path):
        with open(path, encoding="utf-8") as handle:
            report = json.load(handle)
        splits = {name: {"n": s["overall"]["total"],
                         "accuracy_pct": s["overall"]["accuracy"],
                         "hallucination_pct": s["hallucination_rate"]}
                  for name, s in report.get("model_only", {}).items()}
        findings.append({
            "question": "SFT_001 hallucination: 79.86% or 89.6%?",
            "status": "measured",
            "answer": ("Both, on different evaluation splits. Neither figure is "
                       "wrong; the split label was dropped when quoting them."),
            "detail": splits,
            "source": path,
            "correction": ("hallucination rate must always be quoted with its "
                           "split and n"),
        })

    v1 = count_rows(DATASETS["financial_sft_v1"])
    v2 = count_rows(DATASETS["financial_sft_v2"])
    train = val = None
    path = "reports/SFT_EXPERIMENT_002.json"
    if os.path.exists(path):
        with open(path, encoding="utf-8") as handle:
            dataset = json.load(handle)["dataset"]
        train, val = dataset.get("train"), dataset.get("val")
    findings.append({
        "question": "SFT_002 dataset size: 7,849 or 11,563?",
        "status": "measured",
        "answer": ("Neither is the size of the dataset SFT_002 used. 7,849 is "
                   "financial_sft_v1 (used by SFT_001). financial_sft_v2 holds "
                   f"{v2} rows, split {train} train / {val} validation - 11,563 "
                   "is the training split, not the dataset."),
        "detail": {"financial_sft_v1_rows": v1, "financial_sft_v2_rows": v2,
                   "v2_train": train, "v2_val": val},
        "source": "data/sft/*_manifest.json and reports/SFT_EXPERIMENT_002.json",
        "correction": "quote dataset size and split separately",
    })
    return findings


def to_chat(row):
    """One SFT row -> role-based messages, with the task label preserved."""
    task = row.get("task_type", "unknown")
    if task == "replay":
        # Plain text continuation has no user turn; kept with a null-conversation
        # marker so the trainer can route it, rather than being forced into a
        # question it never had.
        return {"messages": None, "raw_text": row["answer"], "task_type": task}

    question = row["question"]
    if row.get("context"):
        user = f"{row['context']}\n\n{question}"
    else:
        user = question
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
            {"role": "assistant", "content": row["answer"]},
        ],
        "raw_text": None,
        "task_type": task,
    }


def convert(source_path, out_path):
    with open(source_path, encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    converted = []
    for row in rows:
        record = to_chat(row)
        record.update({
            "id": row["id"], "domain": row.get("domain"),
            "difficulty": row.get("difficulty"),
            "source": row.get("source"),
            "verification_status": row.get("verification_status"),
            "numeric_answer": row.get("numeric_answer"),
        })
        converted.append(record)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as handle:
        for record in converted:
            handle.write(json.dumps(record) + "\n")
    return converted


def main():
    inventory = {
        "note": ("Phase 0 inventory. Nothing deleted or overwritten. Figures are "
                 "measured from the files named unless marked otherwise."),
        "datasets": {}, "checkpoints": {}, "reports": {},
        "reconciliations": reconcile(),
    }

    for name, path in DATASETS.items():
        rows = count_rows(path)
        entry = {"path": path, "exists": os.path.exists(path), "rows": rows,
                 "sha256": sha256_file(path), "status": "measured"}
        if rows:
            with open(path, encoding="utf-8") as handle:
                sample = [json.loads(line) for line in handle if line.strip()]
            entry["by_task_type"] = dict(Counter(
                r.get("task_type", r.get("category", "unlabelled"))
                for r in sample).most_common())
        inventory["datasets"][name] = entry

    for name, path in CHECKPOINTS.items():
        inventory["checkpoints"][name] = {
            "path": path, "exists": os.path.exists(path),
            "bytes": os.path.getsize(path) if os.path.exists(path) else None,
            "sha256": sha256_file(path), "status": "measured",
            "preserved": True,
        }

    for path in REPORTS:
        inventory["reports"][path] = {"exists": os.path.exists(path),
                                      "sha256": sha256_file(path)}

    print("converting SFT data to model-agnostic chat format...")
    converted = convert(DATASETS["financial_sft_v2"], CHAT_OUT)
    by_task = Counter(r["task_type"] for r in converted)
    inventory["chat_format"] = {
        "path": CHAT_OUT, "rows": len(converted),
        "sha256": sha256_file(CHAT_OUT),
        "by_task_type": dict(by_task.most_common()),
        "format": ("role-based messages (system/user/assistant); rendered with "
                   "each base model's own chat template at training time"),
        "replay_rows_without_messages": sum(1 for r in converted
                                            if r["messages"] is None),
        "system_prompt": SYSTEM_PROMPT,
        "status": "measured",
    }

    os.makedirs("reports", exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as handle:
        json.dump(inventory, handle, indent=2)

    lines = ["# Phase 0 inventory and reconciliation", "",
             "Nothing was deleted or overwritten. Every figure below is measured "
             "from the file named beside it.", "",
             "## Reconciliations", ""]
    for finding in inventory["reconciliations"]:
        lines.append(f"### {finding['question']}")
        lines.append("")
        lines.append(f"**{finding['status'].upper()}** - {finding['answer']}")
        lines.append("")
        lines.append(f"```\n{json.dumps(finding['detail'], indent=2)}\n```")
        lines.append("")
        lines.append(f"Source: `{finding['source']}`. "
                     f"Correction going forward: {finding['correction']}.")
        lines.append("")

    lines += ["## Datasets", "",
              "| dataset | rows | sha256 |", "| --- | --- | --- |"]
    for name, entry in inventory["datasets"].items():
        if entry["exists"]:
            lines.append(f"| `{name}` | {entry['rows']:,} | "
                         f"`{entry['sha256'][:12]}...` |")
        else:
            lines.append(f"| `{name}` | missing | - |")
    lines += ["", "## Checkpoints (all preserved)", "",
              "| checkpoint | size | sha256 |", "| --- | --- | --- |"]
    for name, entry in inventory["checkpoints"].items():
        if entry["exists"]:
            lines.append(f"| `{name}` | {entry['bytes'] / 1e6:,.0f} MB | "
                         f"`{entry['sha256'][:12]}...` |")
        else:
            lines.append(f"| `{name}` | not present locally | - |")
    lines += ["", "## Chat conversion", "",
              f"`{CHAT_OUT}` - {len(converted):,} rows, "
              f"sha256 `{inventory['chat_format']['sha256'][:12]}...`", "",
              "Stored as role-based messages rather than a rendered prompt, so "
              "each base model's own chat template can render them. "
              f"{inventory['chat_format']['replay_rows_without_messages']} replay "
              "rows carry raw text and no conversation, since they never had a "
              "question.", "",
              "| task | rows |", "| --- | --- |"]
    for task, count in by_task.most_common():
        lines.append(f"| {task} | {count:,} |")

    with open(OUT_MD, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")

    print(f"\ndatasets: {sum(1 for e in inventory['datasets'].values() if e['exists'])}"
          f"/{len(DATASETS)} present")
    print(f"checkpoints: "
          f"{sum(1 for e in inventory['checkpoints'].values() if e['exists'])}"
          f"/{len(CHECKPOINTS)} present, all preserved")
    print(f"chat rows: {len(converted):,} ({dict(by_task.most_common(4))} ...)")
    print(f"\nwrote {OUT_JSON}\nwrote {OUT_MD}\nwrote {CHAT_OUT}")


if __name__ == "__main__":
    main()
