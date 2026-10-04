"""Generate the Phase 3 Kaggle notebook: zero-shot and few-shot base models.

The models the plan names need 2.5-8 GB and do not fit the 1.2 GB free on the
development machine. This runs the SAME scripts/eval_base_models.py there, so the
prompts, decoding and scorer are identical to the local 135M run - the only thing
that changes is the hardware.

    python scripts/build_phase3_notebook.py
    cd training/kaggle/phase3 && kaggle kernels push -p .
"""

import json
import os

OUT_DIR = os.path.join("training", "kaggle", "phase3")
NOTEBOOK = "Aivora_Phase3_BaseModels.ipynb"
KERNEL_ID = "aoscjkjhh/aivora-phase3-base-models"

CELLS = []


def md(text):
    CELLS.append({"cell_type": "markdown", "metadata": {},
                  "source": text.splitlines(True)})


def code(text):
    CELLS.append({"cell_type": "code", "execution_count": None, "metadata": {},
                  "outputs": [], "source": text.splitlines(True)})


md("""# Aivora Phase 3 - zero-shot and few-shot base models

Measures open instruct models on the **selection split** of the held-out
evaluation, with identical prompts, identical greedy decoding and the identical
scorer used for every other system in this project.

| model | licence | why |
| --- | --- | --- |
| `HuggingFaceTB/SmolLM2-135M-Instruct` | Apache-2.0 | the local floor, re-run here to confirm the hardware changes nothing |
| `HuggingFaceTB/SmolLM2-360M-Instruct` | Apache-2.0 | next size up |
| `Qwen/Qwen2.5-0.5B-Instruct` | Apache-2.0 | the plan's smallest named model |
| `Qwen/Qwen2.5-1.5B-Instruct` | Apache-2.0 | the model this project already has LoRA history with |
| `HuggingFaceTB/SmolLM2-1.7B-Instruct` | Apache-2.0 | largest that fits a free T4 comfortably |

`meta-llama/Llama-3.2-1B-Instruct` is deliberately **excluded**: it is gated
behind accepting the Llama licence, and nothing here should depend on a licence
the project has not accepted.

**Status: PROVISIONAL.** The scorer has not yet been validated against hand
labels (EXPERIMENT_RULES_v2 section 4), and these are selection-split numbers,
which choose a model and are never reported as results. The frozen set is scored
once per configuration, later.

Settings: Accelerator -> GPU T4, Internet -> On.
""")

code('''import subprocess, os, json, time

REPO_URL = "https://github.com/Ankushk-aosc/Aivora-AI.git"
REPO_DIR = "/kaggle/working/Aivora-AI"

if not os.path.exists(REPO_DIR):
    r = subprocess.run(["git", "clone", "--depth", "1", REPO_URL, REPO_DIR],
                       capture_output=True, text=True)
    print(r.stdout[-800:], r.stderr[-800:])
    if r.returncode != 0:
        raise RuntimeError("STATUS = BLOCKED: git clone failed. Internet -> On?")
os.chdir(REPO_DIR)

for path in ("scripts/eval_base_models.py", "scripts/eval_heldout.py",
             "data/eval_heldout/selection.jsonl",
             "data/eval_heldout/heldout_frozen.jsonl"):
    if not os.path.exists(path):
        raise RuntimeError(f"STATUS = BLOCKED: {path} missing - the clone is "
                           f"stale. Push the Phase 3 commit first.")

head = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                      capture_output=True, text=True).stdout.strip()
print("repo at", head)

# The evaluation set must be the frozen one, byte for byte.
import hashlib
digest = hashlib.sha256(open("data/eval_heldout/heldout_frozen.jsonl", "rb").read()).hexdigest()
print("frozen sha256", digest[:16], "...")
''')

md("## GPU check - refuse rather than silently running on CPU for hours")

code('''import torch

print("torch", torch.__version__, "| cuda", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise RuntimeError("STATUS = BLOCKED: no GPU. Settings -> Accelerator -> GPU.")
for i in range(torch.cuda.device_count()):
    properties = torch.cuda.get_device_properties(i)
    print(f"  {properties.name}, {properties.total_memory / 1e9:.1f} GB")

import transformers
print("transformers", transformers.__version__)
''')

md("""## Run every configuration

Each is a separate process, so one model failing to download cannot take the
others with it. Raw outputs are saved per configuration, as the rules require.""")

code('''MODELS = [
    "HuggingFaceTB/SmolLM2-135M-Instruct",
    "HuggingFaceTB/SmolLM2-360M-Instruct",
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "HuggingFaceTB/SmolLM2-1.7B-Instruct",
]
SHOTS = [0, 3]
SPLIT = "selection"          # never the frozen set at this stage

results, failures = {}, {}
for model in MODELS:
    for shots in SHOTS:
        key = f"{model} ({shots}-shot)"
        print(f"\\n{'=' * 70}\\n{key}\\n{'=' * 70}", flush=True)
        started = time.time()
        proc = subprocess.run(
            ["python", "scripts/eval_base_models.py", "--model", model,
             "--shots", str(shots), "--split", SPLIT, "--force"],
            capture_output=True, text=True)
        print(proc.stdout[-2500:])
        if proc.returncode != 0:
            print("FAILED:", proc.stderr[-1500:])
            failures[key] = proc.stderr[-600:]
            continue
        slug = model.replace("/", "_") + f"_{shots}shot_{SPLIT}"
        path = f"reports/heldout/{slug}.json"
        if os.path.exists(path):
            with open(path) as handle:
                results[key] = json.load(handle)["summary"]
            print(f"  {time.time() - started:.0f}s")
''')

md("## Summary across models")

code('''rows = []
for key, summary in results.items():
    gates = summary["by_gate"]
    rows.append({
        "model": key,
        "overall": summary["overall"]["accuracy_pct"],
        "copy": gates.get("copy", {}).get("accuracy_pct"),
        "extraction": gates.get("extraction", {}).get("accuracy_pct"),
        "wording": gates.get("wording", {}).get("accuracy_pct"),
        "calculation": gates.get("calculation", {}).get("accuracy_pct"),
        "abstention": gates.get("abstention", {}).get("accuracy_pct"),
        "invented_values_pct": summary["hallucination_invented_values"]["pct"],
        "over_refusal_pct": summary["over_refusal"]["pct"],
        "seconds_per_item": summary.get("seconds_per_item"),
    })

rows.sort(key=lambda r: -(r["extraction"] or 0))
header = (f"{'model':<46}{'overall':>8}{'copy':>7}{'extr':>7}{'word':>7}"
          f"{'calc':>7}{'abst':>7}{'inv%':>7}")
print(header)
print("-" * len(header))
for row in rows:
    print(f"{row['model']:<46}{row['overall']:>7.1f}%{row['copy'] or 0:>6.1f}%"
          f"{row['extraction'] or 0:>6.1f}%{row['wording'] or 0:>6.1f}%"
          f"{row['calculation'] or 0:>6.1f}%{row['abstention'] or 0:>6.1f}%"
          f"{row['invented_values_pct']:>6.1f}%")

payload = {
    "phase": 3,
    "status": "PROVISIONAL - scorer not yet validated against hand labels; "
              "selection split, used to choose a model and never reported as a result",
    "split": SPLIT,
    "repo_commit": head,
    "frozen_sha256": digest,
    "results": results,
    "failures": failures,
    "aivora_reference_frozen_split": {
        "note": "frozen split, NOT comparable item-for-item with the rows above",
        "baseline": {"copy": 13.04, "extraction": 1.45, "wording": 2.67,
                     "calculation": 0.0, "abstention": 0.0,
                     "invented_values_pct": 42.91},
        "sft_002": {"copy": 34.78, "extraction": 13.04, "wording": 13.33,
                    "calculation": 2.08, "abstention": 16.67,
                    "invented_values_pct": 36.78},
    },
}
with open("/kaggle/working/phase3_base_models.json", "w") as handle:
    json.dump(payload, handle, indent=2)
print("\\nwrote /kaggle/working/phase3_base_models.json")
if failures:
    print("failed configurations:", list(failures))
''')

md("""## What this does and does not establish

**Does:** a like-for-like ranking of open instruct models on the same items, with
the same prompts, decoding and scorer, on hardware where they actually fit.

**Does not:** a result. These are selection-split numbers under an unvalidated
scorer. They choose which model goes forward; the frozen set is scored once, for
the chosen configuration, after the scorer is validated against hand labels.

**Also does not:** say anything about the tool pipeline. Calculation and
abstention are expected to be weak here - Python arithmetic and rule-based
refusal are Phase 4's job, not the base model's.

Download `/kaggle/working/phase3_base_models.json` and the per-configuration
files under `reports/heldout/` before the session ends.
""")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    notebook = {
        "cells": CELLS,
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                    "name": "python3"},
                     "language_info": {"name": "python", "version": "3.11"}},
        "nbformat": 4, "nbformat_minor": 5,
    }
    path = os.path.join(OUT_DIR, NOTEBOOK)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(notebook, handle, indent=1)

    metadata = {
        "id": KERNEL_ID,
        "title": "Aivora Phase3 Base Models",
        "code_file": NOTEBOOK,
        "language": "python", "kernel_type": "notebook",
        "is_private": "true", "enable_gpu": "true", "enable_tpu": "false",
        "enable_internet": "true",
        "dataset_sources": [], "competition_sources": [],
        "kernel_sources": [], "model_sources": [],
    }
    with open(os.path.join(OUT_DIR, "kernel-metadata.json"), "w",
              encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    print(f"wrote {path} ({len(CELLS)} cells)")
    print(f"wrote {os.path.join(OUT_DIR, 'kernel-metadata.json')}")


if __name__ == "__main__":
    main()
