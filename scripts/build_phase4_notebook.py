"""Generate the Phase 4 Kaggle notebook: the tool pipeline on a GPU.

Qwen2.5-1.5B-Instruct needs about 7 GB and does not fit the development machine,
so the SAME scripts/run_tool_pipeline_eval.py runs here. Nothing about the
pipeline, the prompts or the scorer changes - only the hardware.

    python scripts/build_phase4_notebook.py
    cd training/kaggle/phase4 && kaggle kernels push -p .
"""

import json
import os

OUT_DIR = os.path.join("training", "kaggle", "phase4")
NOTEBOOK = "Aivora_Phase4_ToolPipeline.ipynb"
TAG = "synonyms"
KERNEL_ID = "aoscjkjhh/aivora-phase4-tool-pipeline"

CELLS = []


def md(text):
    CELLS.append({"cell_type": "markdown", "metadata": {},
                  "source": text.splitlines(True)})


def code(text):
    CELLS.append({"cell_type": "code", "execution_count": None, "metadata": {},
                  "outputs": [], "source": text.splitlines(True)})


md("""# Aivora Phase 4 (v2) - the tool pipeline with an operand vocabulary

**Configuration: Qwen2.5-1.5B-Instruct, zero-shot** (set by the owner,
2026-10-05). Phase 3 put 0-shot and 3-shot within one wording item of each other
with overlapping intervals, and 0-shot avoids the few-shot abstention artifact in
`reports/phase3/PHASE3_REPORT.md` section 3. The pipeline brings its own
structured prompt, so a few-shot block adds nothing.

Division of labour - the model does only what Phase 3 showed it is good at:

| | does |
| --- | --- |
| the model | names the field and copies the value **with its source span** |
| Python | validates the span, picks the formula, performs **all** arithmetic |
| rules | abstain whenever validation fails or an operand is missing |

**What changed since the first run.** Calculation scored 83.3% and every one of
the 8 failures was the same thing: the formula asked for "total debt" while the
statement said "Borrowings", so the model truthfully reported the label absent
and the pipeline abstained. `SYNONYMS` in `pipeline/tool_pipeline.py` now maps
each operand to the captions a statement may actually use, and Python - not the
model - decides which caption is present before asking for it.

**Read the frozen number with this caveat.** The diagnosis came from frozen-set
failures, so this configuration was informed by the frozen set even though the
vocabulary was written from statutory captions rather than transcribed from the
failing items. Two entries are tailored to phrasings seen there and are labelled
TAILORED in the source. A fully clean measurement of this configuration needs a
held-out set it has never informed.

Two splits are run, in this order and for different reasons:

1. **selection** - confirms the pipeline behaves, and is the only split allowed
   to influence any choice.
2. **frozen** - scored **once** for this configuration, with raw outputs saved.

Raw per-item outputs are preserved for both, so when the scorer is validated
against the owner's hand labels the frozen run can be **re-scored without
another GPU session**. Re-scoring saved outputs is not a second look at the
frozen set: the model is run once, and the outputs are fixed from that moment.

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

REQUIRED = ("scripts/run_tool_pipeline_eval.py", "pipeline/tool_pipeline.py",
            "pipeline/schema.py", "scripts/eval_heldout.py",
            "tests/test_tool_pipeline.py",
            "data/eval_heldout/selection.jsonl",
            "data/eval_heldout/heldout_frozen.jsonl",
            "data/eval_heldout/heldout_manifest.json")
for path in REQUIRED:
    if not os.path.exists(path):
        raise RuntimeError(f"STATUS = BLOCKED: {path} missing - the clone is "
                           f"stale. Push the Phase 4 commit first.")

head = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                      capture_output=True, text=True).stdout.strip()
print("repo at", head)
''')

md("""## Eval-set integrity, the portable way

The Phase 3 run reported a frozen-set hash that disagreed with the manifest. That
was line endings - `core.autocrlf=true` gives the Windows checkout CRLF and this
one LF - but a byte hash that is *expected* to differ between platforms cannot
detect anything, so it was replaced with a hash over the parsed items in
canonical order. It is identical on every platform and still changes if any item
changes. Both splits must come back VERIFIED.""")

code('''import sys
sys.path.insert(0, os.getcwd())
from scripts.eval_heldout import content_sha256, verify_split, FROZEN, SELECTION

for path, split in ((SELECTION, "selection"), (FROZEN, "frozen")):
    digest, status = verify_split(path, split)      # raises if it does not match
    print(f"{split:<10} {status:<10} content_sha256 {digest}")
''')

md("""## The pipeline's guarantees, before any score

These run against a scripted stub, so they test the guarantees rather than the
model: a value whose span is absent never reaches the user, unparseable output
becomes an abstention, arithmetic happens in Python, a missing operand abstains,
and answerable questions are not refused. If any of this fails, no score from
this notebook means anything.""")

code('''proc = subprocess.run(["python", "tests/test_tool_pipeline.py"],
                      capture_output=True, text=True)
print(proc.stdout[-1500:])
if proc.returncode != 0:
    raise RuntimeError("STATUS = BLOCKED: the pipeline's own guarantees fail. "
                       "Do not score anything until this passes.")
''')

md("""## GPU check - refuse rather than run for hours on CPU""")

code('''import torch

print("torch", torch.__version__, "| cuda", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise RuntimeError("STATUS = BLOCKED: no GPU. Settings -> Accelerator -> GPU.")
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"  {p.name}, {p.total_memory / 1e9:.1f} GB")
''')

md("""## Wire check with a stub, on the real items

A model that finds nothing must produce 100% abstention, 0% invented values and
zero span violations. This costs no GPU time and catches a broken harness before
the real run.""")

code('''proc = subprocess.run(["python", "scripts/run_tool_pipeline_eval.py",
                       "--stub", "--split", "selection"],
                      capture_output=True, text=True)
print(proc.stdout[-1800:])
if proc.returncode != 0:
    raise RuntimeError("STATUS = BLOCKED: harness failure\\n" + proc.stderr[-1200:])
''')

md("""## 1. Selection split - behaviour check

The only split allowed to influence a choice.""")

code('''MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
TAG = "synonyms"          # names the configuration, so nothing is overwritten
runs = {}

def run(split):
    print(f"\\n{'=' * 70}\\n{MODEL} pipeline, {split} split\\n{'=' * 70}", flush=True)
    started = time.time()
    proc = subprocess.run(["python", "scripts/run_tool_pipeline_eval.py",
                           "--model", MODEL, "--split", split,
                           "--tag", TAG, "--force"],
                          capture_output=True, text=True)
    print(proc.stdout[-4000:])
    if proc.returncode != 0:
        print("FAILED:", proc.stderr[-2000:])
        return None
    path = f"reports/heldout/{MODEL.replace('/', '_')}_pipeline_{split}_{TAG}.json"
    with open(path) as handle:
        payload = json.load(handle)
    print(f"  {time.time() - started:.0f}s")
    return payload

runs["selection"] = run("selection")
''')

md("""## 2. Frozen split - scored once, raw outputs saved

This is the reportable run. It happens exactly once for this configuration.""")

code('''runs["frozen"] = run("frozen")
''')

md("""## Summary""")

code('''summary_payload = {
    "phase": 4,
    "configuration": f"{MODEL} (0-shot) + tool pipeline [{TAG}]",
    "chosen_by": "owner, 2026-10-05, on the Phase 3 selection-split evidence",
    "repo_commit": head,
    "status": "PROVISIONAL SCORES, FIXED OUTPUTS - the scorer is not yet "
              "validated against hand labels (EXPERIMENT_RULES_v2 section 4) "
              "and the section 1 thresholds are unset. The model ran once per "
              "split; raw outputs are saved and can be re-scored without a GPU.",
    "splits": {},
}

for split, payload in runs.items():
    if payload is None:
        summary_payload["splits"][split] = {"status": "FAILED"}
        continue
    s = payload["summary"]
    summary_payload["splits"][split] = {
        "overall": s["overall"], "by_gate": s["by_gate"],
        "over_refusal": s["over_refusal"],
        "hallucination_invented_values": s["hallucination_invented_values"],
        "span_audit": s["span_audit"], "components": s["components"],
        "abstained_pct": s["abstained_pct"],
        "llm_calls_per_item": s["llm_calls_per_item"],
        "seconds_per_item": s["seconds_per_item"],
        "eval_set_integrity": s.get("eval_set_integrity"),
    }

header = f"{'split':<12}{'overall':>9}{'copy':>7}{'extr':>7}{'word':>7}{'calc':>7}{'abst':>7}{'inv%':>7}{'span%':>8}"
print(header)
print("-" * len(header))
for split, block in summary_payload["splits"].items():
    if block.get("status") == "FAILED":
        print(f"{split:<12}  FAILED")
        continue
    g = block["by_gate"]
    pick = lambda name: g.get(name, {}).get("accuracy_pct") or 0.0
    print(f"{split:<12}{block['overall']['accuracy_pct']:>8.1f}%"
          f"{pick('copy'):>6.1f}%{pick('extraction'):>6.1f}%{pick('wording'):>6.1f}%"
          f"{pick('calculation'):>6.1f}%{pick('abstention'):>6.1f}%"
          f"{block['hallucination_invented_values']['pct']:>6.1f}%"
          f"{block['span_audit']['span_validated_pct']:>7.1f}%")

print()
for split, block in summary_payload["splits"].items():
    if block.get("status") == "FAILED":
        continue
    audit = block["span_audit"]
    verdict = ("HOLDS" if audit["violation_count"] == 0
               else f"VIOLATED {audit['violation_count']}x - BLOCKS SHIP")
    print(f"{split}: span rule {verdict} "
          f"({audit['answers_checked']} stated values checked)")
    print(f"{split}: components {block['components']}, "
          f"abstained {block['abstained_pct']}%")

with open(f"/kaggle/working/phase4_tool_pipeline_{TAG}.json", "w") as handle:
    json.dump(summary_payload, handle, indent=2)
print("\\nwrote /kaggle/working/phase4_tool_pipeline.json")
''')

md("""## Copy the raw outputs out

The per-item files are the point: they carry every answer, the spans it cited,
the component that produced it and the operands of every calculation. They are
what makes a failure diagnosable in Phase 6, and what allows re-scoring once the
scorer is validated.""")

code('''import shutil

copied = []
for split in ("selection", "frozen"):
    src = f"reports/heldout/{MODEL.replace('/', '_')}_pipeline_{split}_{TAG}.json"
    if os.path.exists(src):
        dst = f"/kaggle/working/{os.path.basename(src)}"
        shutil.copy(src, dst)
        copied.append((dst, os.path.getsize(dst)))
for path, size in copied:
    print(f"{size / 1024:>8.0f} KB  {path}")
print("\\nDownload these and phase4_tool_pipeline_{TAG}.json before the session ends.")
''')

md("""## What this establishes, and what it does not

**Establishes:** how a pre-trained extractor behind a span-validating,
Python-calculating pipeline scores on the frozen held-out set, with the same
scorer every other system in this project was measured by, and with every stated
value independently re-checked against its source span.

**Does not establish:** a ship decision. The scorer is not yet validated against
the owner's 30 hand labels, and the EXPERIMENT_RULES_v2 section 1 thresholds are
still unset. A span-rule violation blocks ship regardless of the scores.

**Does not establish:** anything about Aivora by comparison. Aivora's weights are
not in this repository, so it cannot run here; the paired comparison on identical
items is done on the development machine.
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
        "title": "Aivora Phase4 Tool Pipeline",
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
