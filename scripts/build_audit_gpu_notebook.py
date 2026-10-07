"""Kaggle notebook for audit Tasks 4 and 5 — both need the 1.5B model.

Task 4: the same 1.5B instruct model on the same 297 scored items, with the same
scorer, prompts and decoding, but WITHOUT the tool pipeline — the model answers
directly. This is the comparison the pipeline's 275/297 has never been measured
against.

Task 5: the pipeline on the fresh 44-item calculation split, which the operand
vocabulary has never seen. The frozen calculation figure is 100% and the
captions that produced it were written after the frozen failures were inspected,
so that figure is not a clean held-out result. This is.

Nothing is tuned for either run. Same model, same decoding, same scorer.

    python scripts/build_audit_gpu_notebook.py
    cd training/kaggle/audit_gpu && kaggle kernels push -p .
"""

import json
import os

OUT_DIR = os.path.join("training", "kaggle", "audit_gpu")
NOTEBOOK = "Aivora_Audit_GPU.ipynb"
KERNEL_ID = "aoscjkjhh/aivora-audit-gpu"
BRANCH = "audit/nivora-v1.1"

CELLS = []


def md(text):
    CELLS.append({"cell_type": "markdown", "metadata": {},
                  "source": text.splitlines(True)})


def code(text):
    CELLS.append({"cell_type": "code", "execution_count": None, "metadata": {},
                  "outputs": [], "source": text.splitlines(True)})


md("""# Aivora audit — Tasks 4 and 5

Two measurements that need the model, run in one session.

**Task 4 — baseline without the pipeline.** The same Qwen2.5-1.5B-Instruct, the
same 297 scored items, the same scorer, the same prompts and decoding, but
answering directly instead of through the tool pipeline. The pipeline scores
275/297; what the model scores unaided has never been measured on this split,
so the pipeline's contribution has never actually been isolated.

**Task 5 — the fresh calculation split.** The frozen calculation score is 100%,
and the 147 captions that produced it were committed *after* the eight frozen
failures were inspected. That figure is therefore not a clean held-out result.
`data/eval_fresh/calculation_fresh.jsonl` holds 44 items the vocabulary has
never seen, with expected answers computed in the builder rather than by the
pipeline.

Nothing is tuned for either. Settings: Accelerator → GPU T4, Internet → On.
""")

code(f'''import subprocess, os, json, time

REPO_URL = "https://github.com/Ankushk-aosc/Aivora-AI.git"
REPO_DIR = "/kaggle/working/Aivora-AI"
BRANCH = "{BRANCH}"

if not os.path.exists(REPO_DIR):
    r = subprocess.run(["git", "clone", "--depth", "1", "--branch", BRANCH,
                        REPO_URL, REPO_DIR], capture_output=True, text=True)
    print(r.stdout[-600:], r.stderr[-600:])
    if r.returncode != 0:
        raise RuntimeError("STATUS = BLOCKED: clone failed")
os.chdir(REPO_DIR)
import sys
sys.path.insert(0, REPO_DIR)

head = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                      capture_output=True, text=True).stdout.strip()
print("repo at", head, "on", BRANCH)

import torch
if not torch.cuda.is_available():
    raise RuntimeError("STATUS = BLOCKED: no GPU")
print(torch.cuda.get_device_properties(0).name)

for path in ("data/eval_fresh/calculation_fresh.jsonl",
             "scripts/eval_base_models.py",
             "scripts/run_tool_pipeline_eval.py"):
    if not os.path.exists(path):
        raise RuntimeError(f"STATUS = BLOCKED: {{path}} missing")
print("inputs present")
''')

md("""## Integrity: the frozen set must be the frozen set

Task 4 scores the frozen split, so its content hash is checked before the model
is loaded.""")

code('''from scripts.eval_heldout import verify_split, FROZEN
digest, status = verify_split(FROZEN, "frozen")
print(f"frozen: {status}, content_sha256 {digest[:16]}...")

import json
fresh = [json.loads(l) for l in open("data/eval_fresh/calculation_fresh.jsonl")
         if l.strip()]
print(f"fresh split: {len(fresh)} items (no manifest - not a frozen split)")
''')

md("""## Task 4 — the 1.5B model alone, on the frozen split

`scripts/eval_base_models.py` is the Phase 3 harness: one system prompt, greedy
decoding, the shared scorer. Using it unchanged is what makes this comparable to
the pipeline run by construction rather than by assertion.""")

code('''MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

started = time.time()
proc = subprocess.run(["python", "scripts/eval_base_models.py",
                       "--model", MODEL, "--shots", "0",
                       "--split", "frozen", "--force"],
                      capture_output=True, text=True)
print(proc.stdout[-3000:])
if proc.returncode != 0:
    print("FAILED:", proc.stderr[-2000:])
    task4 = None
else:
    slug = MODEL.replace("/", "_") + "_0shot_frozen"
    with open(f"reports/heldout/{slug}.json") as handle:
        task4 = json.load(handle)
    print(f"  {time.time() - started:.0f}s")
''')

md("""## Task 3b — the pipeline on the FROZEN split, with the current code

The change under test reads a stated figure instead of recomputing it. It
affects the pipeline on the frozen split, which the first version of this
notebook never re-ran - so the change was committed and described as measured
when it was not. This runs it.

Baseline to beat: 275/297 overall, extraction 60/69.""")

code('''started = time.time()
proc = subprocess.run(["python", "scripts/run_tool_pipeline_eval.py",
                       "--model", MODEL, "--split", "frozen",
                       "--tag", "stated", "--force"],
                      capture_output=True, text=True)
print(proc.stdout[-3000:])
if proc.returncode != 0:
    print("FAILED:", proc.stderr[-2000:])
    task3b = None
else:
    with open("reports/heldout/"
              f"{MODEL.replace('/', '_')}_pipeline_frozen_stated.json") as handle:
        task3b = json.load(handle)
    print(f"  {time.time() - started:.0f}s")
''')

md("""## Task 5 — the pipeline on the fresh split""")

code('''started = time.time()
proc = subprocess.run(["python", "scripts/run_tool_pipeline_eval.py",
                       "--model", MODEL,
                       "--items-file", "data/eval_fresh/calculation_fresh.jsonl",
                       "--tag", "fresh", "--force"],
                      capture_output=True, text=True)
print(proc.stdout[-3000:])
if proc.returncode != 0:
    print("FAILED:", proc.stderr[-2000:])
    task5 = None
else:
    import glob
    hits = glob.glob("reports/heldout/*fresh*.json")
    with open(sorted(hits)[-1]) as handle:
        task5 = json.load(handle)
    print(f"  {time.time() - started:.0f}s")
''')

md("""## Task 5b — captions the vocabulary does NOT have

44 items whose every operand caption is verified absent from `SYNONYMS`. When
resolution fails the pipeline asks the model for the canonical operand name
against an unfamiliar caption, so this measures whether the **model** bridges
the gap the vocabulary does not cover. The outcome to watch is wrong values
rather than refusals: a span can be valid while the line it cites is the wrong
one.""")

code('''started = time.time()
proc = subprocess.run(["python", "scripts/run_tool_pipeline_eval.py",
                       "--model", MODEL,
                       "--items-file", "data/eval_fresh/calculation_uncovered.jsonl",
                       "--tag", "uncovered", "--force"],
                      capture_output=True, text=True)
print(proc.stdout[-3000:])
if proc.returncode != 0:
    print("FAILED:", proc.stderr[-2000:])
    task5b = None
else:
    import glob
    hits = glob.glob("reports/heldout/*uncovered*.json")
    with open(sorted(hits)[-1]) as handle:
        task5b = json.load(handle)
    print(f"  {time.time() - started:.0f}s")
''')

md("""## Results""")

code('''out = {"repo_commit": head, "branch": BRANCH, "model": MODEL}

print("=" * 72)
print("TASK 3b - pipeline on frozen, reading stated figures")
print("=" * 72)
if task3b:
    s3 = task3b["summary"]
    out["task3b_stated_preference"] = s3
    o3 = s3["overall"]
    before = {"copy": (68, 69), "extraction": (60, 69), "wording": (64, 75),
              "calculation": (48, 48), "abstention": (35, 36)}
    print(f"{'gate':<14}{'before':>10}{'after':>10}{'delta':>8}")
    for gate, (bk, bn) in before.items():
        st = s3["by_gate"].get(gate, {})
        ak = st.get("k", 0)
        print(f"{gate:<14}{bk:>5}/{bn:<4}{ak:>5}/{st.get('n', bn):<4}{ak - bk:>+8}")
    print(f"{'OVERALL':<14}{275:>5}/297 {o3['k']:>5}/{o3['n']:<4}"
          f"{o3['k'] - 275:>+8}")
    print(f"invented values {s3['hallucination_invented_values']['k']}"
          f"/{s3['hallucination_invented_values']['n']}")
    print(f"span rule {s3['span_audit']['span_validated_pct']}% of "
          f"{s3['span_audit']['answers_checked']}, "
          f"{s3['span_audit']['violation_count']} violations")
else:
    print("FAILED")

print()
print("=" * 72)
print("TASK 4 - the model WITHOUT the pipeline, frozen split")
print("=" * 72)
if task4:
    s = task4["summary"]
    out["task4_baseline_no_pipeline"] = s
    print(f"{'gate':<14}{'k/n':>10}{'acc':>9}{'95% CI':>20}")
    for gate, st in s["by_gate"].items():
        print(f"{gate:<14}{st['k']:>4}/{st['n']:<5}{st['accuracy_pct']:>8.1f}%"
              f"   [{st['ci95'][0]:.1f}, {st['ci95'][1]:.1f}]")
    o = s["overall"]
    print(f"{'OVERALL':<14}{o['k']:>4}/{o['n']:<5}{o['accuracy_pct']:>8.2f}%"
          f"   [{o['ci95'][0]:.1f}, {o['ci95'][1]:.1f}]")
    print(f"invented values {s['hallucination_invented_values']['k']}/"
          f"{s['hallucination_invented_values']['n']} "
          f"({s['hallucination_invented_values']['pct']}%)")
    print()
    print("against the pipeline on the same items: 275/297 = 92.59%, "
          "invented 0/261")
    delta = o["accuracy_pct"] - 92.59
    print(f"pipeline advantage: {-delta:+.2f} percentage points")
else:
    print("FAILED")

print()
print("=" * 72)
print("TASK 5 - the pipeline on the FRESH calculation split")
print("=" * 72)
if task5:
    s = task5["summary"]
    out["task5_fresh_split"] = s
    o = s["overall"]
    print(f"fresh calculation: {o['k']}/{o['n']} = {o['accuracy_pct']:.1f}%"
          f"   [{o['ci95'][0]:.1f}, {o['ci95'][1]:.1f}]")
    print(f"frozen calculation (vocabulary written after seeing its failures): "
          f"48/48 = 100.0%")
    print(f"span rule: {s['span_audit']['span_validated_pct']}% of "
          f"{s['span_audit']['answers_checked']}, "
          f"{s['span_audit']['violation_count']} violations")
    print(f"abstained: {s['abstained_pct']}%")
    gap = 100.0 - o["accuracy_pct"]
    print()
    print(f"gap between the contaminated and clean measurements: {gap:.1f} points")
    if gap > 10:
        print("READING: the frozen 100% substantially overstates this capability.")
    elif gap > 3:
        print("READING: the frozen 100% modestly overstates this capability.")
    else:
        print("READING: the capability holds on captions the vocabulary has "
              "never seen.")
else:
    print("FAILED")

print()
print("=" * 72)
print("TASK 5b - the pipeline on UNCOVERED captions")
print("=" * 72)
if task5b:
    s = task5b["summary"]
    out["task5b_uncovered_captions"] = s
    o = s["overall"]
    cats = s["by_gate"].get("calculation", {}).get("categories", {})
    wrong = cats.get("wrong_value", 0)
    refused = cats.get("incorrect_abstention", 0)
    print(f"accuracy   {o['k']}/{o['n']} = {o['accuracy_pct']:.1f}%")
    print(f"abstained  {s['abstained_pct']}%")
    print(f"span rule  {s['span_audit']['span_validated_pct']}% of "
          f"{s['span_audit']['answers_checked']}, "
          f"{s['span_audit']['violation_count']} violations")
    print(f"failure split: refused {refused}, WRONG VALUES {wrong}")
    print()
    print("  covered captions  (first fresh split): 44/44 = 100.0%")
    print(f"  uncovered captions (this split)      : {o['k']}/{o['n']} = "
          f"{o['accuracy_pct']:.1f}%")
    if wrong > 0:
        print(f"\\n  WATCH: {wrong} wrong values returned rather than refusals. "
              f"A valid span does not mean the right line.")
    elif o["accuracy_pct"] > 80:
        print("\\n  READING: the model bridges the gap unaided. The caption "
              "table is doing less work than its size suggests.")
    else:
        print("\\n  READING: the vocabulary is load-bearing, and uncovered "
              "captions degrade to refusal rather than error - the safe "
              "failure.")
else:
    print("FAILED")

with open("/kaggle/working/audit_gpu_results.json", "w") as handle:
    json.dump(out, handle, indent=2)
print("\\nwrote /kaggle/working/audit_gpu_results.json")
''')

md("""## Copy the raw outputs out

Task 4 requires raw model outputs saved. Both per-item files go to the working
directory.""")

code('''import shutil, glob
for src in glob.glob("reports/heldout/*frozen*.json") + \\
           glob.glob("reports/heldout/*fresh*.json") + \
           glob.glob("reports/heldout/*uncovered*.json") + \
           glob.glob("reports/heldout/*stated*.json"):
    if "Qwen" in src:
        dst = "/kaggle/working/" + os.path.basename(src)
        shutil.copy(src, dst)
        print(f"{os.path.getsize(dst)/1024:>8.0f} KB  {dst}")

# Task 4 also wants the raw outputs as JSONL.
slug = MODEL.replace("/", "_") + "_0shot_frozen"
path = f"reports/heldout/{slug}.json"
if os.path.exists(path):
    with open(path) as handle:
        recs = json.load(handle)["records"]
    with open("/kaggle/working/04_baseline_raw.jsonl", "w") as handle:
        for r in recs:
            handle.write(json.dumps(r) + "\\n")
    print(f"wrote /kaggle/working/04_baseline_raw.jsonl ({len(recs)} records)")
''')


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    notebook = {"cells": CELLS,
                "metadata": {"kernelspec": {"display_name": "Python 3",
                                            "language": "python",
                                            "name": "python3"},
                             "language_info": {"name": "python",
                                               "version": "3.11"}},
                "nbformat": 4, "nbformat_minor": 5}
    path = os.path.join(OUT_DIR, NOTEBOOK)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(notebook, handle, indent=1)
    with open(os.path.join(OUT_DIR, "kernel-metadata.json"), "w",
              encoding="utf-8") as handle:
        json.dump({"id": KERNEL_ID, "title": "Aivora Audit GPU",
                   "code_file": NOTEBOOK, "language": "python",
                   "kernel_type": "notebook", "is_private": "true",
                   "enable_gpu": "true", "enable_tpu": "false",
                   "enable_internet": "true", "dataset_sources": [],
                   "competition_sources": [], "kernel_sources": [],
                   "model_sources": []}, handle, indent=2)
    print(f"wrote {path} ({len(CELLS)} cells)")


if __name__ == "__main__":
    main()
