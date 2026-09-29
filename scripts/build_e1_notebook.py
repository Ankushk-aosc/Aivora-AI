"""Generate the E1 Kaggle notebook.

Written as a generator rather than a hand-edited .ipynb because the two arms
must run byte-identical code - the only difference is the ARM constant in the
first cell. Editing two notebooks by hand is how arms end up differing in some
way nobody recorded.

    python scripts/build_e1_notebook.py
"""

import json
import os

import argparse

_parser = argparse.ArgumentParser()
_parser.add_argument("--arm", type=int, default=1, choices=(1, 2))
_ARGS, _ = _parser.parse_known_args()
ARM_VALUE = _ARGS.arm

# Arm 1 was pushed as "e1" / aivora-e1-format-experiment before this was
# parameterised; arm 2 gets its own directory and kernel id so neither arm can
# overwrite the other's outputs on Kaggle.
if ARM_VALUE == 1:
    OUT_DIR = os.path.join("training", "kaggle", "e1")
    KERNEL_ID = "aoscjkjhh/aivora-e1-format-experiment"
    KERNEL_TITLE = "Aivora E1 Format Experiment"
else:
    OUT_DIR = os.path.join("training", "kaggle", f"e1_arm{ARM_VALUE}")
    KERNEL_ID = f"aoscjkjhh/aivora-e1-arm{ARM_VALUE}"
    KERNEL_TITLE = f"Aivora E1 Arm {ARM_VALUE}"

NOTEBOOK = f"Aivora_E1_Arm{ARM_VALUE}.ipynb"

CELLS = []


def md(text):
    CELLS.append({"cell_type": "markdown", "metadata": {}, "source": text.splitlines(True)})


def code(text):
    CELLS.append({"cell_type": "code", "execution_count": None, "metadata": {},
                  "outputs": [], "source": text.splitlines(True)})


md("""# Aivora E1 - format-controlled comparison

Tests whether Aivora is weak because it was under-fed (data) or because it was
trained to continue text rather than to answer and stop (format).

Two arms, **identical in everything except three data-format flags**:

| | arm 1 (control) | arm 2 (treated) |
| --- | --- | --- |
| `<\\|endoftext\\|>` between documents | no | **yes** |
| prompt template `Question: ...\\nAnswer:` | no | **yes** |
| prompt masking (loss on the answer only) | no | **yes** |

Same corpus, same budget, same seed, same architecture, fresh start both times.

**Set `ARM` in the next cell, then Run All.** Run arm 1 in one session and arm 2
in another; each is ~8 hours including data preparation, inside Kaggle's 12-hour
cap and the 30 GPU-hours/week quota.

Before running: Accelerator -> GPU, Internet -> On.

Outcomes and what each would mean are in `docs/AIVORA_TRAINING_DECISION.md`.
Nothing here is simulated: every cell does the real thing or raises
`RuntimeError('STATUS = BLOCKED: ...')` naming what is missing.
""")

code('''# ============================================================
# THE ONLY LINE THAT DIFFERS BETWEEN THE TWO ARMS
ARM = __ARM__          # 1 = control (old format), 2 = treated (all three fixes)
# ============================================================

SEED = 1234      # identical for both arms
PRESET = "e1_format"
print(f"E1 arm {ARM}")
''')

md("## 1. GPU gate\n\nRaises rather than silently training on CPU for eight hours.")

code('''import subprocess

smi = subprocess.run(["nvidia-smi", "--query-gpu=name,compute_cap,memory.total",
                      "--format=csv,noheader"], capture_output=True, text=True)
print("nvidia-smi:", smi.stdout.strip() or smi.stderr.strip())

NEEDS_OLDER_TORCH = False
if smi.returncode == 0 and smi.stdout.strip():
    line = smi.stdout.strip().splitlines()[0]
    try:
        compute_cap = float(line.split(",")[1])
        # This account is often assigned a P100 (sm_60); the base image's torch
        # only supports 7.0+, and every CUDA kernel launch fails without an
        # older build. Same check as the main training notebook.
        NEEDS_OLDER_TORCH = compute_cap < 7.0
    except (IndexError, ValueError):
        pass
print("needs older torch:", NEEDS_OLDER_TORCH)
if smi.returncode != 0:
    raise RuntimeError("STATUS = BLOCKED: no GPU attached. Settings -> Accelerator -> GPU.")
''')

code('''if NEEDS_OLDER_TORCH:
    subprocess.run(["pip", "install", "-q", "torch==2.1.2", "--index-url",
                    "https://download.pytorch.org/whl/cu118"], check=False)

import torch
print("torch", torch.__version__, "| cuda", torch.cuda.is_available(),
      "|", torch.cuda.device_count(), "device(s)")
if not torch.cuda.is_available():
    raise RuntimeError("STATUS = BLOCKED: torch cannot see the GPU.")
for i in range(torch.cuda.device_count()):
    print(" ", torch.cuda.get_device_name(i),
          f"{torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB")
''')

md("## 2. Repository")

code('''import os, subprocess

REPO_URL = "https://github.com/Ankushk-aosc/Aivora-AI.git"
REPO_DIR = "/kaggle/working/Aivora-AI"

if not os.path.exists(REPO_DIR):
    r = subprocess.run(["git", "clone", "--depth", "1", REPO_URL, REPO_DIR],
                       capture_output=True, text=True)
    print(r.stdout, r.stderr)
    if r.returncode != 0:
        raise RuntimeError("STATUS = BLOCKED: git clone failed. Internet -> On?")
os.chdir(REPO_DIR)

for path in ("data_sources/training_format.py", "configs/e1_format.yaml",
             "training/trainer.py", "tests/test_training_format.py"):
    if not os.path.exists(path):
        raise RuntimeError(f"STATUS = BLOCKED: {path} missing - the clone is stale. "
                           "Push the E1 commit before running this notebook.")
print("repo ready:", subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                                    capture_output=True, text=True).stdout.strip())
''')

md("""## 3. Format tests before training

Eight hours of GPU time depends on the mask lining up with the prompt after the
label shift. A misaligned mask trains on the wrong tokens and nothing visibly
fails, so the tests run first and the notebook stops if any of them does.""")

code('''r = subprocess.run(["python", "tests/test_training_format.py"],
                   capture_output=True, text=True)
print(r.stdout[-2500:])
if r.returncode != 0:
    raise RuntimeError("STATUS = BLOCKED: format tests failed - do not spend GPU time "
                       "on a pipeline that cannot be trusted.")
''')

md("## 4. The arm's format")

code('''from data_sources.training_format import TrainingFormat

if ARM == 1:
    fmt = TrainingFormat()                      # exactly the old pipeline
elif ARM == 2:
    fmt = TrainingFormat(document_separator=True, template=True, mask_prompt=True)
else:
    raise ValueError("ARM must be 1 or 2")

print(json.dumps(fmt.describe(), indent=2) if (json := __import__("json")) else None)
''')

md("""## 5. Data preparation

Same budget and same seed for both arms, so the corpora differ only by the
format flags. ~121M unique tokens, which is what the completed run used.""")

code('''import time, yaml
from data_sources.dataset_mixer import validate_mix
from data_sources.dataset_registry import list_entries
from data_sources.prepare import prepare_dataset

with open(f"configs/{PRESET}.yaml") as f:
    preset = yaml.safe_load(f)

mix = preset["dataset_mix"]
validate_mix(mix)
total_budget = int(preset["train_tokens"]) + int(preset["validation_tokens"])
print(f"corpus budget: {total_budget:,} tokens, seed {SEED}")

started = time.time()
prepared = {}
entries = list_entries()
from data_sources.dataset_mixer import BUCKET_TO_CATEGORY

for bucket, weight in mix.items():
    category = BUCKET_TO_CATEGORY.get(bucket, bucket)
    names = [e.name for e in entries if e.category == category]
    if not names:
        print(f"  {bucket}: no datasets registered, skipping")
        continue
    per_dataset = int(total_budget * weight / len(names))
    for name in names:
        try:
            stats = prepare_dataset(name, max_tokens=per_dataset, seed=SEED,
                                    training_format=fmt)
            prepared[name] = stats
            print(f"  {name}: {stats.get('train_tokens_used', 0):,} train tokens")
        except Exception as e:
            print(f"  {name}: FAILED {type(e).__name__}: {e}")

print(f"\\nprepared in {(time.time() - started) / 60:.1f} min")
''')

code('''# What the arm actually produced: separators present or absent, tokens masked
# or not. Recorded so the two runs can be compared on evidence, not intent.
import glob, numpy as np
from data_sources.shard_writer import load_shard_index
from data_sources.tokenizer import get_encoding

enc = get_encoding()
separators = masked = total = 0
for index_path in glob.glob("data/shards/*/train/index.json"):
    shard_dir = os.path.dirname(index_path)
    index = load_shard_index(shard_dir)
    for shard in index["shards"]:
        tokens = np.fromfile(os.path.join(shard_dir, shard["file"]), dtype=np.uint16)
        separators += int((tokens == enc.eot_token).sum())
        total += len(tokens)
        if shard.get("mask_file"):
            mask = np.fromfile(os.path.join(shard_dir, shard["mask_file"]), dtype=np.uint8)
            masked += int((mask == 0).sum())

CORPUS_FACTS = {"arm": ARM, "total_train_tokens": total,
                "separator_tokens": separators, "masked_tokens": masked,
                "format": fmt.describe()}
print(json.dumps(CORPUS_FACTS, indent=2))

if ARM == 2 and separators == 0:
    raise RuntimeError("STATUS = BLOCKED: arm 2 produced no separators - the format "
                       "flag did not reach the writer.")
if ARM == 1 and separators > 0:
    raise RuntimeError("STATUS = BLOCKED: arm 1 contains separators - it is not the "
                       "control any more.")
''')

md("""## 6. Train

`max_steps` is computed from the token-pass target and the ACTUAL tokens per
step, because that depends on how many GPUs Kaggle assigned. Hard-coding it is
what made the original run's budget ambiguous.""")

code('''import torch

gpus = max(1, torch.cuda.device_count())
batch_size = int(preset["batch_size"]) * gpus     # DataParallel splits the batch
tokens_per_step = batch_size * int(preset["seq_len"])
max_steps = int(preset["train_token_passes"]) // tokens_per_step

print(f"{gpus} GPU(s), batch {batch_size}, {tokens_per_step:,} tokens/step")
print(f"max_steps {max_steps:,} for {int(preset['train_token_passes']):,} token passes")
print(f"epochs over the corpus: {int(preset['train_token_passes']) / max(total, 1):.2f}")
''')

code('''import time

from training.trainer import train_model

started = time.time()

model, config, checkpoint_path = train_model(
    preset_name=PRESET,
    seed=SEED,
    batch_size_override=batch_size,
    max_steps_override=max_steps,
    # The cosine horizon must be the ACTUAL step count. Left at the config's
    # value it would anneal against a different number on a one-GPU session,
    # and the arms would then differ by more than the format flags.
    lr_horizon_override=max_steps,
    checkpoints_dir=f"/kaggle/working/e1_arm{ARM}",
    max_train_seconds=float(preset["max_train_seconds"]),
    keep_last_checkpoints=int(preset["keep_last_checkpoints"]),
    divergence_val_threshold=float(preset["divergence_val_threshold"]),
)

print(f"\\ntrained in {(time.time() - started) / 3600:.2f} h")
print("checkpoint:", checkpoint_path)
''')

md("""## 7. Did arm 2 learn to stop?

The single clearest signal, and the cheapest to check. The baseline checkpoint
emitted `<\\|endoftext\\|>` in 0 of 204 generations because the token was never in
its training data. If arm 2 now emits it and arm 1 does not, the format change
did what it was supposed to.""")

code('''import torch
from data_sources.tokenizer import get_encoding

enc = get_encoding()
device = "cuda" if torch.cuda.is_available() else "cpu"
model.eval()

PROBES = ["What is EBITDA?", "What is revenue?", "What is a balance sheet?",
          "What is working capital?", "What is net income?", "What is ROE?",
          "What is free cash flow?", "What is gross margin?",
          "What is operating income?", "What is debt-to-equity?"]

eos_hits, lengths, samples = 0, [], []
for question in PROBES:
    prompt = f"Question: {question}\\nAnswer:"
    ids = enc.encode_ordinary(prompt)
    torch.manual_seed(SEED)
    with torch.no_grad():
        out = model.generate(torch.tensor(ids, device=device).unsqueeze(0), 128,
                             temperature=0.7, top_k=40, top_p=0.9,
                             repetition_penalty=1.3, stop_on_repetition=True,
                             eos_token_id=enc.eot_token)
    generated = out[0, len(ids):].tolist()
    if enc.eot_token in generated:
        eos_hits += 1
        generated = generated[:generated.index(enc.eot_token)]
    lengths.append(len(generated))
    samples.append({"question": question, "answer": enc.decode(generated).strip()[:200]})

EOS_RESULT = {"arm": ARM, "probes": len(PROBES), "eos_emitted": eos_hits,
              "eos_rate_pct": round(100.0 * eos_hits / len(PROBES), 1),
              "mean_generated_tokens": round(sum(lengths) / len(lengths), 1)}
print(json.dumps(EOS_RESULT, indent=2))
for s in samples[:5]:
    print(f"\\nQ: {s['question']}\\nA: {s['answer']}")
''')

md("""## 8. Model-only accuracy

The same benchmarks and the same scorer used for the baseline, with every
deterministic layer disabled. `blind` is the authored prose set; `dev` is used
here only as a relative measure between the two arms, never to tune anything.""")

code('''from app.backend.services.chat_service import FinancialChat
from evaluation.benchmark import load_benchmark, score_item, summarise

chat = FinancialChat(model=model, device=device, max_new_tokens=64,
                     use_calculator=False, use_knowledge=False, use_guard=False)

ACCURACY = {}
for split, limit in (("blind", None), ("dev", 200)):
    items = load_benchmark(split)
    if limit:
        items = items[:limit]
    records = []
    for i, item in enumerate(items, 1):
        torch.manual_seed(SEED)
        records.append(score_item(item, chat.ask(item["question"]).answer))
        if i % 50 == 0:
            print(f"  {split} {i}/{len(items)}")
    ACCURACY[split] = summarise(records)
    print(f"{split}: {ACCURACY[split]['overall']}  "
          f"hallucination {ACCURACY[split]['hallucination_rate']}%")
''')

md("## 9. Save the result")

code('''import json, shutil

RESULT = {
    "arm": ARM,
    "seed": SEED,
    "format": fmt.describe(),
    "corpus": CORPUS_FACTS,
    "steps": max_steps,
    "tokens_per_step": tokens_per_step,
    "token_passes": int(preset["train_token_passes"]),
    "checkpoint": checkpoint_path,
    "eos_probe": EOS_RESULT,
    "accuracy": ACCURACY,
    "baseline_for_comparison": {
        "checkpoint": "checkpoint_247850",
        "model_only_dev": 1.64, "model_only_blind": 7.64,
        "eos_rate_pct": 0.0,
        "note": "from AIVORA_PRE_TRAINING_BASELINE; different corpus SIZE is not "
                "being varied here, only the format",
    },
}
out = f"/kaggle/working/e1_arm{ARM}_result.json"
with open(out, "w") as f:
    json.dump(RESULT, f, indent=2)
print(json.dumps(RESULT, indent=2)[:2000])
print("\\nwrote", out)
print("\\nDownload this file and e1_arm{}/ checkpoints before the session ends - "
      "Kaggle discards outputs from cancelled sessions.".format(ARM))
''')

md("""## What each outcome means

Filled in only after both arms have run. From
`docs/AIVORA_TRAINING_DECISION.md`:

* **arm 2 emits `<\\|endoftext\\|>` and accuracy moves materially** - the model was
  mis-trained rather than under-trained. Format (B/F) dominates and the
  expensive corpus programme can be scoped with a working recipe.
* **arm 2 learns to stop but accuracy does not move** - format was necessary and
  insufficient. Data (A) is the live hypothesis and the ~53-hour 2B-token run is
  justified with a clean prior.
* **arm 2 does not learn to stop** - something is wrong beyond what the audit
  found; no scaling run should start until it is understood.

A null result here is a real result. Do not rerun with different settings until
it has been written down.
""")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    notebook = {
        "cells": CELLS,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4, "nbformat_minor": 5,
    }
    path = os.path.join(OUT_DIR, NOTEBOOK)
    body = json.dumps(notebook, indent=1).replace("__ARM__", str(ARM_VALUE))
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(body)

    metadata = {
        "id": KERNEL_ID,
        "title": KERNEL_TITLE,
        "code_file": NOTEBOOK,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": "true",
        "enable_gpu": "true",
        "enable_tpu": "false",
        "enable_internet": "true",
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }
    with open(os.path.join(OUT_DIR, "kernel-metadata.json"), "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print(f"wrote {path} ({len(CELLS)} cells)")
    print(f"wrote {os.path.join(OUT_DIR, 'kernel-metadata.json')}")


if __name__ == "__main__":
    main()
