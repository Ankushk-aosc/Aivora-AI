"""SFT_002 sanity experiment: small, CPU-only, gated on capability.

Step 6 of the brief. The purpose is NOT a final model. It is to answer one
question before any GPU time is spent:

    does the revised dataset and recipe move extraction in the right direction
    WITHOUT destroying copy ability?

Differences from SFT_001, each traceable to a measured cause of its failure:

  * financial_sft_v2: extraction holds 37.6% of answer tokens instead of 10%,
    no task repeats a sentence, 1,200 copy rows and a real-text replay slice
  * peak learning rate 2e-5 instead of 5e-5
  * replay rows train WITHOUT the prompt template, as plain continuation
  * evaluation runs the capability gates, not just validation loss, and the run
    STOPS when copy ability drops below the floor - the failure SFT_001 hid

    python scripts/run_sft_002.py --steps 600
"""

import argparse
import hashlib
import json
import math
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

DATA = os.path.join("data", "sft", "financial_sft_v2.jsonl")
BASELINE = os.path.join("checkpoints", "final", "checkpoint_247850.pt")
GATES_BASELINE = os.path.join("reports", "gates_baseline.json")
OUT_DIR = os.path.join("checkpoints", "sft_002")
REPORT = os.path.join("reports", "SFT_EXPERIMENT_002.json")
SEED = 1234
VAL_FRACTION = 0.05
COPY_FLOOR_FRACTION = 0.8


def sha256_file(path, chunk=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def build(block_size):
    from data_sources.tokenizer import get_encoding
    from data_sources.training_format import IGNORE_INDEX, PROMPT_TEMPLATE

    enc = get_encoding()
    with open(DATA, encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]

    train, val, skipped = [], [], 0
    for row in rows:
        if row["task_type"] == "replay":
            # Plain continuation: no template, no mask, every token trained.
            # This is the replay slice, and templating it would defeat its point.
            ids = enc.encode_ordinary(row["answer"]) + [enc.eot_token]
            labels = list(ids)
        else:
            question = row["question"]
            if row.get("context"):
                question = f"{row['context']}\n\n{question}"
            prompt_ids = enc.encode_ordinary(PROMPT_TEMPLATE.format(prompt=question))
            answer_ids = enc.encode_ordinary(" " + row["answer"]) + [enc.eot_token]
            ids = prompt_ids + answer_ids
            labels = [IGNORE_INDEX] * len(prompt_ids) + answer_ids
        if len(ids) > block_size:
            skipped += 1
            continue
        bucket = int(hashlib.sha256(row["id"].encode()).hexdigest(), 16) % 100
        (val if bucket < VAL_FRACTION * 100 else train).append((ids, labels, row))
    return train, val, skipped, enc, rows


def collate(batch, device, pad_to):
    import torch

    from data_sources.training_format import IGNORE_INDEX

    xs, ys = [], []
    for ids, labels, _ in batch:
        pad = pad_to - len(ids)
        xs.append((ids + [0] * pad)[:pad_to][:-1])
        ys.append((labels + [IGNORE_INDEX] * pad)[:pad_to][1:])
    return (torch.tensor(xs, dtype=torch.long, device=device),
            torch.tensor(ys, dtype=torch.long, device=device))


def validation_loss(model, val, device, block_size, batch_size, limit=12):
    import torch

    model.eval()
    total, batches = 0.0, 0
    with torch.no_grad():
        for start in range(0, min(len(val), limit * batch_size), batch_size):
            batch = val[start:start + batch_size]
            if not batch:
                break
            X, Y = collate(batch, device, block_size)
            _, loss, _, _ = model(X, Y, return_logits=False)
            total += loss.mean().item()
            batches += 1
    model.train()
    return total / max(batches, 1)


def quick_gates(model, enc):
    """Copy and extraction only - the two that decide whether to continue."""
    from evaluation.financial_metrics import numeric_match

    from scripts.capability_gates import (
        COPY_PROBES, generate, generate_extraction_probes,
    )
    import random

    model.eval()
    copy_hits = 0
    for prompt, needed in COPY_PROBES:
        answer, _, _ = generate(model, enc, prompt, 24)
        copy_hits += needed.lower() in answer.lower()

    rng = random.Random(99001)
    probes = generate_extraction_probes(rng, 10)
    extraction_hits = 0
    for probe in probes:
        answer, _, _ = generate(model, enc, probe["prompt"], 32)
        extraction_hits += numeric_match(answer, probe["expected"], 0.01)
    model.train()
    return copy_hits, len(COPY_PROBES), extraction_hits, len(probes)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--block-size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--eval-every", type=int, default=100)
    args = parser.parse_args()

    import torch

    from inference import load_model_for_inference

    torch.manual_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train, val, skipped, enc, rows = build(args.block_size)
    tokens = sum(len(ids) for ids, _, _ in train)
    print(f"{len(train)} train / {len(val)} val, {skipped} too long, "
          f"{tokens:,} tokens, device {device}")

    baseline_gates = {}
    if os.path.exists(GATES_BASELINE):
        with open(GATES_BASELINE, encoding="utf-8") as handle:
            baseline_gates = json.load(handle)
    baseline_copy = baseline_gates.get("gate_a_copy", {}).get("score", 5)
    copy_floor = COPY_FLOOR_FRACTION * baseline_copy
    baseline_extraction = baseline_gates.get("gate_b_extraction", {}).get("score", 0)
    print(f"baseline copy {baseline_copy}/6 (floor {copy_floor:.1f}), "
          f"baseline extraction {baseline_extraction}/20")

    model, _ = load_model_for_inference(BASELINE, device=device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95),
                                  weight_decay=0.01, eps=1e-9)
    rng = torch.Generator().manual_seed(SEED)
    warmup = max(20, args.steps // 20)

    os.makedirs(OUT_DIR, exist_ok=True)
    best_path = os.path.join(OUT_DIR, "sft_002_best.pt")
    history, best_score, stopped_because = [], -1.0, None
    started = time.time()

    for step in range(args.steps):
        if step < warmup:
            lr = args.lr * (step + 1) / warmup
        else:
            progress = (step - warmup) / max(1, args.steps - warmup)
            lr = args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))
        for group in optimizer.param_groups:
            group["lr"] = lr

        picks = torch.randint(0, len(train), (args.batch_size,), generator=rng).tolist()
        X, Y = collate([train[i] for i in picks], device, args.block_size)
        _, loss, _, _ = model(X, Y, return_logits=False)
        loss = loss.mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if (step + 1) % args.eval_every == 0 or step == args.steps - 1:
            val_loss = validation_loss(model, val, device, args.block_size,
                                       args.batch_size)
            copy_hits, copy_of, extract_hits, extract_of = quick_gates(model, enc)
            entry = {"step": step + 1, "train_loss": round(loss.item(), 4),
                     "val_loss": round(val_loss, 4),
                     "copy": f"{copy_hits}/{copy_of}",
                     "extraction": f"{extract_hits}/{extract_of}",
                     "minutes": round((time.time() - started) / 60, 1)}
            history.append(entry)
            print(f"  step {step + 1:>4}  val {val_loss:.4f}  copy {copy_hits}/{copy_of}"
                  f"  extraction {extract_hits}/{extract_of}  "
                  f"{entry['minutes']:.0f} min")

            # The gate SFT_001 did not have: stop when copy collapses, whatever
            # the loss is doing.
            if copy_hits < copy_floor:
                stopped_because = (f"copy collapsed to {copy_hits}/{copy_of}, below "
                                   f"the floor of {copy_floor:.1f} "
                                   f"(baseline {baseline_copy}/{copy_of})")
                print(f"  STOPPING: {stopped_because}")
                break

            # Keep the checkpoint that is best on CAPABILITY, not on loss:
            # extraction first, copy retention as the tie-break.
            score = extract_hits + 0.1 * copy_hits
            if score > best_score:
                best_score = score
                torch.save({"model_state_dict": model.state_dict(),
                            "step": step + 1, "val_loss": val_loss,
                            "copy": copy_hits, "extraction": extract_hits,
                            "dataset_sha256": sha256_file(DATA)}, best_path)

    report = {
        "experiment": "SFT_EXPERIMENT_002",
        "purpose": ("sanity check before any GPU time: does the revised dataset "
                    "move extraction without destroying copy ability?"),
        "dataset": {"path": DATA, "version": "financial_sft_v2",
                    "sha256": sha256_file(DATA), "rows": len(rows),
                    "train": len(train), "val": len(val), "skipped": skipped,
                    "training_tokens": tokens},
        "base_checkpoint": {"path": BASELINE, "sha256": sha256_file(BASELINE)},
        "recipe_changes_from_001": {
            "peak_lr": f"5e-5 -> {args.lr}",
            "extraction_token_share": "10.0% -> 37.6%",
            "replay_rows": "0 -> trained without a template",
            "copy_rows": "0 -> 1200",
            "early_stopping_signal": "validation loss -> copy retention + extraction",
            "checkpoint_selection": "lowest loss -> best capability",
        },
        "optimizer": {"name": "AdamW", "betas": [0.9, 0.95], "weight_decay": 0.01},
        "schedule": {"peak_lr": args.lr, "warmup_steps": warmup,
                     "decay": "cosine to 10%"},
        "batch_size": args.batch_size, "block_size": args.block_size,
        "steps_requested": args.steps,
        "steps_run": history[-1]["step"] if history else 0,
        "seed": SEED, "device": device, "history": history,
        "stopped_because": stopped_because,
        "baseline_gates": {"copy": f"{baseline_copy}/6",
                           "extraction": f"{baseline_extraction}/20"},
        "checkpoint": best_path if os.path.exists(best_path) else None,
        "checkpoint_sha256": sha256_file(best_path) if os.path.exists(best_path) else None,
        "minutes": round((time.time() - started) / 60, 1),
    }
    os.makedirs("reports", exist_ok=True)
    with open(REPORT, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(f"\nwrote {REPORT}")
    if history:
        first, last = history[0], history[-1]
        print(f"copy {first['copy']} -> {last['copy']}, "
              f"extraction {first['extraction']} -> {last['extraction']}, "
              f"val {first['val_loss']} -> {last['val_loss']}")


if __name__ == "__main__":
    main()
