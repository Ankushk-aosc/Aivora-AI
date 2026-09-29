"""Phase 14: a small financial SFT experiment, then a model-only evaluation.

Deliberately small. The brief's instruction is to run a small experiment rather
than launch a multi-billion-token run, and this dataset is 522k tokens, so it
fits on a CPU in a couple of hours and costs no GPU quota.

Trains from the frozen baseline using arm 2's format - one template, prompt
masked, EOS trained - because the tiny-overfit gate showed that is what lets the
model learn to answer and stop.

Everything Phase 14 asks to record is written to reports/SFT_EXPERIMENT_001.json:
dataset version and hash, example and token counts, optimizer, learning rate,
batch, steps, validation loss per evaluation, and the resulting checkpoint's
sha256.

    python scripts/run_sft_experiment.py --steps 1000
    python scripts/run_sft_experiment.py --steps 1000 --skip-eval
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

DATA = os.path.join("data", "sft", "financial_sft_v1.jsonl")
BASELINE = os.path.join("checkpoints", "final", "checkpoint_247850.pt")
OUT_DIR = os.path.join("checkpoints", "sft_001")
REPORT = os.path.join("reports", "SFT_EXPERIMENT_001.json")
SEED = 1234
VAL_FRACTION = 0.05


def sha256_file(path, chunk=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def build(block_size):
    """Tokenize with the template, mask the prompt, train on the EOS.

    Note on the validation loss this produces: it is NOT comparable to the
    pretraining validation loss (4.37). Only answer tokens contribute, the
    prompt is ignored, and the distribution is instruction data - a different
    quantity that happens to share a name."""
    from data_sources.tokenizer import get_encoding
    from data_sources.training_format import IGNORE_INDEX, PROMPT_TEMPLATE

    enc = get_encoding()
    with open(DATA, encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]

    train, val, skipped = [], [], 0
    for row in rows:
        question = row["question"]
        if row.get("context"):
            question = f"{row['context']}\n\n{question}"
        prompt_ids = enc.encode_ordinary(PROMPT_TEMPLATE.format(prompt=question))
        answer_ids = enc.encode_ordinary(" " + row["answer"]) + [enc.eot_token]
        if len(prompt_ids) + len(answer_ids) > block_size:
            # Truncating a prompt would change the question; drop instead.
            skipped += 1
            continue
        ids = prompt_ids + answer_ids
        labels = [IGNORE_INDEX] * len(prompt_ids) + answer_ids
        # Deterministic split on the row id, so the validation set is stable
        # across runs and never mixes with training.
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


def validation_loss(model, val, device, block_size, batch_size, limit=15):
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--block-size", type=int, default=192)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--patience", type=int, default=3,
                        help="evaluations without improvement before stopping")
    parser.add_argument("--skip-eval", action="store_true")
    args = parser.parse_args()

    import torch

    from inference import load_model_for_inference

    torch.manual_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train, val, skipped, enc, rows = build(args.block_size)
    tokens = sum(len(ids) for ids, _, _ in train)
    print(f"{len(train)} train / {len(val)} val examples, {skipped} too long to fit "
          f"{args.block_size} tokens")
    print(f"{tokens:,} training tokens, device {device}")

    model, _ = load_model_for_inference(BASELINE, device=device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95),
                                  weight_decay=0.01, eps=1e-9)
    rng = torch.Generator().manual_seed(SEED)
    warmup = max(20, args.steps // 20)

    history, best, best_step, since_best = [], float("inf"), 0, 0
    os.makedirs(OUT_DIR, exist_ok=True)
    best_path = os.path.join(OUT_DIR, "sft_001_best.pt")
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
            history.append({"step": step + 1, "train_loss": round(loss.item(), 4),
                            "val_loss": round(val_loss, 4), "lr": lr,
                            "minutes": round((time.time() - started) / 60, 1)})
            improved = val_loss < best - 1e-4
            print(f"  step {step + 1:>5}  train {loss.item():.4f}  val {val_loss:.4f}"
                  f"{'  *' if improved else ''}  "
                  f"{(time.time() - started) / 60:.1f} min")
            if improved:
                best, best_step, since_best = val_loss, step + 1, 0
                torch.save({"model_state_dict": model.state_dict(),
                            "step": step + 1, "best_val_loss": best,
                            "dataset_sha256": sha256_file(DATA)}, best_path)
            else:
                since_best += 1
                if since_best >= args.patience:
                    print(f"  early stop: {args.patience} evaluations without "
                          f"improvement (best {best:.4f} at step {best_step})")
                    break

    # Always evaluate the BEST checkpoint, not the last one.
    if os.path.exists(best_path):
        state = torch.load(best_path, map_location=device)
        model.load_state_dict(state["model_state_dict"])
        print(f"reloaded best checkpoint from step {state['step']} "
              f"(val {state['best_val_loss']:.4f})")

    report = {
        "experiment": "SFT_EXPERIMENT_001",
        "dataset": {"path": DATA, "version": "financial_sft_v1",
                    "sha256": sha256_file(DATA),
                    "examples_total": len(rows),
                    "examples_train": len(train), "examples_val": len(val),
                    "examples_skipped_too_long": skipped,
                    "training_tokens": tokens},
        "base_checkpoint": {"path": BASELINE, "sha256": sha256_file(BASELINE)},
        "format": {"template": "Question: {prompt}\\nAnswer:", "prompt_masked": True,
                   "eos_trained": True},
        "optimizer": {"name": "AdamW", "betas": [0.9, 0.95], "weight_decay": 0.01,
                      "eps": 1e-9},
        "schedule": {"peak_lr": args.lr, "warmup_steps": warmup,
                     "decay": "cosine to 10% of peak"},
        "batch_size": args.batch_size, "block_size": args.block_size,
        "steps_requested": args.steps, "steps_run": history[-1]["step"] if history else 0,
        "seed": SEED, "device": device,
        "history": history,
        "best_val_loss": round(best, 4) if history else None,
        "best_step": best_step,
        "minutes": round((time.time() - started) / 60, 1),
        "checkpoint": best_path if os.path.exists(best_path) else None,
        "checkpoint_sha256": sha256_file(best_path) if os.path.exists(best_path) else None,
    }

    if not args.skip_eval:
        from app.backend.services.chat_service import FinancialChat
        from evaluation.benchmark import load_benchmark, score_item, summarise

        chat = FinancialChat(model=model, device=device, max_new_tokens=64,
                             use_calculator=False, use_knowledge=False,
                             use_guard=False)
        report["model_only"] = {}
        for split, limit in (("blind", None), ("dev", 200)):
            items = load_benchmark(split)
            if limit:
                items = items[:limit]
            records = []
            for index, item in enumerate(items, 1):
                torch.manual_seed(SEED)
                records.append(score_item(item, chat.ask(item["question"]).answer))
                if index % 50 == 0:
                    print(f"    {split} {index}/{len(items)}")
            summary = summarise(records)
            report["model_only"][split] = summary
            print(f"  {split}: {summary['overall']}  "
                  f"hallucination {summary['hallucination_rate']}%  "
                  f"abstention {summary['abstention_accuracy']}")

    os.makedirs("reports", exist_ok=True)
    with open(REPORT, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(f"\nwrote {REPORT}")


if __name__ == "__main__":
    main()
