"""Phase 6: can the stack learn anything at all?

Before spending GPU hours, train on a tiny verified financial set and check that
the model substantially learns it. This is a gate on the machinery - labels,
masking, loss, optimizer, tokenizer, forward pass, output head - not a claim
about capability. If a 101.7M model cannot memorise 54 examples, nothing larger
is worth launching.

It starts from the frozen baseline checkpoint rather than random init, because
that tests the whole stack as it actually exists, and converges in minutes on a
CPU instead of hours.

    python scripts/tiny_overfit.py --steps 400
    python scripts/tiny_overfit.py --steps 400 --from-scratch

GATE (both must hold):
  * training loss falls below 0.5, or by at least 10x
  * at least 3 of the 4 probe questions are answered correctly
"""

import argparse
import json
import math
import os
import re
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

DATA = os.path.join("data", "tiny_overfit", "tiny_financial.jsonl")
CHECKPOINT = os.path.join("checkpoints", "final", "checkpoint_247850.pt")
OUT = os.path.join("reports", "tiny_overfit_result.json")
SEED = 1234

# The brief's four probes, plus what a learned answer must contain.
PROBES = [
    ("What is EBITDA?", None,
     ["earnings before interest", "amortization", "amortisation"]),
    ("Revenue is 1000 and cost is 600. What is gross margin?", 40.0, []),
    ("What is cash?", None, ["500"]),
    ("Revenue increased while margin declined. What could explain this?", None,
     ["cost", "mix", "discount", "input"]),
]
PROBE_CONTEXT = {"What is cash?": "Revenue: 1,200\nCash: 500\nEBITDA: 300"}


def build_examples(block_size):
    from data_sources.tokenizer import get_encoding
    from data_sources.training_format import IGNORE_INDEX, PROMPT_TEMPLATE

    enc = get_encoding()
    rows = [json.loads(line) for line in open(DATA, encoding="utf-8") if line.strip()]
    examples = []
    for row in rows:
        question = row["question"]
        if row.get("context"):
            question = f"{row['context']}\n\n{question}"
        prompt_ids = enc.encode_ordinary(PROMPT_TEMPLATE.format(prompt=question))
        answer_ids = enc.encode_ordinary(" " + row["answer"]) + [enc.eot_token]
        ids = (prompt_ids + answer_ids)[:block_size]
        labels = ([IGNORE_INDEX] * len(prompt_ids) + answer_ids)[:block_size]
        if len(ids) < 8:
            continue
        examples.append((ids, labels, row))
    return examples, enc, rows


def collate(batch, device, pad_to):
    import torch

    from data_sources.training_format import IGNORE_INDEX

    xs, ys = [], []
    for ids, labels, _ in batch:
        ids = ids[:pad_to]
        labels = labels[:pad_to]
        pad = pad_to - len(ids)
        xs.append(ids[:-1] + [0] * (pad + 1))
        ys.append(labels[1:] + [IGNORE_INDEX] * (pad + 1))
    return (torch.tensor(xs, dtype=torch.long, device=device),
            torch.tensor(ys, dtype=torch.long, device=device))


def probe(model, enc, device, max_new_tokens=60):
    import torch

    from data_sources.training_format import PROMPT_TEMPLATE

    model.eval()
    results = []
    for question, numeric, keywords in PROBES:
        asked = question
        if question in PROBE_CONTEXT:
            asked = f"{PROBE_CONTEXT[question]}\n\n{question}"
        prompt = PROMPT_TEMPLATE.format(prompt=asked)
        ids = enc.encode_ordinary(prompt)
        torch.manual_seed(SEED)
        with torch.no_grad():
            out = model.generate(torch.tensor(ids, device=device).unsqueeze(0),
                                 max_new_tokens, temperature=1e-5, top_k=1,
                                 stop_on_repetition=True, eos_token_id=enc.eot_token)
        generated = out[0, len(ids):].tolist()
        if enc.eot_token in generated:
            generated = generated[:generated.index(enc.eot_token)]
        answer = enc.decode(generated).strip()

        if numeric is not None:
            numbers = [float(n.replace(",", "")) for n in
                       re.findall(r"-?\d{1,3}(?:,\d{3})+(?:\.\d+)?|-?\d+(?:\.\d+)?", answer)]
            correct = any(abs(v - numeric) < 0.05 for v in numbers)
        else:
            correct = any(k.lower() in answer.lower() for k in keywords)
        results.append({"question": question, "answer": answer[:200],
                        "correct": bool(correct),
                        "expected": numeric if numeric is not None else keywords})
    model.train()
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--from-scratch", action="store_true",
                        help="random init instead of the frozen baseline")
    parser.add_argument("--checkpoint", default=CHECKPOINT)
    args = parser.parse_args()

    import torch

    from models import DeepSeekConfig, DeepSeekV3

    torch.manual_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    examples, enc, rows = build_examples(args.block_size)
    print(f"{len(examples)} examples, {len(rows)} rows, device {device}")
    by_task = {}
    for _, _, row in examples:
        by_task[row["task_type"]] = by_task.get(row["task_type"], 0) + 1
    print("by task type:", by_task)

    if args.from_scratch:
        model = DeepSeekV3(DeepSeekConfig.default()).to(device)
        started_from = "random init"
    else:
        from inference import load_model_for_inference

        model, _ = load_model_for_inference(args.checkpoint, device=device)
        started_from = args.checkpoint
    model.train()
    print(f"started from: {started_from}")

    before = probe(model, enc, device)
    print("\nbefore training:")
    for r in before:
        print(f"  [{'OK' if r['correct'] else 'XX'}] {r['question'][:46]} -> {r['answer'][:70]!r}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95),
                                  weight_decay=0.0, eps=1e-9)
    rng = torch.Generator().manual_seed(SEED)
    losses, started = [], time.time()
    # Warmup matters here: the optimizer state is fresh, and without it the
    # first few updates moved the loss UP (4.70 -> 7.01 in a 6-step smoke run).
    warmup = max(10, args.steps // 10)

    for step in range(args.steps):
        if step < warmup:
            lr = args.lr * (step + 1) / warmup
        else:
            progress = (step - warmup) / max(1, args.steps - warmup)
            lr = args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))
        for group in optimizer.param_groups:
            group["lr"] = lr

        picks = torch.randint(0, len(examples), (args.batch_size,), generator=rng).tolist()
        X, Y = collate([examples[i] for i in picks], device, args.block_size)
        _, loss, _, _ = model(X, Y, return_logits=False)
        loss = loss.mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(loss.detach().item())
        if step % 25 == 0 or step == args.steps - 1:
            print(f"  step {step:>4}  lr {lr:.2e}  loss {loss.detach().item():.4f}  "
                  f"{(time.time() - started) / 60:.1f} min")

    after = probe(model, enc, device)
    print("\nafter training:")
    for r in after:
        print(f"  [{'OK' if r['correct'] else 'XX'}] {r['question'][:46]} -> {r['answer'][:110]!r}")

    first, last = sum(losses[:5]) / 5, sum(losses[-5:]) / 5
    correct_after = sum(1 for r in after if r["correct"])
    loss_gate = last < 0.5 or (first / max(last, 1e-9)) >= 10
    probe_gate = correct_after >= 3
    result = {
        "started_from": started_from,
        "examples": len(examples), "by_task_type": by_task,
        "steps": args.steps, "batch_size": args.batch_size,
        "block_size": args.block_size, "lr": args.lr, "seed": SEED,
        "loss_first5": round(first, 4), "loss_last5": round(last, 4),
        "loss_reduction_factor": round(first / max(last, 1e-9), 2),
        "minutes": round((time.time() - started) / 60, 1),
        "probes_before": before, "probes_after": after,
        "probes_correct_after": correct_after,
        "gate_loss": loss_gate, "gate_probes": probe_gate,
        "GATE_PASSED": bool(loss_gate and probe_gate),
    }
    os.makedirs("reports", exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)

    print(f"\nloss {first:.4f} -> {last:.4f} ({result['loss_reduction_factor']}x)")
    print(f"probes correct: {correct_after}/4")
    print(f"GATE: {'PASSED' if result['GATE_PASSED'] else 'FAILED'}")
    print(f"wrote {OUT}")
    sys.exit(0 if result["GATE_PASSED"] else 1)


if __name__ == "__main__":
    main()
