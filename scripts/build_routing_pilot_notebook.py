"""Generate the Kaggle notebook for the MoE routing A/B pilot.

Two CPU runs failed to reproduce expert collapse, so neither could prove the
routing fix prevents it. Collapse was observed in the real model at 8 layers
after 247,850 steps, and it worsened with depth - blocks 5 to 7 were the dead
ones. A 4-layer, 1,200-step CPU run cannot reach that regime.

This runs the same A/B at the real depth and width, on a GPU, for enough steps
to enter the drift regime. It is the decisive test, and it is also stage one of
any retrain: if the fixed routing holds its balance here, a full retrain is
justified; if it collapses anyway, the diagnosis is wrong and no GPU weeks
should be spent on it.

    python scripts/build_routing_pilot_notebook.py
    cd training/kaggle/routing_pilot && kaggle kernels push -p .
"""

import json
import os

OUT_DIR = os.path.join("training", "kaggle", "routing_pilot")
NOTEBOOK = "Aivora_Routing_Pilot.ipynb"
KERNEL_ID = "aoscjkjhh/aivora-routing-pilot"

CELLS = []


def md(text):
    CELLS.append({"cell_type": "markdown", "metadata": {},
                  "source": text.splitlines(True)})


def code(text):
    CELLS.append({"cell_type": "code", "execution_count": None, "metadata": {},
                  "outputs": [], "source": text.splitlines(True)})


md("""# Aivora - MoE routing A/B pilot

**Question.** Does the routing fix prevent expert collapse at the real model's
depth?

**Why this run exists.** Checkpoint `sft_003` has 28 of 64 routed expert slots
receiving under 1% of traffic. In blocks 5, 6 and 7 the top-2 router sends 100%
of traffic to the same two experts whatever the input - verified across a
financial filing, technology news, prose, code and general questions. The
balancing bias had run to **+113.5** on an expert receiving nothing, while the
two experts taking everything held the two lowest biases in the block: the
correction was pushing the right way and losing.

**Two defects were found in `models/moe.py`:**

1. The gate value was softmaxed from the **biased** logits, so the balancing
   bias entered the router's gradient path and the router learned to fight it.
2. The bias moved a fixed step in a fixed direction, unbounded, so it walked
   instead of settling - a spread of 148 within one block after 247,850 steps.

**What is already known.** Two CPU A/B runs did *not* reproduce collapse in
either arm, so neither proves the fix works. They did reproduce the runaway and
show the fix bounds it: peak |bias| 4.30 unfixed against 0.63 fixed, at equal
loss. Collapse needs the real depth and far more steps, which is what this runs.

**Decision rule, fixed before the run:**

| Outcome | Reading |
| --- | --- |
| Arm A collapses, Arm B does not | The fix works. A full retrain is justified. |
| Neither collapses | Still inconclusive. Do not spend GPU weeks; investigate what else differed in the original run. |
| Both collapse | The diagnosis is wrong. Stop and re-diagnose. |

Settings: Accelerator -> GPU T4, Internet -> On.
""")

code('''import subprocess, os, json, time

REPO_URL = "https://github.com/Ankushk-aosc/Aivora-AI.git"
REPO_DIR = "/kaggle/working/Aivora-AI"

if not os.path.exists(REPO_DIR):
    r = subprocess.run(["git", "clone", "--depth", "1", REPO_URL, REPO_DIR],
                       capture_output=True, text=True)
    print(r.stdout[-600:], r.stderr[-600:])
    if r.returncode != 0:
        raise RuntimeError("STATUS = BLOCKED: clone failed. Internet -> On?")
os.chdir(REPO_DIR)
import sys
sys.path.insert(0, REPO_DIR)

head = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                      capture_output=True, text=True).stdout.strip()
print("repo at", head)

import torch
print("torch", torch.__version__, "| cuda", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise RuntimeError("STATUS = BLOCKED: no GPU. Settings -> Accelerator -> GPU.")
print(" ", torch.cuda.get_device_properties(0).name)
''')

md("""## Confirm the fix is present in this checkout

The run is meaningless if the clone predates the fix.""")

code('''source = open("models/moe.py").read()
assert "selection_logits" in source, "STATUS = BLOCKED: clone predates the fix"
assert "max_expert_bias" in source, "STATUS = BLOCKED: bias bound missing"
print("routing fix present")
print("  bias steers selection only :", "affinity.gather" in source)
print("  bias bounded              :", "clamp_" in source)
''')

md("""## Training data

`data/shards/*/` is gitignored, so the clone carries no tokenised corpus - the
first attempt at this pilot stopped here, correctly. The corpus is therefore
built on the machine from a public dataset. Expert collapse is a property of the
routing mechanism rather than of any particular domain, so a general corpus
tests it just as well; what matters is that both arms see exactly the same
tokens in the same order.""")

code('''import numpy as np, glob, os

TARGET_TOKENS = int(os.environ.get("PILOT_TOKENS", "18000000"))
corpus = None

# 1. Shards, if a future clone ever carries them.
shards = sorted(glob.glob("data/shards/*/train/*.bin"))
if shards:
    corpus = np.concatenate([np.fromfile(p, dtype=np.uint16) for p in shards])
    source = f"repository shards ({len(shards)} files)"

# 2. Otherwise build one with the project's own tokeniser.
if corpus is None or len(corpus) < TARGET_TOKENS // 4:
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    from datasets import load_dataset

    pieces, total = [], 0
    for name, kwargs in (
            ("HuggingFaceFW/fineweb-edu", {"name": "sample-10BT", "split": "train"}),
            ("wikitext", {"name": "wikitext-103-raw-v1", "split": "train"})):
        try:
            print(f"streaming {name} ...", flush=True)
            ds = load_dataset(name, streaming=True, **kwargs)
            for row in ds:
                text = row.get("text") or ""
                if not text.strip():
                    continue
                ids = enc.encode_ordinary(text)
                ids.append(enc.eot_token)
                pieces.append(np.array(ids, dtype=np.uint16))
                total += len(ids)
                if total >= TARGET_TOKENS:
                    break
            if total >= TARGET_TOKENS // 2:
                source = name
                break
        except Exception as error:
            print(f"  {name} unavailable: {type(error).__name__}: {error}")
            pieces, total = [], 0
    if total == 0:
        raise RuntimeError("STATUS = BLOCKED: no corpus could be obtained")
    corpus = np.concatenate(pieces)

print(f"corpus: {len(corpus):,} tokens from {source}")
print("NOTE: the model was originally trained on 120,857,129 unique tokens. "
      "This pilot tests routing balance, not model quality.")
''')

md("""## The A/B

Both arms use the real configuration - 8 layers, 512 hidden, 8 experts, top-2 -
the same seed and the same data order. The only difference is the routing code.""")

code('''import torch, torch.nn.functional as F, numpy as np, time
from models.config import DeepSeekConfig
from models.moe import MoELayer
from models.model import DeepSeekV3

STEPS = int(os.environ.get("PILOT_STEPS", "4000"))
BATCH = 4
SEED = 1234
device = "cuda"

def old_forward(self, x):
    """models/moe.py before the fix: bias inside the gate, unbounded update."""
    b, t, h = x.shape
    x_flat = x.view(-1, h)
    router_logits = self.router(x_flat) + self.expert_bias
    top_k_logits, top_k_indices = torch.topk(router_logits, self.top_k, dim=-1)
    routing_weights = torch.zeros_like(router_logits)
    routing_weights.scatter_(-1, top_k_indices, F.softmax(top_k_logits, dim=-1))
    out = torch.zeros_like(x_flat)
    usage = torch.zeros(self.n_experts, device=x.device)
    for i in range(self.n_experts):
        mask = (top_k_indices == i).any(dim=-1)
        usage[i] = mask.sum().float()
        if mask.any():
            out[mask] += self.experts[i](x_flat[mask]) * \\
                routing_weights[mask, i].unsqueeze(-1)
    if self.shared_expert is not None:
        out += self.shared_expert(x_flat)
    if self.training:
        with torch.no_grad():
            avg = usage.mean()
            for i in range(self.n_experts):
                if usage[i] > avg:
                    self.expert_bias[i] -= self.bias_update_rate
                else:
                    self.expert_bias[i] += self.bias_update_rate
    return out.view(b, t, h)


def batches(config, n, generator):
    data = []
    for _ in range(n):
        idx = torch.randint(0, len(corpus) - config.block_size - 1, (BATCH,),
                            generator=generator)
        x = np.stack([corpus[i:i + config.block_size] for i in idx.tolist()])
        y = np.stack([corpus[i + 1:i + 1 + config.block_size] for i in idx.tolist()])
        data.append((torch.from_numpy(x.astype(np.int64)),
                     torch.from_numpy(y.astype(np.int64))))
    return data


def measure(model):
    shares = {}
    handles = []
    def hook(name):
        def fn(module, inputs, output):
            flat = inputs[0].reshape(-1, inputs[0].shape[-1])
            sel = module.router(flat) + module.expert_bias
            _, idx = torch.topk(sel, module.top_k, dim=-1)
            c = torch.bincount(idx.reshape(-1), minlength=module.n_experts).float()
            shares[name] = (c / c.sum() * 100).tolist()
        return fn
    for n, m in model.named_modules():
        if isinstance(m, MoELayer) and n.startswith("h."):
            handles.append(m.register_forward_hook(hook(n)))
    return shares, handles


def run_arm(label, use_old):
    torch.manual_seed(SEED)
    gen = torch.Generator().manual_seed(SEED)
    config = DeepSeekConfig.default()          # the real architecture
    model = DeepSeekV3(config).to(device)
    model.train()
    original = MoELayer.forward
    if use_old:
        MoELayer.forward = old_forward
    try:
        opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
        data = batches(config, STEPS, gen)
        started = time.time()
        losses = []
        for step, (x, y) in enumerate(data):
            x, y = x.to(device), y.to(device)
            out = model(x, targets=y)
            loss = out[1] if isinstance(out, tuple) else out["loss"]
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach()))
            if step % 500 == 0:
                print(f"    step {step:>5}  loss {np.mean(losses[-100:]):.4f}  "
                      f"{time.time() - started:.0f}s", flush=True)
        model.eval()
        shares, handles = measure(model)
        with torch.no_grad():
            model(data[0][0].to(device))
        for h in handles: h.remove()
    finally:
        MoELayer.forward = original

    dead = sum(1 for v in shares.values() for s in v if s < 1.0)
    top2 = [round(sum(sorted(v, reverse=True)[:2]), 1) for v in shares.values()]
    bias = max(float(m.expert_bias.abs().max())
               for m in model.modules() if isinstance(m, MoELayer))
    print(f"\\n{label}")
    print(f"  final loss         {np.mean(losses[-100:]):.4f}")
    print(f"  dead expert slots  {dead} of {len(shares) * 8}")
    print(f"  top-2 share/block  {top2}")
    print(f"  largest |bias|     {bias:.2f}")
    for n in sorted(shares, key=lambda s: int(s.split('.')[1])):
        print(f"    {n:<12}" + " ".join(f"{s:5.1f}" for s in shares[n]))
    return {"label": label, "dead_slots": dead, "slots": len(shares) * 8,
            "top2_per_block": top2, "max_abs_bias": round(bias, 3),
            "final_loss": round(float(np.mean(losses[-100:])), 4),
            "shares": shares}

print(f"STEPS = {STEPS}\\n" + "=" * 64)
arm_a = run_arm("ARM A - routing as it was", True)
arm_b = run_arm("ARM B - routing fixed", False)
''')

md("""## Verdict against the pre-registered rule""")

code('''if arm_a["dead_slots"] == 0:
    verdict = ("INCONCLUSIVE - the old routing did not collapse even at full "
               "depth. Do not spend GPU weeks on a retrain; find what else "
               "differed in the original 247,850-step run.")
    justified = False
elif arm_b["dead_slots"] < arm_a["dead_slots"]:
    verdict = (f"FIX WORKS - dead slots {arm_a['dead_slots']} -> "
               f"{arm_b['dead_slots']}. A full retrain is justified.")
    justified = True
else:
    verdict = ("DIAGNOSIS WRONG - both arms collapse. Stop and re-diagnose "
               "before any retrain.")
    justified = False

payload = {"repo_commit": head, "steps": STEPS,
           "corpus_tokens": int(len(corpus)),
           "corpus_note": "the full 120.9M-token corpus is not in version "
                          "control; this pilot cycles the shards present",
           "arm_a_old": arm_a, "arm_b_fixed": arm_b,
           "verdict": verdict, "full_retrain_justified": justified}
with open("/kaggle/working/routing_pilot.json", "w") as handle:
    json.dump(payload, handle, indent=2)

print("=" * 64)
print(f"dead slots   {arm_a['dead_slots']} -> {arm_b['dead_slots']} "
      f"(of {arm_a['slots']})")
print(f"max |bias|   {arm_a['max_abs_bias']} -> {arm_b['max_abs_bias']}")
print(f"final loss   {arm_a['final_loss']} -> {arm_b['final_loss']}")
print(f"\\nVERDICT: {verdict}")
print("\\nwrote /kaggle/working/routing_pilot.json")
''')

md("""## What this does and does not settle

**Does:** whether the old routing collapses at the real depth, and whether the
fixed routing survives the same run.

**Does not:** anything about model quality. The corpus here is a fraction of the
original, the step count is a fraction of 247,850, and no capability gate is
run. A balanced router is a precondition for a worthwhile retrain, not a result
in itself.
""")


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
    metadata = {"id": KERNEL_ID, "title": "Aivora Routing Pilot",
                "code_file": NOTEBOOK, "language": "python",
                "kernel_type": "notebook", "is_private": "true",
                "enable_gpu": "true", "enable_tpu": "false",
                "enable_internet": "true", "dataset_sources": [],
                "competition_sources": [], "kernel_sources": [],
                "model_sources": []}
    with open(os.path.join(OUT_DIR, "kernel-metadata.json"), "w",
              encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    print(f"wrote {path} ({len(CELLS)} cells)")


if __name__ == "__main__":
    main()
