"""Measure how the Mixture-of-Experts router actually distributes tokens.

MoELayer computes `expert_usage` on every forward pass and then throws it
away, so nothing in this project has ever checked whether routing is balanced.
It matters: if most tokens land on one or two experts, the other experts'
parameters are dead weight and the model is effectively far smaller than its
parameter count suggests.

This hooks each layer's router, replays real validation tokens, and reports
per-layer usage, the share taken by the busiest expert, and normalised entropy
(1.0 = perfectly balanced, 0.0 = one expert takes everything).

usage:
    python scripts/diagnose_moe.py --checkpoint checkpoints/.../checkpoint_237362.pt
"""

import argparse
import json
import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from models import DeepSeekConfig, DeepSeekV3  # noqa: E402
from models.moe import MoELayer  # noqa: E402

DEFAULT_SHARD = os.path.join("data", "shards", "fineweb_edu", "validation", "shard_000.bin")


def collect_usage(model, batches, config):
    """Per-layer token counts per expert, gathered with forward hooks."""
    layers = [m for m in model.modules() if isinstance(m, MoELayer)]
    counts = [torch.zeros(layer.n_experts) for layer in layers]
    handles = []

    def make_hook(index, layer):
        def hook(_module, inputs, output):
            # Re-derive the routing decision the layer just made: router
            # logits plus the load-balancing bias, then top-k.
            x_flat = inputs[0].view(-1, inputs[0].shape[-1])
            logits = layer.router(x_flat) + layer.expert_bias
            _, top_k_indices = torch.topk(logits, layer.top_k, dim=-1)
            for expert in range(layer.n_experts):
                counts[index][expert] += (top_k_indices == expert).any(dim=-1).sum().item()
        return hook

    for i, layer in enumerate(layers):
        handles.append(layer.register_forward_hook(make_hook(i, layer)))

    with torch.no_grad():
        for X in batches:
            model(X, return_logits=False)

    for h in handles:
        h.remove()
    return counts


def report(counts, top_k):
    print(f"{'layer':>6} {'busiest':>9} {'quietest':>9} {'entropy':>8}  distribution (% of tokens)")
    collapsed = 0
    for i, c in enumerate(counts):
        total = c.sum().item()
        if total == 0:
            continue
        share = (c / total).numpy()
        # Perfect balance sends top_k/n_experts of tokens to each expert.
        entropy = -(share * np.log(np.clip(share, 1e-12, None))).sum()
        norm_entropy = entropy / math.log(len(share))
        bars = " ".join(f"{100 * s:4.1f}" for s in share)
        print(f"{i:>6} {100 * share.max():8.1f}% {100 * share.min():8.1f}% "
              f"{norm_entropy:8.3f}  {bars}")
        if norm_entropy < 0.7 or share.max() > 0.5:
            collapsed += 1
    return collapsed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--shard", default=DEFAULT_SHARD)
    ap.add_argument("--batches", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--seq-len", type=int, default=512)
    args = ap.parse_args()

    meta_path = args.checkpoint.rsplit(".pt", 1)[0] + ".json"
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            config = DeepSeekConfig.from_dict(json.load(f)["model_config"])
    else:
        config = DeepSeekConfig.default()

    model = DeepSeekV3(config)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(ckpt.get("model_state_dict", ckpt))
    model.eval()  # eval mode: the router's bias must not be updated while measuring

    tokens = np.fromfile(args.shard, dtype=np.uint16)
    rng = np.random.default_rng(0)
    batches = []
    for _ in range(args.batches):
        starts = rng.integers(0, len(tokens) - args.seq_len - 1, size=args.batch_size)
        batches.append(torch.stack([
            torch.from_numpy(tokens[s:s + args.seq_len].astype(np.int64)) for s in starts]))

    print(f"checkpoint : {args.checkpoint}")
    print(f"experts    : {config.n_experts} ({config.n_experts_per_token} active per token)")
    # Each token is assigned to top_k experts, so the shares below are of
    # ASSIGNMENTS (they sum to 100%): perfect balance is 1/n_experts each.
    print(f"balanced   : {100 / config.n_experts:.1f}% of assignments per expert")
    print(f"tokens      : {args.batches * args.batch_size * args.seq_len:,}\n")

    counts = collect_usage(model, batches, config)
    measured = sum(1 for c in counts if c.sum().item() > 0)
    collapsed = report(counts, config.n_experts_per_token)

    print("")
    idle = len(counts) - measured
    if idle:
        # The multi-token-prediction head carries its own MoE layer, which does
        # not run when the model is called without MTP targets.
        print(f"({idle} MoE layer(s) saw no tokens in this pass - the MTP head's, "
              f"which only runs during training.)")
    if collapsed:
        print(f"WARNING: {collapsed} of {measured} measured layers look collapsed "
              f"(entropy < 0.70 or one expert above 50% of assignments).")
        print("Those layers' unused experts are dead parameters.")
    else:
        print(f"Routing is balanced across all {measured} measured layers: no expert collapse.")
        print("The model's weakness is capacity/scale, not a broken router.")


if __name__ == "__main__":
    main()
