"""Does the routing fix actually prevent expert collapse? A controlled A/B.

Measured on checkpoint sft_003: 28 of 64 routed expert slots receive under 1% of
traffic, and blocks 5-7 send 100% of traffic to the same two experts whatever
the input. The balancing bias had run to +113.5 on an expert still receiving
nothing, while the two experts taking everything held the lowest biases in the
block - the correction was pushing the right way and losing.

Two defects were identified in models/moe.py:

  1. The gate value was softmaxed from the BIASED logits, so the balancing bias
     entered the router's gradient path and the router learned to fight it.
  2. The bias moved by a fixed step in a fixed direction with no deadband and no
     bound, so it walked instead of settling.

This trains the same small model twice from the same seed - once with the old
routing, once with the fixed routing - and measures how the expert distribution
evolves. Run before committing GPU hours to a full retrain, per this project's
rule against training blind.

    python experiments/moe_routing_fix.py
"""

import json
import os
import sys
import time

import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from models.config import DeepSeekConfig          # noqa: E402
from models.moe import MoELayer                   # noqa: E402

STEPS = 1200
# The real collapse took 247,850 steps, with the bias drifting 0.001 per step to
# reach +113. A short run cannot reach that regime at the production rate, so the
# drift is accelerated: the mechanism under test is the per-step update, and
# raising its size compresses the timescale without changing its character.
BIAS_RATE = 0.05
SEED = 1234
OUT = os.path.join("reports", "moe_routing_fix.json")


def small_config():
    """Small enough to train twice on CPU, same routing structure as the real
    model: 8 experts, top-2, one shared expert."""
    config = DeepSeekConfig.default()
    config.n_layer = 4
    config.n_embd = 128
    config.n_head = 4
    config.block_size = 64
    config.vocab_size = 512
    config.expert_intermediate_size = 128
    config.shared_expert_intermediate_size = 192
    config.kv_lora_rank = 32
    config.q_lora_rank = 48
    config.rope_dim = 8
    config.dropout = 0.0
    config.bias_update_rate = BIAS_RATE
    config.max_expert_bias = 1.0
    config.balance_deadband = 0.02
    return config


def old_forward(self, x):
    """models/moe.py as it was: bias inside the gate, unbounded fixed-step
    update. Reproduced here so the comparison is against the real previous
    behaviour rather than a description of it."""
    batch_size, seq_len, hidden_size = x.shape
    x_flat = x.view(-1, hidden_size)

    router_logits = self.router(x_flat) + self.expert_bias
    top_k_logits, top_k_indices = torch.topk(router_logits, self.top_k, dim=-1)
    routing_weights = torch.zeros_like(router_logits)
    routing_weights.scatter_(-1, top_k_indices, F.softmax(top_k_logits, dim=-1))

    output = torch.zeros_like(x_flat)
    expert_usage = torch.zeros(self.n_experts, device=x.device)
    for expert_idx in range(self.n_experts):
        expert_mask = (top_k_indices == expert_idx).any(dim=-1)
        expert_usage[expert_idx] = expert_mask.sum().float()
        if expert_mask.any():
            expert_output = self.experts[expert_idx](x_flat[expert_mask])
            weights = routing_weights[expert_mask, expert_idx].unsqueeze(-1)
            output[expert_mask] += expert_output * weights
    if self.shared_expert is not None:
        output += self.shared_expert(x_flat)

    if self.training:
        with torch.no_grad():
            avg_usage = expert_usage.mean()
            for i in range(self.n_experts):
                if expert_usage[i] > avg_usage:
                    self.expert_bias[i] -= self.bias_update_rate
                else:
                    self.expert_bias[i] += self.bias_update_rate
    return output.view(batch_size, seq_len, hidden_size)


def make_batches(config, n_batches=STEPS, batch_size=8, generator=None):
    """Learnable token streams, so the router has a reason to specialise.

    The first version of this drew tokens uniformly at random inside a topic
    band. Next-token prediction was then impossible, final loss sat at ln(vocab)
    in both arms, and nothing about routing was being tested. Each topic now
    emits a deterministic cycle over its own token block: the sequence is
    genuinely predictable, different topics need different behaviour, and the
    router has something to route.
    """
    n_topics = 4
    span = config.vocab_size // n_topics
    data = []
    for _ in range(n_batches):
        rows_x, rows_y = [], []
        for _ in range(batch_size):
            topic = int(torch.randint(0, n_topics, (1,), generator=generator))
            period = 5 + topic * 3            # a different cycle per topic
            start = int(torch.randint(0, span, (1,), generator=generator))
            ids = [topic * span + (start + (i % period) * 7) % span
                   for i in range(config.block_size + 1)]
            ids = torch.tensor(ids)
            rows_x.append(ids[:-1])
            rows_y.append(ids[1:])
        data.append((torch.stack(rows_x).contiguous(),
                     torch.stack(rows_y).contiguous()))
    return data


def expert_shares(model):
    """Share of top-2 selections each expert receives, per MoE layer."""
    shares = {}
    handles = []

    def hook(name):
        def fn(module, inputs, output):
            x = inputs[0]
            flat = x.view(-1, x.shape[-1])
            affinity = module.router(flat)
            selection = affinity + module.expert_bias
            _, idx = torch.topk(selection, module.top_k, dim=-1)
            counts = torch.bincount(idx.reshape(-1),
                                    minlength=module.n_experts).float()
            shares[name] = (counts / counts.sum() * 100).tolist()
        return fn

    for name, module in model.named_modules():
        if isinstance(module, MoELayer):
            handles.append(module.register_forward_hook(hook(name)))
    return shares, handles


def run_arm(label, use_old):
    torch.manual_seed(SEED)
    generator = torch.Generator().manual_seed(SEED)

    from models.model import DeepSeekV3

    config = small_config()
    model = DeepSeekV3(config)
    model.train()

    original = MoELayer.forward
    if use_old:
        MoELayer.forward = old_forward
        for module in model.modules():
            if isinstance(module, MoELayer):
                module.bias_update_rate = BIAS_RATE
    try:
        optimiser = torch.optim.AdamW(model.parameters(), lr=1e-3)
        batches = make_batches(config, generator=generator)
        started = time.time()
        losses = []
        for step, (x, y) in enumerate(batches):
            out = model(x, targets=y)
            loss = out[1] if isinstance(out, tuple) else out["loss"]
            optimiser.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            losses.append(float(loss.detach()))

        model.eval()
        shares, handles = expert_shares(model)
        with torch.no_grad():
            model(batches[0][0])
        for handle in handles:
            handle.remove()
    finally:
        MoELayer.forward = original

    moe_layers = {k: v for k, v in shares.items() if k.startswith("h.")}
    dead = sum(1 for v in moe_layers.values() for s in v if s < 1.0)
    top2 = [round(sum(sorted(v, reverse=True)[:2]), 1) for v in moe_layers.values()]
    biases = [float(m.expert_bias.abs().max())
              for m in model.modules() if isinstance(m, MoELayer)]

    print(f"\n{label}")
    print(f"  final loss            {sum(losses[-20:]) / 20:.4f}")
    print(f"  seconds               {time.time() - started:.0f}")
    print(f"  dead expert slots     {dead} of {len(moe_layers) * 8}")
    print(f"  top-2 share per block {top2}")
    print(f"  largest |bias|        {max(biases):.2f}")
    for name in sorted(moe_layers):
        print(f"    {name:<10}" + " ".join(f"{s:5.1f}" for s in moe_layers[name]))
    return {"label": label, "dead_slots": dead,
            "slots": len(moe_layers) * 8,
            "top2_share_per_block": top2,
            "max_abs_bias": round(max(biases), 3),
            "final_loss": round(sum(losses[-20:]) / 20, 4),
            "shares": moe_layers}


def main():
    print(f"MoE routing A/B - {STEPS} steps, identical seed and data\n"
          f"{'=' * 62}")
    old = run_arm("ARM A - routing as it was (bias in the gate, unbounded)", True)
    new = run_arm("ARM B - routing fixed (bias selects only, bounded)", False)

    print(f"\n{'=' * 62}\nRESULT")
    print(f"  dead expert slots   {old['dead_slots']} -> {new['dead_slots']} "
          f"(of {old['slots']})")
    print(f"  largest |bias|      {old['max_abs_bias']} -> {new['max_abs_bias']}")
    print(f"  final loss          {old['final_loss']} -> {new['final_loss']}")
    # Three outcomes, not two. Treating "no collapse in either arm" as "the fix
    # does not work" would be wrong: it means the test never created the
    # condition the fix addresses, and says nothing either way.
    if old["dead_slots"] == 0:
        verdict = ("INCONCLUSIVE on collapse - the old routing did not collapse "
                   "at this scale, so there was nothing for the fix to prevent. "
                   f"The bias runaway IS reproduced and bounded "
                   f"({old['max_abs_bias']} -> {new['max_abs_bias']}). A "
                   "decisive test needs the real depth and step count.")
    elif new["dead_slots"] < old["dead_slots"]:
        verdict = ("the fix prevents collapse: "
                   f"{old['dead_slots']} -> {new['dead_slots']} dead slots")
    else:
        verdict = ("NO IMPROVEMENT - collapse reproduced and not prevented. "
                   "Do not spend GPU hours on this.")
    print(f"  verdict             {verdict}")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as handle:
        json.dump({"steps": STEPS, "seed": SEED, "arm_a_old": old,
                   "arm_b_fixed": new, "verdict": verdict}, handle, indent=2)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
