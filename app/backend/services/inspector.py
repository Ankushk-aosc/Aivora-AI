"""Model / Token / MoE inspectors (Parts 32-35 / §39-42).

Every value returned here is read from an actual forward pass of the
loaded model. Nothing is illustrative or precomputed. Where a quantity
genuinely cannot be produced, the field says "Not available" rather than
carrying a plausible-looking number.
"""

import torch
import torch.nn.functional as F

from data_sources.tokenizer import get_encoding


# ----------------------------------------------------------------------
# Token inspector (Part 34)
# ----------------------------------------------------------------------

def inspect_tokens(text: str) -> dict:
    enc = get_encoding()
    ids = enc.encode_ordinary(text)
    return {
        "text": text,
        "token_count": len(ids),
        "tokens": [
            {"position": i, "id": tid, "piece": enc.decode([tid])}
            for i, tid in enumerate(ids)
        ],
        "token_ids": ids,
    }


# ----------------------------------------------------------------------
# Architecture page (Part 35)
# ----------------------------------------------------------------------

def inspect_architecture(model, device="cpu") -> dict:
    config = model.config
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    by_component = {}
    for name, p in model.named_parameters():
        by_component[name.split(".")[0]] = by_component.get(name.split(".")[0], 0) + p.numel()

    gpu = "Not available"
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)

    return {
        "model": "DeepSeek-V3-inspired Financial LLM",
        "total_parameters": total,
        "trainable_parameters": trainable,
        "parameters_by_component": by_component,
        "vocab_size": config.vocab_size,
        "context_length": config.block_size,
        "embedding_size": config.n_embd,
        "layers": config.n_layer,
        "attention_heads": config.n_head,
        "kv_lora_rank": config.kv_lora_rank,
        "q_lora_rank": config.q_lora_rank,
        "rope_dim": config.rope_dim,
        "n_experts": config.n_experts,
        "experts_per_token": config.n_experts_per_token,
        "mtp_heads": config.mtp_num_heads,
        "weight_tying": bool(model.wte.weight is model.lm_head.weight),
        "device": str(device),
        "gpu": gpu,
    }


# ----------------------------------------------------------------------
# MoE inspector (Part 33) - real router output
# ----------------------------------------------------------------------

@torch.no_grad()
def inspect_moe_routing(model, text: str, layer: int = 0, device="cpu") -> dict:
    """Capture the actual router distribution for every token at `layer`.

    Hooks the MoE router Linear so the values are exactly what the model
    computed during this forward pass.
    """
    enc = get_encoding()
    ids = enc.encode_ordinary(text)
    if not ids:
        return {"error": "empty input"}
    if layer >= len(model.h):
        return {"error": f"layer {layer} out of range (model has {len(model.h)})"}

    idx = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
    moe = model.h[layer].mlp
    captured = {}

    def hook(_module, inputs, output):
        captured["router_logits"] = output.detach()

    handle = moe.router.register_forward_hook(hook)
    try:
        model(idx)
    finally:
        handle.remove()

    if "router_logits" not in captured:
        return {"error": "router output was not captured"}

    # Reproduce the model's own routing arithmetic, including expert_bias.
    logits = captured["router_logits"] + moe.expert_bias
    top_k = moe.top_k
    top_logits, top_indices = torch.topk(logits, top_k, dim=-1)
    top_weights = F.softmax(top_logits, dim=-1)
    full_probs = F.softmax(logits, dim=-1)

    tokens = []
    for i, tid in enumerate(ids):
        selected = top_indices[i].tolist()
        tokens.append({
            "position": i,
            "token_id": tid,
            "piece": enc.decode([tid]),
            "expert_scores": {
                f"expert_{e}": round(float(full_probs[i, e]), 4)
                for e in range(logits.size(-1))
            },
            "selected_experts": selected,
            "selected_weights": {
                f"expert_{e}": round(float(w), 4)
                for e, w in zip(selected, top_weights[i].tolist())
            },
        })

    return {
        "layer": layer,
        "n_experts": int(logits.size(-1)),
        "experts_per_token": top_k,
        "expert_bias": [round(float(b), 6) for b in moe.expert_bias.tolist()],
        "tokens": tokens,
    }


# ----------------------------------------------------------------------
# Full forward-pass inspector (Part 32)
# ----------------------------------------------------------------------

@torch.no_grad()
def inspect_forward(model, text: str, device="cpu", top_n: int = 10) -> dict:
    """Walk the real stack: embeddings -> blocks (MLA+MoE) -> RMSNorm ->
    LM head -> logits -> next-token probabilities."""
    enc = get_encoding()
    ids = enc.encode_ordinary(text)
    if not ids:
        return {"error": "empty input"}

    idx = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
    t = idx.size(1)

    def stats(tensor):
        return {
            "shape": list(tensor.shape),
            "mean": round(float(tensor.mean()), 6),
            "std": round(float(tensor.std()), 6),
            "min": round(float(tensor.min()), 6),
            "max": round(float(tensor.max()), 6),
            "finite": bool(torch.isfinite(tensor).all()),
        }

    stages = []

    pos = torch.arange(0, t, dtype=torch.long, device=device)
    tok_emb = model.wte(idx)
    pos_emb = model.wpe(pos)
    x = tok_emb + pos_emb
    stages.append({"stage": "token_embedding", **stats(tok_emb)})
    stages.append({"stage": "position_embedding", **stats(pos_emb)})
    stages.append({"stage": "embeddings_sum", **stats(x)})

    for i, block in enumerate(model.h):
        attn_out = block.attn(block.ln_1(x))
        x = x + attn_out
        stages.append({"stage": f"block{i}_MLA", **stats(attn_out)})
        moe_out = block.mlp(block.ln_2(x))
        x = x + moe_out
        stages.append({"stage": f"block{i}_MoE", **stats(moe_out)})

    x = model.ln_f(x)
    stages.append({"stage": "final_RMSNorm", **stats(x)})

    logits = model.lm_head(x)
    stages.append({"stage": "lm_head_logits", **stats(logits)})

    next_logits = logits[0, -1, :]
    probs = F.softmax(next_logits, dim=-1)
    top_p, top_i = torch.topk(probs, top_n)

    mtp = "Not available (model has no MTP heads)"
    if model.mtp_heads is not None and t > 1:
        head = model.mtp_heads[0]
        future = model.wte(idx[:, 1:])
        if future.size(1) < x.size(1):
            pad = torch.zeros(1, x.size(1) - future.size(1), model.config.n_embd,
                               device=device, dtype=future.dtype)
            future = torch.cat([future, pad], dim=1)
        mtp_hidden = head(x, future)
        mtp = {"stage": "MTP_head_0", **stats(mtp_hidden)}

    return {
        "input_text": text,
        "token_ids": ids,
        "token_count": len(ids),
        "stages": stages,
        "mtp": mtp,
        "next_token_probabilities": [
            {"rank": r + 1, "token_id": int(i), "piece": enc.decode([int(i)]),
             "probability": round(float(p), 6)}
            for r, (p, i) in enumerate(zip(top_p.tolist(), top_i.tolist()))
        ],
    }
