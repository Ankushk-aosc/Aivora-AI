"""Stage B: financial instruction tuning (Part 15 / §22).

Loads a Stage A (base) checkpoint and continues training on
instruction/response data, saving to checkpoints/instruction/.
"""

import datetime
import json
import math
import os
import time
from contextlib import nullcontext

import torch
from tqdm.auto import tqdm

from models import DeepSeekConfig, DeepSeekV3

from .instruction_dataset import InstructionDataset, load_instruction_records
from .trainer import build_context, detect_device, runtime_info

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INSTRUCTION_CKPT_DIR = os.path.join(ROOT, "checkpoints", "instruction")
DEFAULT_DATA = os.path.join("data", "instruction", "financial_instructions.jsonl")


def save_instruction_checkpoint(model, optimizer, config, step, train_loss, val_loss,
                                 dataset_path, dataset_stats, base_checkpoint, seed,
                                 tokens_processed, device, hyperparams):
    os.makedirs(INSTRUCTION_CKPT_DIR, exist_ok=True)
    ckpt_path = os.path.join(INSTRUCTION_CKPT_DIR, f"checkpoint_{step}.pt")
    meta_path = os.path.join(INSTRUCTION_CKPT_DIR, f"checkpoint_{step}.json")

    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": step,
        "tokens_processed": tokens_processed,
    }, ckpt_path)

    with open(meta_path, "w") as f:
        json.dump({
            "stage": "instruction",
            "step": step,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "tokens_processed": tokens_processed,
            "model_config": config.to_dict(),
            "base_checkpoint": base_checkpoint,
            "instruction_dataset": dataset_path,
            "instruction_dataset_stats": dataset_stats,
            "hyperparameters": hyperparams,
            "seed": seed,
            "runtime": runtime_info(device),
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        }, f, indent=2)

    return ckpt_path, meta_path


def train_instruction(base_checkpoint: str, data_path: str = DEFAULT_DATA,
                       max_steps: int = 100, batch_size: int = 2, seq_len: int = 256,
                       learning_rate: float = 1e-4, warmup_steps: int = 10,
                       min_lr: float = 1e-5, eval_interval: int = 25, eval_iters: int = 5,
                       val_fraction: float = 0.1, seed: int = 42):
    if not os.path.exists(base_checkpoint):
        raise FileNotFoundError(f"Base checkpoint not found: {base_checkpoint}")

    # Load the config the base checkpoint was actually trained with.
    meta_path = base_checkpoint.rsplit(".pt", 1)[0] + ".json"
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            config = DeepSeekConfig.from_dict(json.load(f)["model_config"])
    else:
        config = DeepSeekConfig.default()

    device = detect_device()
    device = "cpu" if device == "mps" else device
    device_type = "cuda" if device == "cuda" else "cpu"
    ctx, dtype = build_context(device_type)

    print(f"Stage B: financial instruction tuning | device: {device} | dtype: {dtype}")
    print(f"Base checkpoint: {base_checkpoint}")

    torch.manual_seed(seed)
    model = DeepSeekV3(config)
    ckpt = torch.load(base_checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
    model = model.to(device)
    print(f"Loaded base weights ({sum(p.numel() for p in model.parameters()):,} parameters)")

    records = load_instruction_records(data_path)
    if not records:
        raise RuntimeError(f"No usable instruction records in {data_path}")

    split_at = max(1, int(len(records) * (1 - val_fraction)))
    train_records, val_records = records[:split_at], records[split_at:]
    effective_seq = min(seq_len, config.block_size)
    train_ds = InstructionDataset(train_records, effective_seq, seed=seed)
    val_ds = InstructionDataset(val_records or train_records[-1:], effective_seq, seed=seed + 1)

    print(f"Instruction data: {data_path}")
    print(f"  train: {train_ds.stats()}")
    print(f"  val:   {val_ds.stats()}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.95),
                                   weight_decay=0.1, eps=1e-9)

    hyperparams = {
        "max_steps": max_steps, "batch_size": batch_size, "seq_len": effective_seq,
        "learning_rate": learning_rate, "warmup_steps": warmup_steps, "min_lr": min_lr,
    }

    @torch.no_grad()
    def estimate():
        model.eval()
        out = {}
        for name, ds in (("train", train_ds), ("val", val_ds)):
            losses = []
            for _ in range(eval_iters):
                X, Y = ds.get_batch(batch_size, device, device_type)
                with ctx:
                    _, loss, _, _ = model(X, Y)
                losses.append(loss.item())
            out[name] = sum(losses) / len(losses)
        model.train()
        return out

    model.train()
    tokens_processed = 0
    last_train_loss = last_val_loss = None
    run_start = time.time()

    for step in tqdm(range(max_steps)):
        if step % eval_interval == 0 and step != 0:
            losses = estimate()
            last_train_loss, last_val_loss = losses["train"], losses["val"]
            print(f"step {step}: train {last_train_loss:.4f}, val {last_val_loss:.4f} | "
                  f"tokens {tokens_processed:,}")

        X, Y = train_ds.get_batch(batch_size, device, device_type)
        with ctx:
            _, total_loss, _, _ = model(X, Y)

        if not torch.isfinite(total_loss):
            raise RuntimeError(f"Non-finite loss at instruction step {step}. Stopping run.")

        total_loss.backward()
        tokens_processed += X.numel()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        if step < warmup_steps:
            lr = learning_rate * (step + 1) / max(1, warmup_steps)
        else:
            progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
            lr = min_lr + (learning_rate - min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
        for g in optimizer.param_groups:
            g["lr"] = lr

    losses = estimate()
    last_train_loss, last_val_loss = losses["train"], losses["val"]

    ckpt_path, _ = save_instruction_checkpoint(
        model, optimizer, config, max_steps, last_train_loss, last_val_loss,
        data_path, train_ds.stats(), base_checkpoint, seed, tokens_processed, device, hyperparams,
    )
    print(f"Instruction tuning completed in {time.time() - run_start:.0f}s")
    print(f"Final train loss: {last_train_loss:.4f} | val loss: {last_val_loss:.4f}")
    print(f"Checkpoint: {ckpt_path}")
    return model, config, ckpt_path
