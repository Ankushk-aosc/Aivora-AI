import datetime
import json
import math
import os
import sys
import time
from contextlib import nullcontext

import torch
import yaml
from tqdm.auto import tqdm

from models import DeepSeekConfig, DeepSeekV3

from .data_loader import MixedShardedLoader, estimate_loss, get_batch

CONFIGS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs")
CHECKPOINTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "checkpoints", "base")


def load_preset(preset_name: str) -> dict:
    path = os.path.join(CONFIGS_DIR, f"{preset_name}.yaml")
    if not os.path.exists(path):
        available = [f[:-5] for f in os.listdir(CONFIGS_DIR) if f.endswith(".yaml") and f != "model_config.yaml"]
        raise ValueError(f"Unknown preset '{preset_name}'. Available presets: {available}")
    with open(path) as f:
        return yaml.safe_load(f)


def detect_device():
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_context(device_type: str):
    if device_type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.cuda.amp.autocast(dtype=torch.bfloat16), "bfloat16"
    if device_type == "cuda":
        return torch.cuda.amp.autocast(dtype=torch.float16), "float16"
    return nullcontext(), "float32"


def runtime_info(device):
    info = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "device": device,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info["gpu_name"] = props.name
        info["gpu_memory_gb"] = round(props.total_memory / 1024 ** 3, 2)
    else:
        info["gpu_name"] = "Not available"
        info["gpu_memory_gb"] = "Not available"
    return info


def save_checkpoint(model, optimizer, config, preset, step, train_loss, val_loss, best_val_loss,
                     seed, tokens_processed=0, device="cpu"):
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
    ckpt_path = os.path.join(CHECKPOINTS_DIR, f"checkpoint_{step}.pt")
    meta_path = os.path.join(CHECKPOINTS_DIR, f"checkpoint_{step}.json")

    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": step,
        "best_val_loss": best_val_loss,
        "tokens_processed": tokens_processed,
    }, ckpt_path)

    metadata = {
        "step": step,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "best_val_loss": best_val_loss,
        "tokens_processed": tokens_processed,
        "model_config": config.to_dict(),
        "dataset_config": {
            "preset": preset.get("name"),
            "dataset_mix": preset.get("dataset_mix"),
            "batch_size": preset.get("batch_size"),
            "seq_len": preset.get("seq_len"),
            "gradient_accumulation_steps": preset.get("gradient_accumulation_steps"),
            "learning_rate": preset.get("learning_rate"),
        },
        "train_tokens_budget": preset.get("train_tokens"),
        "validation_tokens_budget": preset.get("validation_tokens"),
        "dataset_manifest": _read_manifest_snapshot(),
        "seed": seed,
        "runtime": runtime_info(device),
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
    }
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    return ckpt_path, meta_path


def _read_manifest_snapshot():
    """Embed the dataset provenance (ids, licenses, revisions, token counts)
    that was actually used to build the shards, so a checkpoint is
    self-describing."""
    path = os.path.join("data", "dataset_manifest.json")
    if not os.path.exists(path):
        return "Not available"
    with open(path) as f:
        return json.load(f)


def load_checkpoint(path, model, optimizer=None):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    meta_path = path.rsplit(".pt", 1)[0] + ".json"
    metadata = {}
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            metadata = json.load(f)
    return ckpt.get("step", metadata.get("step", 0)), metadata


def train_model(preset_name: str = "tiny_debug", resume: str = None, use_wandb: bool = False,
                 shards_root: str = os.path.join("data", "shards"), seed: int = 42):
    preset = load_preset(preset_name)
    config = DeepSeekConfig.default()

    learning_rate = float(preset["learning_rate"])
    max_iters = int(preset["max_steps"])
    warmup_steps = int(preset["warmup_steps"])
    min_lr = float(preset["min_lr"])
    eval_interval = int(preset["eval_interval"])
    eval_iters = int(preset["eval_iters"])
    batch_size = int(preset["batch_size"])
    gradient_accumulation_steps = int(preset["gradient_accumulation_steps"])
    dataset_mix = preset["dataset_mix"]
    seq_len = preset.get("seq_len")  # None -> use full config.block_size

    device = detect_device()
    device_type = "cuda" if device == "cuda" else "cpu"
    ctx, dtype = build_context(device_type if device == "cuda" else "cpu")

    from .data_loader import resolve_seq_len
    effective_seq_len = resolve_seq_len(config, seq_len)
    print(f"Preset: {preset_name} | device: {device} | dtype: {dtype}")
    print(f"batch_size={batch_size} seq_len={effective_seq_len} "
          f"(model block_size={config.block_size}) grad_accum={gradient_accumulation_steps}")

    wandb = None
    if use_wandb:
        try:
            import wandb as _wandb
            from dotenv import load_dotenv
            load_dotenv()
            wandb = _wandb
            wandb.init(project="deepseek-v3-financial", config={**preset, **config.to_dict(), "device": device})
        except Exception as e:
            print(f"wandb logging disabled ({e})")
            wandb = None

    torch.manual_seed(seed)
    model = DeepSeekV3(config).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {total_params:,} parameters")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, betas=(0.9, 0.95), weight_decay=0.1, eps=1e-9,
    )

    train_loader = MixedShardedLoader(shards_root, dataset_mix, split="train", seed=seed)
    val_loader = MixedShardedLoader(shards_root, dataset_mix, split="validation", seed=seed)
    loaders = {"train": train_loader, "validation": val_loader}

    start_step = 0
    best_val_loss = float("inf")
    tokens_processed = 0
    if resume is not None:
        start_step, metadata = load_checkpoint(resume, model, optimizer)
        best_val_loss = metadata.get("best_val_loss", float("inf"))
        tokens_processed = metadata.get("tokens_processed", 0)
        print(f"Resumed from {resume} at step {start_step} "
              f"(best_val_loss={best_val_loss}, tokens_processed={tokens_processed:,})")

    model.train()
    last_train_loss = None
    last_val_loss = None
    run_start = time.time()

    for step in tqdm(range(start_step, max_iters)):
        if step % eval_interval == 0 and step != start_step:
            losses = estimate_loss(model, loaders, config, eval_iters, batch_size,
                                    device_type, device, ctx, seq_len=seq_len)
            last_train_loss, last_val_loss = losses["train"].item(), losses["val"].item()
            elapsed = time.time() - run_start
            print(f"step {step}: train {last_train_loss:.4f}, val {last_val_loss:.4f} | "
                  f"tokens {tokens_processed:,} | {tokens_processed / max(elapsed, 1e-9):.0f} tok/s | "
                  f"elapsed {elapsed:.0f}s")
            if wandb:
                wandb.log({"step": step, "train_loss": last_train_loss, "val_loss": last_val_loss,
                            "tokens_processed": tokens_processed})

            if last_val_loss < best_val_loss:
                best_val_loss = last_val_loss
                save_checkpoint(model, optimizer, config, preset, step, last_train_loss,
                                 last_val_loss, best_val_loss, seed, tokens_processed, device)

        X, y = get_batch(loaders, "train", config, batch_size, device_type, device, seq_len)
        with ctx:
            _, total_loss, main_loss, mtp_loss = model(X, y)
            loss = total_loss / gradient_accumulation_steps

        # Part 29: never silently train through NaN/Inf.
        if not torch.isfinite(total_loss):
            raise RuntimeError(
                f"Non-finite loss at step {step}: total={total_loss.item()} "
                f"main={main_loss.item() if main_loss is not None else None} "
                f"mtp={mtp_loss.item() if mtp_loss is not None else None}. "
                f"lr={optimizer.param_groups[0]['lr']}, batch={X.shape}. Stopping run."
            )

        loss.backward()
        tokens_processed += X.numel()

        if ((step + 1) % gradient_accumulation_steps == 0) or (step + 1 == max_iters):
            bad_grad = next(
                (n for n, p in model.named_parameters()
                 if p.grad is not None and not torch.isfinite(p.grad).all()),
                None,
            )
            if bad_grad is not None:
                raise RuntimeError(
                    f"Non-finite gradient in '{bad_grad}' at step {step}. Stopping run."
                )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        if step < warmup_steps:
            lr = learning_rate * (step + 1) / max(1, warmup_steps)
        else:
            progress = (step - warmup_steps) / max(1, (max_iters - warmup_steps))
            lr = min_lr + (learning_rate - min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        if wandb:
            wandb.log({
                "step": step, "total_loss": total_loss.item(), "main_loss": main_loss.item(),
                "mtp_loss": mtp_loss.item() if mtp_loss is not None else None, "learning_rate": lr,
            })

    # Final measured losses before saving the last checkpoint.
    losses = estimate_loss(model, loaders, config, eval_iters, batch_size,
                            device_type, device, ctx, seq_len=seq_len)
    last_train_loss, last_val_loss = losses["train"].item(), losses["val"].item()
    best_val_loss = min(best_val_loss, last_val_loss)

    ckpt_path, meta_path = save_checkpoint(
        model, optimizer, config, preset, max_iters, last_train_loss, last_val_loss,
        best_val_loss, seed, tokens_processed, device,
    )
    elapsed = time.time() - run_start
    print(f"Training completed in {elapsed:.0f}s | tokens processed: {tokens_processed:,}")
    print(f"Final train loss: {last_train_loss:.4f} | Final val loss: {last_val_loss:.4f}")
    print(f"Final checkpoint: {ckpt_path}")

    if wandb:
        wandb.finish()

    return model, config, ckpt_path
