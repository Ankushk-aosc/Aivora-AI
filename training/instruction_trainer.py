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
                                 tokens_processed, device, hyperparams, checkpoints_dir=None):
    target_dir = checkpoints_dir or INSTRUCTION_CKPT_DIR
    os.makedirs(target_dir, exist_ok=True)
    ckpt_path = os.path.join(target_dir, f"checkpoint_{step}.pt")
    meta_path = os.path.join(target_dir, f"checkpoint_{step}.json")

    # Unwrap DataParallel so the saved keys have no "module." prefix and the
    # checkpoint loads the same way regardless of GPU count.
    raw_model = model.module if isinstance(model, torch.nn.DataParallel) else model
    torch.save({
        "model_state_dict": raw_model.state_dict(),
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
                       learning_rate: float = 2e-5, warmup_steps: int = 10,
                       min_lr: float = 2e-6, eval_interval: int = 25, eval_iters: int = 5,
                       val_fraction: float = 0.1, seed: int = 42,
                       grad_clip: float = 1.0, gradient_accumulation_steps: int = 1,
                       checkpoints_dir: str = None, max_train_seconds: float = None,
                       early_stop_patience: int = None, prompt_style: str = "alpaca"):
    """Stage B, with the same safeguards Stage A needed (training/trainer.py):
    gradient clipping, an fp16 loss scaler, the LR applied before the step it
    belongs to, a best-checkpoint save during the run, and a wall-clock budget.

    learning_rate defaults to 2e-5, not the 1e-4 this used to use: fine-tuning
    at 3x the pretraining peak LR (3e-5) risks undoing what pretraining
    learned."""
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

    if device == "cuda" and torch.cuda.device_count() > 1:
        print(f"Using DataParallel across {torch.cuda.device_count()} GPUs")
        model = torch.nn.DataParallel(model)

    records = load_instruction_records(data_path)
    if not records:
        raise RuntimeError(f"No usable instruction records in {data_path}")

    split_at = max(1, int(len(records) * (1 - val_fraction)))
    train_records, val_records = records[:split_at], records[split_at:]
    effective_seq = min(seq_len, config.block_size)
    train_ds = InstructionDataset(train_records, effective_seq, seed=seed,
                                   prompt_style=prompt_style)
    val_ds = InstructionDataset(val_records or train_records[-1:], effective_seq, seed=seed + 1,
                                 prompt_style=prompt_style)

    print(f"Instruction data: {data_path}")
    print(f"  train: {train_ds.stats()}")
    print(f"  val:   {val_ds.stats()}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.95),
                                   weight_decay=0.1, eps=1e-9)

    # fp16 needs loss scaling or small gradients underflow to zero; bf16 and
    # fp32 do not, hence the dtype gate (same rule as training/trainer.py).
    scaler = torch.amp.GradScaler("cuda", enabled=(dtype == "float16"))
    print(f"grad_clip={grad_clip} | grad_accum={gradient_accumulation_steps} | "
          f"GradScaler enabled={scaler.is_enabled()} (dtype={dtype})")
    print(f"learning_rate={learning_rate:.2e} min_lr={min_lr:.2e} warmup_steps={warmup_steps}")

    hyperparams = {
        "max_steps": max_steps, "batch_size": batch_size, "seq_len": effective_seq,
        "learning_rate": learning_rate, "warmup_steps": warmup_steps, "min_lr": min_lr,
        "grad_clip": grad_clip, "gradient_accumulation_steps": gradient_accumulation_steps,
        "effective_batch": batch_size * gradient_accumulation_steps,
        "amp_dtype": dtype, "grad_scaler_enabled": bool(scaler.is_enabled()),
        "prompt_style": prompt_style,
        "base_checkpoint": base_checkpoint,
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
                    _, loss, _, _ = model(X, Y, return_logits=False)
                # DataParallel returns one loss per GPU, so this is a 2-element
                # tensor on Kaggle's T4 x2 and .item() would raise
                # "a Tensor with 2 elements cannot be converted to Scalar".
                losses.append(loss.mean().item() if loss.dim() > 0 else loss.item())
            out[name] = sum(losses) / len(losses)
        model.train()
        return out

    def lr_at(step):
        if step < warmup_steps:
            return learning_rate * (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
        return min_lr + (learning_rate - min_lr) * 0.5 * (1 + math.cos(math.pi * progress))

    model.train()
    tokens_processed = 0
    last_train_loss = last_val_loss = None
    best_val_loss = float("inf")
    best_ckpt_path = None
    evals_since_best = 0
    clipped_updates = nonfinite_events = 0
    last_grad_norm = float("nan")
    run_start = time.time()
    end_step = max_steps

    for step in tqdm(range(max_steps)):
        if max_train_seconds is not None and time.time() - run_start > max_train_seconds:
            end_step = step
            print(f"Time budget of {max_train_seconds:.0f}s reached at step {step}; stopping.")
            break

        if step % eval_interval == 0 and step != 0:
            losses = estimate()
            last_train_loss, last_val_loss = losses["train"], losses["val"]
            print(f"step {step}: train {last_train_loss:.4f}, val {last_val_loss:.4f} | "
                  f"lr {optimizer.param_groups[0]['lr']:.3e} | grad_norm {last_grad_norm:.3f} "
                  f"(clipped {clipped_updates}/{step}) | nonfinite {nonfinite_events} | "
                  f"scale {scaler.get_scale():.0f} | tokens {tokens_processed:,} | "
                  f"elapsed {time.time() - run_start:.0f}s")
            # Keep the best checkpoint during the run, not just the last one:
            # fine-tuning on a small set starts overfitting well before the
            # final step, and a crash used to lose everything.
            if last_val_loss < best_val_loss:
                best_val_loss = last_val_loss
                evals_since_best = 0
                best_ckpt_path, _ = save_instruction_checkpoint(
                    model, optimizer, config, step, last_train_loss, last_val_loss,
                    data_path, train_ds.stats(), base_checkpoint, seed, tokens_processed,
                    device, hyperparams, checkpoints_dir=checkpoints_dir)
                print(f"  new best val {best_val_loss:.4f} -> {best_ckpt_path}")
            else:
                evals_since_best += 1
                # Fine-tuning a small model on a small set overfits quickly:
                # train loss keeps falling while val rises. Once val has not
                # improved for `patience` evals the rest of the budget only
                # makes the model worse, so stop and keep the best checkpoint.
                if early_stop_patience and evals_since_best >= early_stop_patience:
                    end_step = step
                    print(f"Early stop: val has not improved for {evals_since_best} evals "
                          f"(best {best_val_loss:.4f}). Stopping at step {step}.")
                    break

        # This step's LR, set BEFORE the update it applies to (it used to be
        # assigned after optimizer.step(), so every update ran on the previous
        # step's LR and step 0 ran at the full LR with no warmup).
        lr = lr_at(step)
        for g in optimizer.param_groups:
            g["lr"] = lr

        for micro in range(gradient_accumulation_steps):
            X, Y = train_ds.get_batch(batch_size, device, device_type)
            with ctx:
                _, total_loss, _, _ = model(X, Y, return_logits=False)
                # DataParallel returns one loss per GPU.
                if total_loss.dim() > 0:
                    total_loss = total_loss.mean()
                total_loss = total_loss / gradient_accumulation_steps
            if not torch.isfinite(total_loss):
                raise RuntimeError(f"Non-finite loss at instruction step {step}. Stopping run.")
            scaler.scale(total_loss).backward()
            tokens_processed += X.numel()

        # Unscale before clipping, or the threshold would be compared against
        # gradients still multiplied by the scaler's factor.
        scaler.unscale_(optimizer)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        last_grad_norm = float(norm)
        if not math.isfinite(last_grad_norm):
            nonfinite_events += 1
        elif last_grad_norm > grad_clip:
            clipped_updates += 1
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

    losses = estimate()
    last_train_loss, last_val_loss = losses["train"], losses["val"]

    ckpt_path, _ = save_instruction_checkpoint(
        model, optimizer, config, end_step, last_train_loss, last_val_loss,
        data_path, train_ds.stats(), base_checkpoint, seed, tokens_processed, device, hyperparams,
        checkpoints_dir=checkpoints_dir,
    )
    print(f"Instruction tuning completed in {time.time() - run_start:.0f}s")
    print(f"Final train loss: {last_train_loss:.4f} | val loss: {last_val_loss:.4f}")
    print(f"Final checkpoint: {ckpt_path}")
    # Hand back whichever is actually better - the final save often is, since
    # the in-run "best" only sees the eval points before it.
    if best_ckpt_path and best_val_loss < last_val_loss:
        print(f"Best checkpoint (val {best_val_loss:.4f}): {best_ckpt_path}")
        chosen = best_ckpt_path
    else:
        chosen = ckpt_path
    # Return the bare model, not the multi-GPU wrapper: callers read
    # model.config, which DataParallel does not forward.
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    return model, config, chosen
