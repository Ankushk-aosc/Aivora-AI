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
from .notifier_utils import notify_checkpoint_saved, notify_training_failed

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
    # torch.cuda.is_bf16_supported() is not reliable across torch versions/
    # hardware (real bf16 tensor cores only exist from Ampere/compute
    # capability 8.0 onward; this call has returned True on pre-Ampere GPUs
    # in some torch builds, e.g. observed on a Tesla P100/compute 6.0 under
    # torch 2.7.1+cu118 - bf16 tensors work there but with no hardware
    # acceleration, silently ballooning memory use versus real bf16). Gate
    # on compute capability directly instead of trusting that function.
    if device_type == "cuda" and torch.cuda.get_device_capability(0)[0] >= 8:
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
                     seed, tokens_processed=0, device="cpu", checkpoints_dir=None,
                     protect_paths=(), effective_hparams=None):
    """checkpoints_dir lets a continuation run write to its own directory
    instead of checkpoints/base/, so a recovery run can never land on top of
    the known-good checkpoint it is being recovered from.

    protect_paths is a belt-and-braces guard: if the path about to be written
    resolves to one of these, refuse rather than overwrite. Filenames are
    keyed on step number so a collision already requires writing at the exact
    resumed step, but "already unlikely" is not the same as "cannot happen",
    and the file this protects is the only good artifact in the project."""
    target_dir = checkpoints_dir or CHECKPOINTS_DIR
    os.makedirs(target_dir, exist_ok=True)
    ckpt_path = os.path.join(target_dir, f"checkpoint_{step}.pt")
    meta_path = os.path.join(target_dir, f"checkpoint_{step}.json")

    protected = {os.path.realpath(p) for p in protect_paths if p}
    if os.path.realpath(ckpt_path) in protected:
        raise RuntimeError(
            f"Refusing to overwrite protected checkpoint {ckpt_path}. This is the "
            f"known-good recovery checkpoint; the run should be writing to a "
            f"separate continuation directory."
        )

    # Unwrap DataParallel so the saved state_dict has the same key names
    # (no "module." prefix) whether the run used 1 or several GPUs - keeps
    # every checkpoint resumable regardless of GPU count.
    raw_model = model.module if isinstance(model, torch.nn.DataParallel) else model
    torch.save({
        "model_state_dict": raw_model.state_dict(),
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
        # What the run ACTUALLY used. dataset_config["learning_rate"] above is
        # the preset's from-scratch value, which is NOT the LR a continuation
        # run trains at - checkpoint_17000 was trained at 3e-5 but recorded
        # 3e-4, i.e. the metadata named the exact setting that had destroyed
        # the model. Record the effective values so a checkpoint is honest
        # about its own provenance.
        "effective_hparams": effective_hparams or "Not recorded",
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
                 shards_root: str = os.path.join("data", "shards"), seed: int = 42,
                 max_steps_override: int = None, eval_interval_override: int = None,
                 batch_size_override: int = None, checkpoints_dir: str = None,
                 lr_horizon_override: int = None, divergence_val_threshold: float = None):
    """max_steps_override/eval_interval_override/batch_size_override let a
    caller run a short smoke test against a preset's real seq_len/
    dataset_mix without editing the preset yaml. max_steps_override is the
    number of ADDITIONAL steps to run past the resumed checkpoint's step
    (or from 0 if not resuming) - NOT an absolute target step - because the
    training loop is `range(start_step, max_iters)`, and an absolute small
    target below start_step would silently produce zero iterations."""
    preset = load_preset(preset_name)
    config = DeepSeekConfig.default()

    learning_rate = float(preset["learning_rate"])
    warmup_steps = int(preset["warmup_steps"])
    min_lr = float(preset["min_lr"])
    eval_interval = int(eval_interval_override) if eval_interval_override is not None else int(preset["eval_interval"])
    eval_iters = int(preset["eval_iters"])
    batch_size = int(batch_size_override) if batch_size_override is not None else int(preset["batch_size"])
    gradient_accumulation_steps = int(preset["gradient_accumulation_steps"])
    dataset_mix = preset["dataset_mix"]
    seq_len = preset.get("seq_len")  # None -> use full config.block_size

    # ---- Continuation-training (resume) stability controls -------------
    # A learning rate that is correct for training FROM SCRATCH is not
    # automatically correct for continuing an already-converged checkpoint.
    # A fresh model has large, well-conditioned gradients and sits far from
    # any minimum; a converged one sits in a sharp minimum and a large LR
    # ejects it. Measured on this project: resuming checkpoint_16000
    # (val 6.07) at peak lr=3e-4 diverged smoothly to val 53.90 by step
    # 20,000, with NO NaN/Inf at any point - ordinary gradient-descent
    # blowup, not a numerical fault.
    #
    # So a resume reads its LR from preset["continuation"] when present,
    # leaving the from-scratch schedule above untouched for fresh runs.
    is_continuation = resume is not None
    continuation = preset.get("continuation") or {}
    grad_clip = float(preset.get("grad_clip", 1.0))
    if is_continuation and continuation:
        learning_rate = float(continuation.get("learning_rate", learning_rate))
        min_lr = float(continuation.get("min_lr", min_lr))
        warmup_steps = int(continuation.get("warmup_steps", warmup_steps))
        grad_clip = float(continuation.get("grad_clip", grad_clip))

    device = detect_device()
    device_type = "cuda" if device == "cuda" else "cpu"
    ctx, dtype = build_context(device_type if device == "cuda" else "cpu")

    from .data_loader import resolve_seq_len
    effective_seq_len = resolve_seq_len(config, seq_len)
    print(f"Preset: {preset_name} | device: {device} | dtype: {dtype}")
    print(f"batch_size={batch_size} seq_len={effective_seq_len} "
          f"(model block_size={config.block_size}) grad_accum={gradient_accumulation_steps}")

    # GradScaler is ONLY correct for float16. bfloat16 has the same exponent
    # range as float32, so it does not need loss scaling, and float32 needs
    # none either - enabling it there would be wrong, not merely wasteful.
    # build_context() already picked the dtype from compute capability, so
    # gate on its answer rather than re-deriving it.
    use_amp_scaler = (dtype == "float16")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp_scaler)
    print(f"grad_clip={grad_clip} | GradScaler enabled={scaler.is_enabled()} "
          f"(dtype={dtype}; scaler applies to float16 only)")
    if is_continuation:
        print(f"CONTINUATION run: peak_lr={learning_rate:.3e} min_lr={min_lr:.3e} "
              f"warmup_steps={warmup_steps}"
              + (" (from preset['continuation'])" if continuation else
                 " (no 'continuation' block in preset - using the from-scratch LR!)"))
    else:
        print(f"FROM-SCRATCH run: peak_lr={learning_rate:.3e} min_lr={min_lr:.3e} "
              f"warmup_steps={warmup_steps}")

    def _effective_hparams():
        """The values this run actually trained with - see save_checkpoint."""
        return {
            "is_continuation": is_continuation,
            "learning_rate": learning_rate,
            "min_lr": min_lr,
            "warmup_steps": warmup_steps,
            "grad_clip": grad_clip,
            "batch_size": batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "effective_batch": batch_size * gradient_accumulation_steps,
            "amp_dtype": dtype,
            "grad_scaler_enabled": bool(scaler.is_enabled()),
            "resumed_from": resume,
        }

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

        # torch.optim.Optimizer.load_state_dict() restores param_groups, and
        # that includes the 'lr' the checkpoint was saved with - so the line
        # above silently reinstates the OLD run's learning rate over the one
        # AdamW was just constructed with. There is no torch LR scheduler in
        # this trainer (the schedule is the plain lr_at_step() function
        # below, which holds no state), so the optimizer's param_groups are
        # the only place a stale LR can hide. Overwrite it explicitly and say
        # so, rather than relying on the loop's first assignment to correct
        # it a step later.
        restored_lr = optimizer.param_groups[0].get("lr")
        for param_group in optimizer.param_groups:
            param_group["lr"] = learning_rate
        print(f"Optimizer state restored (Adam moments kept). LR carried in that "
              f"state was {restored_lr:.3e}; overridden to this run's "
              f"{learning_rate:.3e}.")

    if max_steps_override is not None:
        max_iters = start_step + int(max_steps_override)
        print(f"max_steps_override={max_steps_override}: running steps "
              f"{start_step}->{max_iters} ({max_steps_override} steps), not preset "
              f"max_steps={preset['max_steps']}. LR schedule below is computed "
              f"against this shorter horizon, not the preset's real one - "
              f"loss/LR numbers from this run are not meaningful for judging "
              f"training quality, only for smoke-testing throughput/memory.")
    else:
        max_iters = int(preset["max_steps"])

    # The cosine horizon. Normally this is just max_iters, but a SHORT
    # stability test needs to be told the real long-run horizon, otherwise it
    # silently tests the wrong learning rate: shortening max_iters compresses
    # the cosine, so a 1,000-step probe of a 495,700-step schedule would run
    # at ~1.02e-5 instead of the intended ~3.0e-5 and "pass" without ever
    # exercising the LR the real run would use.
    lr_horizon = int(lr_horizon_override) if lr_horizon_override is not None else max_iters
    if lr_horizon_override is not None:
        print(f"lr_horizon_override={lr_horizon}: the LR schedule is computed against "
              f"this horizon while the run itself stops at step {max_iters}, so a short "
              f"probe sees the same LR the full run would.")

    def base_lr_at(step):
        """The preset's own warmup-then-cosine schedule, as a function of
        absolute step."""
        if step < warmup_steps:
            return learning_rate * (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, (lr_horizon - warmup_steps))
        return min_lr + (learning_rate - min_lr) * 0.5 * (1 + math.cos(math.pi * progress))

    # Re-warm the learning rate after a resume.
    #
    # base_lr_at()'s warmup branch is keyed on the ABSOLUTE step, so a run
    # resumed past warmup_steps got no warmup at all - it jumped straight to
    # whatever the cosine says at that step. That is harmless when the
    # schedule is unchanged, but catastrophic when max_steps is raised
    # between runs, because raising max_steps stretches the cosine and moves
    # the LR at the resume point back up toward the peak.
    #
    # This is not hypothetical: resuming checkpoint_16000 (which finished its
    # own 16,000-step schedule at min_lr=1e-5) under the raised
    # max_steps=495700 put step 16000 at 2.993e-4 - a 29.9x jump, applied
    # with no warmup to a model that had converged to val_loss 6.07. It
    # diverged immediately, to train 17.71/val 18.69 by step 16500 and
    # 19.97/19.91 by step 17000 - worse than this model's ~14.1 random-init
    # loss, and still climbing.
    #
    # So: ramp from min_lr up to the scheduled LR over warmup_steps measured
    # FROM THE RESUME POINT. A fresh run (start_step == 0) is unaffected -
    # base_lr_at() already warms up from step 0 there.
    resumed_at = start_step if (resume is not None and start_step > 0) else None

    def lr_at_step(step):
        scheduled = base_lr_at(step)
        if resumed_at is None or step >= resumed_at + warmup_steps:
            return scheduled
        ramp = (step - resumed_at + 1) / max(1, warmup_steps)
        return min_lr + (scheduled - min_lr) * ramp

    if resumed_at is not None:
        print(f"Resume warmup: ramping LR from {min_lr:.3e} to the scheduled "
              f"{base_lr_at(resumed_at + warmup_steps):.3e} over {warmup_steps} steps "
              f"({resumed_at} -> {resumed_at + warmup_steps}). Without this the run "
              f"would start at {base_lr_at(resumed_at):.3e} with no warmup.")

    if device == "cuda" and torch.cuda.device_count() > 1:
        print(f"Using DataParallel across {torch.cuda.device_count()} GPUs")
        model = torch.nn.DataParallel(model)

    model.train()
    last_train_loss = None
    last_val_loss = None
    # Stability telemetry - reported on every eval line so a run can be
    # judged from the Kaggle log alone, without re-deriving it afterwards.
    MAX_CONSECUTIVE_SKIPS = 25
    consecutive_skips = 0
    nonfinite_grad_events = 0
    clipped_updates = 0
    total_updates = 0
    last_grad_norm_before = float("nan")
    last_grad_norm_after = float("nan")
    run_start = time.time()
    last_log_time = run_start
    last_log_tokens = tokens_processed

    for step in tqdm(range(start_step, max_iters)):
        try:
            if step % eval_interval == 0 and step != start_step:
                losses = estimate_loss(model, loaders, config, eval_iters, batch_size,
                                        device_type, device, ctx, seq_len=seq_len)
                last_train_loss, last_val_loss = losses["train"].item(), losses["val"].item()
                elapsed = time.time() - run_start
                now = time.time()
                interval_tok_s = (tokens_processed - last_log_tokens) / max(now - last_log_time, 1e-9)
                last_log_time, last_log_tokens = now, tokens_processed
                gpu_mem = (f"{torch.cuda.max_memory_allocated() / 1024 ** 3:.2f}GiB"
                           if device_type == "cuda" else "n/a")
                print(f"step {step}: train {last_train_loss:.4f}, val {last_val_loss:.4f} | "
                      f"lr {optimizer.param_groups[0]['lr']:.3e} | "
                      f"grad_norm {last_grad_norm_before:.3f}->{last_grad_norm_after:.3f} "
                      f"(clipped {clipped_updates}/{total_updates}) | "
                      f"nonfinite {nonfinite_grad_events} | scale {scaler.get_scale():.0f} | "
                      f"gpu_mem {gpu_mem} | "
                      f"tokens {tokens_processed:,} | {interval_tok_s:.0f} tok/s | "
                      f"elapsed {elapsed:.0f}s")
                if wandb:
                    wandb.log({"step": step, "train_loss": last_train_loss, "val_loss": last_val_loss,
                                "tokens_processed": tokens_processed})

                # Stop a stability probe the moment it is clearly diverging,
                # rather than paying for the whole budget to confirm it. The
                # previous run took ~52 GPU-minutes to reach val 53.90 when
                # the answer was already obvious by val 11.43 at step 16500.
                if (divergence_val_threshold is not None
                        and last_val_loss > divergence_val_threshold):
                    raise RuntimeError(
                        f"STABILITY TEST FAILED: val loss {last_val_loss:.4f} at step "
                        f"{step} exceeded the divergence threshold "
                        f"{divergence_val_threshold:.4f}. Stopping early to preserve GPU "
                        f"quota. Do NOT start a long run; lower the continuation "
                        f"learning_rate (currently {learning_rate:.3e}) and retest."
                    )

                if last_val_loss < best_val_loss:
                    best_val_loss = last_val_loss
                    ckpt_path, _ = save_checkpoint(model, optimizer, config, preset, step, last_train_loss,
                                                    last_val_loss, best_val_loss, seed, tokens_processed, device,
                                                    checkpoints_dir=checkpoints_dir,
                                                    protect_paths=(resume,),
                                                    effective_hparams=_effective_hparams())
                    notify_checkpoint_saved(step, max_iters, last_train_loss, last_val_loss,
                                             best_val_loss, tokens_processed, ckpt_path)

            # Set this step's LR BEFORE any optimizer.step() consumes it.
            # This assignment used to sit AFTER optimizer.step(), so every
            # update ran on the previous iteration's learning rate - an
            # off-by-one that is invisible on a smooth schedule but actively
            # wrong during a warmup ramp, which is exactly when the LR is
            # changing fastest.
            lr = lr_at_step(step)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            X, y = get_batch(loaders, "train", config, batch_size, device_type, device, seq_len)
            with ctx:
                _, total_loss, main_loss, mtp_loss = model(X, y, return_logits=False)
                # DataParallel gathers each replica's scalar loss into a
                # (num_gpus,) tensor; .mean() is a no-op on a single GPU/CPU.
                total_loss = total_loss.mean()
                main_loss = main_loss.mean() if main_loss is not None else None
                mtp_loss = mtp_loss.mean() if mtp_loss is not None else None
                loss = total_loss / gradient_accumulation_steps

            # Part 29: never silently train through NaN/Inf.
            if not torch.isfinite(total_loss):
                raise RuntimeError(
                    f"Non-finite loss at step {step}: total={total_loss.item()} "
                    f"main={main_loss.item() if main_loss is not None else None} "
                    f"mtp={mtp_loss.item() if mtp_loss is not None else None}. "
                    f"lr={optimizer.param_groups[0]['lr']}, batch={X.shape}. Stopping run."
                )

            # scaler.scale() is the identity when the scaler is disabled
            # (bf16/fp32), so this one path is correct for every precision.
            scaler.scale(loss).backward()
            tokens_processed += X.numel()

            if ((step + 1) % gradient_accumulation_steps == 0) or (step + 1 == max_iters):
                # Order matters and is the documented AMP recipe:
                #   unscale_ -> clip_grad_norm_ -> scaler.step -> scaler.update
                # Clipping BEFORE unscaling would clip the loss-scaled
                # gradients, i.e. clip against a threshold that is ~65536x
                # off and does nothing useful.
                scaler.unscale_(optimizer)
                # clip_grad_norm_ returns the total norm measured BEFORE
                # clipping, which is the number worth logging - the norm
                # after is just min(before, grad_clip).
                grad_norm_before = float(
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                )
                grad_norm_after = min(grad_norm_before, grad_clip)

                if not math.isfinite(grad_norm_before):
                    nonfinite_grad_events += 1
                    if scaler.is_enabled():
                        # Expected occasionally under float16: unscale_ has
                        # already flagged the overflow, so scaler.step() will
                        # skip this update and scaler.update() will back the
                        # scale off. Only a run that NEVER recovers is fatal.
                        consecutive_skips += 1
                        if consecutive_skips > MAX_CONSECUTIVE_SKIPS:
                            raise RuntimeError(
                                f"{consecutive_skips} consecutive non-finite gradients at "
                                f"step {step} (loss scale {scaler.get_scale()}). The scaler "
                                f"is not recovering. Stopping run."
                            )
                    else:
                        # No scaler to absorb it - keep the original
                        # never-train-through-NaN/Inf guarantee.
                        raise RuntimeError(
                            f"Non-finite gradient norm at step {step} with the GradScaler "
                            f"disabled (dtype={dtype}). Stopping run."
                        )
                else:
                    consecutive_skips = 0
                    last_grad_norm_before = grad_norm_before
                    last_grad_norm_after = grad_norm_after
                    if grad_norm_before > grad_clip:
                        clipped_updates += 1
                    total_updates += 1

                # No-op if unscale_ found inf/NaN; a plain optimizer.step()
                # when the scaler is disabled.
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            if wandb:
                wandb.log({
                    "step": step, "total_loss": total_loss.item(), "main_loss": main_loss.item(),
                    "mtp_loss": mtp_loss.item() if mtp_loss is not None else None, "learning_rate": lr,
                })
        except Exception as e:
            notify_training_failed(step, e)
            raise

    # Final measured losses before saving the last checkpoint.
    losses = estimate_loss(model, loaders, config, eval_iters, batch_size,
                            device_type, device, ctx, seq_len=seq_len)
    last_train_loss, last_val_loss = losses["train"].item(), losses["val"].item()
    best_val_loss = min(best_val_loss, last_val_loss)

    ckpt_path, meta_path = save_checkpoint(
        model, optimizer, config, preset, max_iters, last_train_loss, last_val_loss,
        best_val_loss, seed, tokens_processed, device,
        checkpoints_dir=checkpoints_dir, protect_paths=(resume,),
        effective_hparams=_effective_hparams(),
    )
    notify_checkpoint_saved(max_iters, max_iters, last_train_loss, last_val_loss,
                             best_val_loss, tokens_processed, ckpt_path)
    elapsed = time.time() - run_start
    print(f"Training completed in {elapsed:.0f}s | tokens processed: {tokens_processed:,}")
    print(f"Final train loss: {last_train_loss:.4f} | Final val loss: {last_val_loss:.4f}")
    print(f"Final checkpoint: {ckpt_path}")

    if wandb:
        wandb.finish()

    return model, config, ckpt_path
