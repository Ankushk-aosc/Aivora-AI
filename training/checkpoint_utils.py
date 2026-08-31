"""Checkpoint discovery + architecture-compatibility helpers.

training/trainer.py's save_checkpoint/load_checkpoint already handle writing
and loading a single known checkpoint path. What's been missing is a reusable
way to find "the checkpoint to resume from" among several candidates and
confirm it actually matches the current model before trusting it - logic that
previously only existed copy-pasted inline in
training/kaggle/Aivora_Kaggle_Training.ipynb (cell 9). This module is that
logic pulled out into one tested place instead of duplicated per-notebook.
"""

import glob
import os
import re

import torch

CHECKPOINT_RE = re.compile(r"checkpoint_(\d+)\.pt$")


def checkpoint_step_number(path: str) -> int:
    """Extract the step number from a 'checkpoint_<step>.pt' filename."""
    match = CHECKPOINT_RE.search(path)
    if not match:
        raise ValueError(f"'{path}' does not match the 'checkpoint_<step>.pt' naming convention")
    return int(match.group(1))


def find_latest_checkpoint(search_dirs, pattern: str = "checkpoint_*.pt"):
    """Return the highest-step checkpoint_*.pt under search_dirs (str or list
    of str), searching recursively, or None if none are found."""
    if isinstance(search_dirs, str):
        search_dirs = [search_dirs]

    candidates = []
    for d in search_dirs:
        candidates.extend(glob.glob(os.path.join(d, pattern)))
        candidates.extend(glob.glob(os.path.join(d, "**", pattern), recursive=True))
    candidates = sorted(set(candidates))

    if not candidates:
        return None
    return max(candidates, key=checkpoint_step_number)


def check_architecture_compatibility(checkpoint_path: str, model: torch.nn.Module) -> list:
    """Compare a checkpoint's state dict against model's current state dict.
    Returns a list of human-readable mismatches (missing keys, shape
    mismatches); an empty list means the checkpoint is compatible."""
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    ckpt_state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    ref_state = model.state_dict()

    mismatches = []
    for key, ref_tensor in ref_state.items():
        if key not in ckpt_state:
            mismatches.append(f"missing key: {key}")
        elif ckpt_state[key].shape != ref_tensor.shape:
            mismatches.append(
                f"shape mismatch on {key}: checkpoint has {ckpt_state[key].shape}, "
                f"current model expects {ref_tensor.shape}"
            )
    return mismatches


def resolve_resume_checkpoint(search_dirs, model: torch.nn.Module, required: bool = True):
    """Find the latest checkpoint under search_dirs and verify it's
    architecture-compatible with model before returning its path.

    Raises RuntimeError (STATUS = BLOCKED, matching this project's other hard
    gates) if required=True and no checkpoint is found, or if the latest one
    found doesn't match the current model. If required=False and none is
    found, returns None instead of raising, for callers doing a fresh run.
    """
    latest = find_latest_checkpoint(search_dirs)
    if latest is None:
        if required:
            raise RuntimeError(
                f"STATUS = BLOCKED: no checkpoint_*.pt found under {search_dirs}."
            )
        return None

    mismatches = check_architecture_compatibility(latest, model)
    if mismatches:
        raise RuntimeError(
            f"STATUS = BLOCKED: checkpoint '{latest}' is not architecture-compatible "
            f"with the current model config: {mismatches[:5]}"
        )
    return latest
