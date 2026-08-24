"""Experiment tracking (Part 36 / §45).

Each experiment is a directory under experiments/ holding experiment.json,
metrics.json, model_config.json and dataset_config.json. Every value is
copied from what actually ran - nothing is defaulted in silently.
"""

import datetime
import json
import os
import platform
import subprocess

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXPERIMENTS_DIR = os.path.join(ROOT, "experiments")


def _git_revision():
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                              capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or "Not available"
    except Exception:
        return "Not available"


def environment() -> dict:
    env = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "platform": platform.platform(),
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda or "Not available",
        "repository_revision": _git_revision(),
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        env["gpu"] = props.name
        env["gpu_memory_gb"] = round(props.total_memory / 1024 ** 3, 2)
    else:
        env["gpu"] = "Not available"
        env["gpu_memory_gb"] = "Not available"
    return env


def record_experiment(name, model_config, dataset_config, metrics,
                       checkpoint=None, seed=None, notes=""):
    """Write one experiment record. Returns the directory it was written to."""
    exp_dir = os.path.join(EXPERIMENTS_DIR, name)
    os.makedirs(exp_dir, exist_ok=True)

    with open(os.path.join(exp_dir, "experiment.json"), "w") as f:
        json.dump({
            "name": name,
            "seed": seed,
            "checkpoint": checkpoint,
            "environment": environment(),
            "notes": notes,
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        }, f, indent=2)

    with open(os.path.join(exp_dir, "model_config.json"), "w") as f:
        json.dump(model_config, f, indent=2)

    with open(os.path.join(exp_dir, "dataset_config.json"), "w") as f:
        json.dump(dataset_config, f, indent=2)

    with open(os.path.join(exp_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    return exp_dir


def record_from_checkpoint(name, checkpoint_path, eval_results=None, notes=""):
    """Build an experiment record from a checkpoint's own metadata, so the
    recorded values are exactly what training measured."""
    meta_path = checkpoint_path.rsplit(".pt", 1)[0] + ".json"
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"No checkpoint metadata at {meta_path}")
    with open(meta_path) as f:
        meta = json.load(f)

    metrics = {
        "step": meta.get("step"),
        "train_loss": meta.get("train_loss"),
        "val_loss": meta.get("val_loss"),
        "best_val_loss": meta.get("best_val_loss"),
        "tokens_processed": meta.get("tokens_processed"),
        "evaluation": eval_results or "Not available",
    }
    dataset_config = {
        "dataset_config": meta.get("dataset_config"),
        "dataset_manifest": meta.get("dataset_manifest"),
        "train_tokens_budget": meta.get("train_tokens_budget"),
        "validation_tokens_budget": meta.get("validation_tokens_budget"),
    }
    return record_experiment(
        name, meta.get("model_config", {}), dataset_config, metrics,
        checkpoint=checkpoint_path, seed=meta.get("seed"), notes=notes,
    )


def list_experiments():
    if not os.path.isdir(EXPERIMENTS_DIR):
        return []
    out = []
    for entry in sorted(os.listdir(EXPERIMENTS_DIR)):
        path = os.path.join(EXPERIMENTS_DIR, entry, "experiment.json")
        if os.path.exists(path):
            with open(path) as f:
                out.append(json.load(f))
    return out
