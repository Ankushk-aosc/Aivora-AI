"""Model serving and versioning (spec Part 2 / continuous-engineering
priority #2).

Adds what the existing checkpoint system didn't have: content-addressed
integrity verification (SHA256, so a corrupted or swapped checkpoint
file is detected rather than silently loaded) and an explicit "which
version is the active/serving one per stage" pointer - real gaps in
the previous "load whatever path you're given" behavior, not
reimplementations of what training/trainer.py's checkpointing already
does well (full metadata, resume support - those are reused as-is).
"""

import datetime
import hashlib
import json
import os

REGISTRY_PATH = os.path.join("checkpoints", "model_registry.json")
STAGES = ("base", "financial", "instruction")


def _sha256_file(path: str, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _read_registry() -> dict:
    if not os.path.exists(REGISTRY_PATH):
        return {"versions": [], "active": {}}
    with open(REGISTRY_PATH) as f:
        return json.load(f)


def _write_registry(data: dict):
    os.makedirs(os.path.dirname(REGISTRY_PATH), exist_ok=True)
    with open(REGISTRY_PATH, "w") as f:
        json.dump(data, f, indent=2)


def register_checkpoint(checkpoint_path: str, stage: str, set_active: bool = True) -> dict:
    """Compute a real SHA256 of the actual .pt file (not the metadata) and
    record it in the registry, along with whatever step/loss info is in
    the sidecar .json. Returns the registry entry."""
    if stage not in STAGES:
        raise ValueError(f"Unknown stage '{stage}'. Use one of {STAGES}")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(checkpoint_path)

    checksum = _sha256_file(checkpoint_path)
    meta_path = checkpoint_path.rsplit(".pt", 1)[0] + ".json"
    checkpoint_meta = {}
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            checkpoint_meta = json.load(f)

    registry = _read_registry()

    existing = [v for v in registry["versions"] if v["checksum"] == checksum]
    if existing:
        entry = existing[0]
    else:
        version_number = sum(1 for v in registry["versions"] if v["stage"] == stage) + 1
        entry = {
            "version": f"{stage}-v{version_number}",
            "stage": stage,
            "path": checkpoint_path.replace("\\", "/"),
            "checksum": checksum,
            "size_bytes": os.path.getsize(checkpoint_path),
            "step": checkpoint_meta.get("step"),
            "train_loss": checkpoint_meta.get("train_loss"),
            "val_loss": checkpoint_meta.get("val_loss"),
            "registered_at": datetime.datetime.utcnow().isoformat() + "Z",
        }
        registry["versions"].append(entry)

    if set_active:
        registry["active"][stage] = entry["version"]

    _write_registry(registry)
    return entry


def verify_integrity(version: str) -> dict:
    """Re-hash the checkpoint file on disk and compare to the registered
    checksum. Detects a corrupted, truncated, or swapped file - this is
    the actual point of registering a checksum, not just recording one."""
    registry = _read_registry()
    entry = next((v for v in registry["versions"] if v["version"] == version), None)
    if entry is None:
        raise KeyError(f"Unknown version '{version}'")

    if not os.path.exists(entry["path"]):
        return {"version": version, "valid": False, "reason": f"File missing: {entry['path']}"}

    current_checksum = _sha256_file(entry["path"])
    valid = current_checksum == entry["checksum"]
    return {
        "version": version, "valid": valid,
        "registered_checksum": entry["checksum"], "current_checksum": current_checksum,
        "reason": None if valid else "Checksum mismatch - file changed since registration",
    }


def get_active(stage: str) -> dict:
    registry = _read_registry()
    version = registry["active"].get(stage)
    if version is None:
        return None
    return next((v for v in registry["versions"] if v["version"] == version), None)


def set_active(version: str):
    registry = _read_registry()
    entry = next((v for v in registry["versions"] if v["version"] == version), None)
    if entry is None:
        raise KeyError(f"Unknown version '{version}'")
    registry["active"][entry["stage"]] = version
    _write_registry(registry)
    return entry


def list_versions(stage: str = None) -> list:
    registry = _read_registry()
    versions = registry["versions"]
    if stage:
        versions = [v for v in versions if v["stage"] == stage]
    return versions


def registry_status() -> dict:
    registry = _read_registry()
    return {"versions": registry["versions"], "active": registry["active"]}
