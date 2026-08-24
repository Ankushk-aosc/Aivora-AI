"""Dataset provenance manifest (Part 37): data/dataset_manifest.json."""

import json
import os

DEFAULT_MANIFEST_PATH = os.path.join("data", "dataset_manifest.json")


def read_manifest(path: str = DEFAULT_MANIFEST_PATH) -> dict:
    if not os.path.exists(path):
        return {"datasets": []}
    with open(path) as f:
        return json.load(f)


def write_manifest(manifest: dict, path: str = DEFAULT_MANIFEST_PATH):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)


def record_preparation(entry_record: dict, path: str = DEFAULT_MANIFEST_PATH):
    """Append (or replace, if the same name+timestamp-less run already
    exists) a provenance record for a `dataset prepare` run."""
    manifest = read_manifest(path)
    manifest["datasets"].append(entry_record)
    write_manifest(manifest, path)
