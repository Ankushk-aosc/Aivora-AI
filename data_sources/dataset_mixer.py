"""Configurable dataset mixing (Part 9).

Maps the six dataset_mix buckets used in configs/*.yaml to the shard
directories actually available on disk, validates weights sum to 1.0,
and samples shards proportionally to build training batches.
"""

import os

from .dataset_registry import list_entries

TOLERANCE = 1e-6

# Maps a dataset_mix bucket name -> the category tag used in the dataset
# registry (data_sources/dataset_registry.py DatasetEntry.category).
BUCKET_TO_CATEGORY = {
    "fineweb_edu": "general",
    "financial_text": "financial_text",
    "financial_qa": "financial_qa",
    "financial_reports": "financial_reports",
    "financial_reasoning": "financial_reasoning",
    "financial_instruction": "financial_instruction",
}


def validate_mix(mix: dict):
    total = sum(mix.values())
    if abs(total - 1.0) > TOLERANCE:
        raise ValueError(
            f"dataset_mix weights must sum to 1.0, got {total:.6f}: {mix}"
        )
    unknown = set(mix) - set(BUCKET_TO_CATEGORY)
    if unknown:
        raise ValueError(f"Unknown dataset_mix buckets: {sorted(unknown)}")


def _bucket_shard_dirs(shards_root: str, bucket: str, split: str):
    """Shard dirs for every registered dataset whose category matches this
    bucket, e.g. data/shards/fineweb_edu/train, data/shards/tinystories/train
    for the "fineweb_edu" bucket (category "general")."""
    category = BUCKET_TO_CATEGORY[bucket]
    dirs = []
    for entry in list_entries(category=category):
        shard_dir = os.path.join(shards_root, entry.name, split)
        if os.path.exists(os.path.join(shard_dir, "index.json")):
            dirs.append(shard_dir)
    return dirs


def resolve_available_buckets(shards_root: str, mix: dict, split: str = "train") -> dict:
    """Return {bucket: (weight, [shard_dirs])} for buckets that actually
    have prepared shards, with weights renormalized to sum to 1.0. Buckets
    with no prepared data are dropped rather than silently training on
    nothing.
    """
    validate_mix(mix)
    available = {}
    for bucket, weight in mix.items():
        dirs = _bucket_shard_dirs(shards_root, bucket, split)
        if dirs:
            available[bucket] = (weight, dirs)

    if not available:
        raise RuntimeError(
            f"None of the dataset_mix buckets have prepared shards under {shards_root}. "
            "Run `python main.py dataset prepare --name <dataset> --max-tokens N` first."
        )

    total = sum(w for w, _ in available.values())
    return {bucket: (w / total, dirs) for bucket, (w, dirs) in available.items()}
