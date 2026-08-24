"""Deterministic, leakage-safe train/validation split.

Uses a seeded hash of each record's content so the split is reproducible
without needing to hold the whole dataset in memory, and so duplicate
records always land in the same split (a record can never appear in both
train and validation).
"""

import hashlib

from .cleaning import content_hash

DEFAULT_SEED = 42


def _bucket(text: str, seed: int) -> float:
    h = hashlib.sha256(f"{seed}:{content_hash(text)}".encode("utf-8")).hexdigest()
    # Use the first 8 hex chars as a uniform float in [0, 1).
    return int(h[:8], 16) / 0xFFFFFFFF


def assign_split(text: str, train_fraction: float = 0.95, seed: int = DEFAULT_SEED) -> str:
    return "train" if _bucket(text, seed) < train_fraction else "validation"


def split_stream(records, train_fraction: float = 0.95, seed: int = DEFAULT_SEED):
    """Generator yielding (split_name, record) for each input record."""
    for record in records:
        split = assign_split(record["text"], train_fraction=train_fraction, seed=seed)
        yield split, record


def check_no_leakage(train_hashes: set, eval_hashes: set) -> list:
    """Return the set of content hashes present in both train and eval."""
    return sorted(train_hashes & eval_hashes)
