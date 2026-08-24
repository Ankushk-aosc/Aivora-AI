"""High-level `dataset prepare` pipeline: stream -> clean/dedup -> leakage-safe
split -> tokenize -> shard -> record provenance in the manifest."""

import datetime
import logging
import os

from .tokenizer import get_encoding
from .cleaning import clean_and_filter
from .dataset_registry import VERIFIED, get_entry
from .huggingface_loader import stream_records
from .manifest import record_preparation
from .shard_writer import ShardWriter
from .splitter import split_stream

logger = logging.getLogger(__name__)

DEFAULT_SHARDS_ROOT = os.path.join("data", "shards")


def prepare_dataset(
    name: str,
    max_tokens: int,
    max_records=None,
    shards_root: str = DEFAULT_SHARDS_ROOT,
    train_fraction: float = 0.95,
    seed: int = 42,
    min_text_length: int = 20,
    allow_unverified: bool = False,
):
    entry = get_entry(name)

    if entry.verification_status != VERIFIED and not allow_unverified:
        raise RuntimeError(
            f"Refusing to prepare '{name}': status is {entry.verification_status!r}. "
            f"{entry.notes} Pass allow_unverified=True only if you have confirmed the "
            f"license and loader yourself."
        )

    raw_records = stream_records(entry, max_records=max_records, max_tokens=max_tokens)

    clean_stats = {}
    cleaned_records = clean_and_filter(raw_records, min_length=min_text_length, stats=clean_stats)

    enc = get_encoding()
    train_writer = ShardWriter(os.path.join(shards_root, name, "train"))
    val_writer = ShardWriter(os.path.join(shards_root, name, "validation"))

    train_tokens = 0
    val_tokens = 0
    train_records = 0
    val_records = 0

    for split, record in split_stream(cleaned_records, train_fraction=train_fraction, seed=seed):
        ids = enc.encode_ordinary(record["text"])
        if not ids:
            continue
        if split == "train":
            train_writer.write_tokens(ids)
            train_tokens += len(ids)
            train_records += 1
        else:
            val_writer.write_tokens(ids)
            val_tokens += len(ids)
            val_records += 1

    train_writer.close()
    val_writer.close()

    summary = {
        "name": entry.name,
        "source": "Hugging Face",
        "dataset_id": entry.hf_id,
        "subset": entry.subset,
        "revision": entry.revision,
        "license": entry.license,
        "source_url": entry.source_url,
        "verification_status": entry.verification_status,
        "split_requested": entry.split,
        "train_fraction": train_fraction,
        "seed": seed,
        "records_seen": clean_stats.get("seen", 0),
        "records_removed_invalid": clean_stats.get("removed_invalid", 0),
        "records_removed_duplicate": clean_stats.get("removed_duplicate", 0),
        "train_records_used": train_records,
        "validation_records_used": val_records,
        "train_tokens_used": train_tokens,
        "validation_tokens_used": val_tokens,
        "max_tokens_requested": max_tokens,
        "max_records_requested": max_records,
        "download_date": datetime.datetime.utcnow().isoformat() + "Z",
    }

    record_preparation(summary)
    logger.info("Prepared %s: %s train tokens, %s validation tokens", name, train_tokens, val_tokens)
    return summary
