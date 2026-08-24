"""Tokenizer statistics for prepared shard data (Part 13)."""

import json
import os

import numpy as np

from .tokenizer import get_encoding

FINANCIAL_TERMS = [
    "ebitda", "revenue", "profit", "margin", "equity", "asset", "liability",
    "cash flow", "eps", "p/e", "roe", "roic", "fcf", "yoy", "qoq", "dividend",
    "balance sheet", "income statement", "gross margin", "operating margin",
]


def compute_shard_stats(shard_dir: str) -> dict:
    """Read every shard_*.bin under shard_dir and compute token statistics."""
    enc = get_encoding()

    shard_files = sorted(
        f for f in os.listdir(shard_dir) if f.startswith("shard_") and f.endswith(".bin")
    ) if os.path.isdir(shard_dir) else []

    if not shard_files:
        return {
            "shard_dir": shard_dir,
            "total_tokens": 0,
            "unique_tokens": 0,
            "num_shards": 0,
            "note": "No shards found.",
        }

    total_tokens = 0
    unique_token_ids = set()
    problematic_records = 0

    for fname in shard_files:
        path = os.path.join(shard_dir, fname)
        arr = np.fromfile(path, dtype=np.uint16)
        total_tokens += len(arr)
        # Sample unique-token tracking on a bounded number of ids to stay cheap.
        unique_token_ids.update(np.unique(arr).tolist())
        if len(arr) == 0:
            problematic_records += 1

    avg_len = total_tokens / max(1, len(shard_files))

    financial_freq = {}
    if total_tokens > 0:
        for term in FINANCIAL_TERMS:
            ids = enc.encode_ordinary(term)
            financial_freq[term] = len(ids)

    return {
        "shard_dir": shard_dir,
        "num_shards": len(shard_files),
        "total_tokens": total_tokens,
        "unique_tokens": len(unique_token_ids),
        "avg_tokens_per_shard": avg_len,
        "vocab_size": enc.n_vocab,
        "financial_term_subword_lengths": financial_freq,
        "problematic_shards": problematic_records,
    }


def print_stats(stats: dict):
    print(json.dumps(stats, indent=2))
