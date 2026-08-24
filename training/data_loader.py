import os

import numpy as np
import torch

from data_sources.dataset_mixer import resolve_available_buckets
from data_sources.shard_writer import load_shard_index

DEFAULT_SHARDS_ROOT = os.path.join("data", "shards")


class ShardedDataset:
    """Memmaps only the shard(s) needed for a batch, never loading an
    entire split into RAM (Part 14)."""

    def __init__(self, shard_dirs):
        self.shards = []  # list of (path, token_count)
        for shard_dir in shard_dirs:
            index = load_shard_index(shard_dir)
            for s in index["shards"]:
                self.shards.append((os.path.join(shard_dir, s["file"]), s["tokens"]))
        self.total_tokens = sum(t for _, t in self.shards)

    def sample_block(self, block_size: int, rng: np.random.Generator):
        if not self.shards or self.total_tokens <= block_size:
            return None
        # Pick a shard weighted by its token count so bigger shards are
        # sampled proportionally more often.
        weights = np.array([t for _, t in self.shards], dtype=np.float64)
        weights = weights / weights.sum()
        path, n_tokens = self.shards[rng.choice(len(self.shards), p=weights)]
        if n_tokens <= block_size:
            return None
        data = np.memmap(path, dtype=np.uint16, mode="r")
        start = int(rng.integers(0, n_tokens - block_size))
        chunk = data[start:start + block_size + 1]
        return chunk


class MixedShardedLoader:
    """Combines multiple ShardedDatasets according to dataset_mix weights."""

    def __init__(self, shards_root: str, mix: dict, split: str, seed: int = 42):
        resolved = resolve_available_buckets(shards_root, mix, split=split)
        self.buckets = []  # list of (weight, ShardedDataset)
        for bucket, (weight, shard_dirs) in resolved.items():
            self.buckets.append((weight, ShardedDataset(shard_dirs)))
        self.rng = np.random.default_rng(seed)

    def get_batch(self, batch_size: int, block_size: int, device_type: str, device: str):
        weights = np.array([w for w, _ in self.buckets], dtype=np.float64)
        weights = weights / weights.sum()

        xs, ys = [], []
        attempts = 0
        while len(xs) < batch_size:
            attempts += 1
            if attempts > batch_size * 50:
                raise RuntimeError(
                    "Could not sample enough blocks - prepared shards may be smaller "
                    "than block_size. Prepare more tokens or lower block_size."
                )
            _, dataset = self.buckets[self.rng.choice(len(self.buckets), p=weights)]
            chunk = dataset.sample_block(block_size, self.rng)
            if chunk is None:
                continue
            xs.append(torch.from_numpy(chunk[:-1].astype(np.int64)))
            ys.append(torch.from_numpy(chunk[1:].astype(np.int64)))

        x = torch.stack(xs)
        y = torch.stack(ys)

        if device_type == "cuda":
            x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
        else:
            x, y = x.to(device), y.to(device)
        return x, y


def resolve_seq_len(config, seq_len=None):
    """Training sequence length may be shorter than the model's block_size
    (the architecture supports any t <= block_size). Presets use this to
    stay within the memory budget of small GPUs / CPU."""
    if seq_len is None:
        return config.block_size
    if seq_len > config.block_size:
        raise ValueError(
            f"seq_len={seq_len} exceeds model block_size={config.block_size}"
        )
    return seq_len


def get_batch(loaders, split, config, batch_size, device_type, device, seq_len=None):
    """loaders: dict {"train": MixedShardedLoader, "validation": MixedShardedLoader}"""
    return loaders[split].get_batch(
        batch_size, resolve_seq_len(config, seq_len), device_type, device
    )


def estimate_loss(model, loaders, config, eval_iters, batch_size, device_type, device, ctx,
                   seq_len=None):
    out = {}
    model.eval()
    with torch.inference_mode():
        for split, key in (("train", "train"), ("val", "validation")):
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                X, Y = get_batch(loaders, key, config, batch_size, device_type, device, seq_len)
                with ctx:
                    _, loss, _, _ = model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
    model.train()
    return out
