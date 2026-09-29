"""Writes tokenized text into fixed-size binary shards instead of one
monolithic train.bin/validation.bin, so the data loader only ever needs
to memmap the shard(s) it is currently reading (Part 14)."""

import json
import os

import numpy as np

from .tokenizer import get_encoding

DEFAULT_SHARD_TOKENS = 1_000_000  # tokens per shard file
DTYPE = np.uint16  # GPT-2 vocab (50257) fits in uint16
MASK_DTYPE = np.uint8  # 1 = train on this token, 0 = ignore it


class ShardWriter:
    def __init__(self, out_dir: str, shard_tokens: int = DEFAULT_SHARD_TOKENS,
                 write_masks: bool = False):
        self.out_dir = out_dir
        self.shard_tokens = shard_tokens
        # Experiment E1: a parallel uint8 stream marking which tokens carry
        # loss. Off by default, so existing shards and their index.json are
        # byte-identical to before.
        self.write_masks = write_masks
        os.makedirs(out_dir, exist_ok=True)
        self._buffer = []
        self._mask_buffer = []
        self._buffer_len = 0
        self._shard_index = 0
        self._shards = []  # list of {"file": ..., "tokens": ...}
        self.total_tokens = 0
        self.masked_tokens = 0

    def _flush(self, force: bool = False):
        while self._buffer_len >= self.shard_tokens or (force and self._buffer_len > 0):
            take = min(self.shard_tokens, self._buffer_len)
            arr = np.concatenate(self._buffer) if len(self._buffer) > 1 else self._buffer[0]
            chunk, rest = arr[:take], arr[take:]

            fname = f"shard_{self._shard_index:03d}.bin"
            path = os.path.join(self.out_dir, fname)
            chunk.astype(DTYPE).tofile(path)

            shard_record = {"file": fname, "tokens": int(len(chunk))}

            if self.write_masks:
                mask_arr = (np.concatenate(self._mask_buffer)
                            if len(self._mask_buffer) > 1 else self._mask_buffer[0])
                mask_chunk, mask_rest = mask_arr[:take], mask_arr[take:]
                mask_name = f"mask_{self._shard_index:03d}.bin"
                mask_chunk.astype(MASK_DTYPE).tofile(
                    os.path.join(self.out_dir, mask_name))
                shard_record["mask_file"] = mask_name
                self.masked_tokens += int((mask_chunk == 0).sum())
                self._mask_buffer = [mask_rest] if len(mask_rest) > 0 else []

            self._shards.append(shard_record)
            self.total_tokens += len(chunk)
            self._shard_index += 1

            self._buffer = [rest] if len(rest) > 0 else []
            self._buffer_len = len(rest)

            if not force:
                break

    def write_tokens(self, token_ids, mask=None):
        arr = np.array(token_ids, dtype=DTYPE)
        if len(arr) == 0:
            return
        if self.write_masks:
            if mask is None:
                mask = [1] * len(arr)
            if len(mask) != len(arr):
                raise ValueError(
                    f"mask length {len(mask)} does not match {len(arr)} tokens - "
                    "a misaligned mask would silently train on the wrong tokens")
            self._mask_buffer.append(np.array(mask, dtype=MASK_DTYPE))
        self._buffer.append(arr)
        self._buffer_len += len(arr)
        self._flush(force=False)

    def close(self):
        self._flush(force=True)
        index_path = os.path.join(self.out_dir, "index.json")
        index = {"shards": self._shards, "total_tokens": self.total_tokens}
        if self.write_masks:
            index["has_masks"] = True
            index["masked_tokens"] = self.masked_tokens
        with open(index_path, "w") as f:
            json.dump(index, f, indent=2)
        return self.total_tokens


def load_shard_index(shard_dir: str) -> dict:
    index_path = os.path.join(shard_dir, "index.json")
    if not os.path.exists(index_path):
        return {"shards": [], "total_tokens": 0}
    with open(index_path) as f:
        return json.load(f)


def tokenize_and_write(text_records, out_dir: str, shard_tokens: int = DEFAULT_SHARD_TOKENS,
                        max_tokens=None):
    """Tokenize a stream of {"text": ...} records and write them to shards.

    Stops once max_tokens is reached (if given).
    """
    enc = get_encoding()
    writer = ShardWriter(out_dir, shard_tokens=shard_tokens)

    for record in text_records:
        if max_tokens is not None and writer.total_tokens + writer._buffer_len >= max_tokens:
            break
        ids = enc.encode_ordinary(record["text"])
        if not ids:
            continue
        writer.write_tokens(ids)

    total = writer.close()
    return total
