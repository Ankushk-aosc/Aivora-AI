"""Financial instruction dataset (Part 16 / §22 Stage B).

Format:

    ### Instruction:
    Explain EBITDA.

    ### Response:
    EBITDA stands for ...

Supports {"instruction", "input", "output"} records. Loss is computed on
the response tokens only, so the model learns to answer rather than to
reproduce the prompt.
"""

import json
import os

import numpy as np
import torch

from data_sources.tokenizer import get_encoding

PROMPT_NO_INPUT = "### Instruction:\n{instruction}\n\n### Response:\n"
PROMPT_WITH_INPUT = "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"

IGNORE_INDEX = -1  # matches the model's F.cross_entropy(ignore_index=-1)


def format_prompt(record: dict) -> str:
    if record.get("input"):
        return PROMPT_WITH_INPUT.format(instruction=record["instruction"], input=record["input"])
    return PROMPT_NO_INPUT.format(instruction=record["instruction"])


def load_instruction_records(path: str):
    """Load a .jsonl or .json file of instruction records."""
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    records = []
    if path.endswith(".jsonl"):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    else:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        records = data if isinstance(data, list) else data.get("data", [])

    valid = []
    for r in records:
        if r.get("instruction") and r.get("output"):
            valid.append({
                "instruction": str(r["instruction"]).strip(),
                "input": str(r.get("input", "") or "").strip(),
                "output": str(r["output"]).strip(),
            })
    return valid


class InstructionDataset:
    """Tokenizes instruction records into fixed-length training examples."""

    def __init__(self, records, block_size: int, seed: int = 42):
        self.enc = get_encoding()
        self.block_size = block_size
        self.examples = []
        self.skipped = 0

        for record in records:
            prompt = format_prompt(record)
            prompt_ids = self.enc.encode_ordinary(prompt)
            answer_ids = self.enc.encode_ordinary(record["output"])
            ids = prompt_ids + answer_ids

            if len(ids) < 2:
                self.skipped += 1
                continue
            if len(ids) > block_size:
                ids = ids[:block_size]
                if len(prompt_ids) >= block_size - 1:
                    self.skipped += 1
                    continue

            # Mask the prompt so loss is only taken on the response.
            labels = list(ids)
            for i in range(min(len(prompt_ids), len(labels))):
                labels[i] = IGNORE_INDEX

            self.examples.append((ids, labels))

        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.examples)

    def get_batch(self, batch_size: int, device: str = "cpu", device_type: str = "cpu"):
        if not self.examples:
            raise RuntimeError("InstructionDataset is empty")

        picks = self.rng.integers(0, len(self.examples), size=batch_size)
        xs, ys = [], []
        for p in picks:
            ids, labels = self.examples[int(p)]
            # x = tokens[:-1], y = labels[1:]  (next-token prediction)
            x = ids[:-1]
            y = labels[1:]
            pad = self.block_size - 1 - len(x)
            if pad > 0:
                x = x + [0] * pad
                y = y + [IGNORE_INDEX] * pad
            xs.append(torch.tensor(x[:self.block_size - 1], dtype=torch.long))
            ys.append(torch.tensor(y[:self.block_size - 1], dtype=torch.long))

        X = torch.stack(xs)
        Y = torch.stack(ys)
        if device_type == "cuda":
            X, Y = X.pin_memory().to(device, non_blocking=True), Y.pin_memory().to(device, non_blocking=True)
        else:
            X, Y = X.to(device), Y.to(device)
        return X, Y

    def stats(self):
        lengths = [len(ids) for ids, _ in self.examples]
        if not lengths:
            return {"examples": 0, "skipped": self.skipped}
        return {
            "examples": len(self.examples),
            "skipped": self.skipped,
            "avg_tokens": round(sum(lengths) / len(lengths), 1),
            "min_tokens": min(lengths),
            "max_tokens": max(lengths),
            "total_tokens": sum(lengths),
        }
