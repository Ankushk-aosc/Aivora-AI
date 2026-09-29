"""Experiment E1's format changes must be exactly right, or the run is wasted.

A misaligned mask trains on the wrong tokens and nothing visibly fails - the
loss simply means something different. These tests pin the three changes:

  1. arm 1 (all flags off) produces byte-identical shards to the old pipeline
  2. the separator token is appended AND trained on
  3. the prompt mask lines up with the prompt tokens after the label shift

Run: deepseek_env/Scripts/python.exe tests/test_training_format.py
"""
import json
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_sources.shard_writer import ShardWriter, load_shard_index  # noqa: E402
from data_sources.tokenizer import get_encoding  # noqa: E402
from data_sources.training_format import (  # noqa: E402
    IGNORE_INDEX, PROMPT_TEMPLATE, TrainingFormat,
)
from training.data_loader import ShardedDataset  # noqa: E402

PASSED, FAILED = [], []
ENC = get_encoding()

RECORD = {"prompt": "What is EBITDA?",
          "completion": "Earnings before interest, taxes, depreciation and amortization."}
PLAIN = {"text": "Inflation is a general rise in prices."}


def check(name, ok, detail=""):
    (PASSED if ok else FAILED).append(name)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" - {detail}" if detail else ""))


def test_arm1_is_unchanged():
    """Everything off must reproduce the old behaviour exactly."""
    fmt = TrainingFormat()
    ids, mask = fmt.encode(PLAIN)
    check("arm 1: plain text tokenized as before",
          ids == ENC.encode_ordinary(PLAIN["text"]), f"{ids[:6]}")
    check("arm 1: no separator appended", ENC.eot_token not in ids)
    check("arm 1: mask is all ones", set(mask) == {1})

    # An instruction record with no template is joined as the old loader did.
    ids, _ = fmt.encode(RECORD)
    text = ENC.decode(ids)
    check("arm 1: no template applied",
          "Question:" not in text and "Answer:" not in text, text[:50])

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
        writer = ShardWriter(d, shard_tokens=1000)
        writer.write_tokens([1, 2, 3])
        writer.close()
        index = load_shard_index(d)
        check("arm 1: index.json has no mask fields",
              "has_masks" not in index and "mask_file" not in index["shards"][0],
              str(index))
        check("arm 1: no mask files written",
              not [f for f in os.listdir(d) if f.startswith("mask_")],
              str(os.listdir(d)))


def test_separator():
    fmt = TrainingFormat(document_separator=True)
    ids, mask = fmt.encode(PLAIN)
    check("separator: appended to the document", ids[-1] == ENC.eot_token, str(ids[-3:]))
    check("separator: only one", ids.count(ENC.eot_token) == 1)
    check("separator: trained on, not ignored", mask[-1] == 1)
    check("separator: mask and ids stay the same length", len(ids) == len(mask))

    # An empty record must not produce a lone separator.
    ids, mask = fmt.encode({"text": ""})
    check("separator: empty record produces nothing", ids == [] and mask == [])


def test_template_and_mask():
    fmt = TrainingFormat(template=True, mask_prompt=True, document_separator=True)
    ids, mask = fmt.encode(RECORD)
    text = ENC.decode(ids)

    check("template: applied", text.startswith("Question: What is EBITDA?\nAnswer:"),
          text[:60])
    check("template: matches the serving prompt exactly",
          PROMPT_TEMPLATE.format(prompt=RECORD["prompt"]) in text,
          PROMPT_TEMPLATE)

    prompt_ids = ENC.encode_ordinary(PROMPT_TEMPLATE.format(prompt=RECORD["prompt"]))
    check("mask: exactly the prompt tokens are ignored",
          mask[:len(prompt_ids)] == [0] * len(prompt_ids)
          and set(mask[len(prompt_ids):]) == {1},
          f"prompt {len(prompt_ids)} tokens, mask starts {mask[:4]}")
    check("mask: the completion is trained on",
          sum(mask) == len(ids) - len(prompt_ids), f"{sum(mask)} of {len(ids)}")

    # Template without masking: the format changes, the loss does not.
    ids2, mask2 = TrainingFormat(template=True).encode(RECORD)
    check("template without mask_prompt trains on everything", set(mask2) == {1})

    # A record with no completion cannot be templated and must not be masked
    # into oblivion.
    ids3, mask3 = fmt.encode(PLAIN)
    check("plain text is never masked", set(mask3) == {1}, str(set(mask3)))


def test_mask_survives_the_label_shift():
    """The mask must still cover the prompt AFTER labels are shifted by one.

    This is the failure that would be invisible: an off-by-one here trains the
    model on the last prompt token and ignores the first answer token, and the
    loss still looks reasonable.
    """
    fmt = TrainingFormat(template=True, mask_prompt=True, document_separator=True)
    ids, mask = fmt.encode(RECORD)
    prompt_len = len(ENC.encode_ordinary(PROMPT_TEMPLATE.format(prompt=RECORD["prompt"])))

    # ignore_cleanup_errors: numpy memmaps keep the file handle open on
    # Windows until they are garbage collected, and the directory cannot be
    # removed while they are.
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
        writer = ShardWriter(d, shard_tokens=10_000, write_masks=True)
        # Pad so a block can be sampled deterministically from offset 0.
        writer.write_tokens(ids, mask)
        writer.write_tokens([ENC.eot_token] * 200, [1] * 200)
        writer.close()

        dataset = ShardedDataset([d])
        check("loader: mask files are discovered", dataset.has_masks)

        path, n_tokens, mask_path = dataset.shards[0]
        tokens = np.memmap(path, dtype=np.uint16, mode="r")
        masks = np.memmap(mask_path, dtype=np.uint8, mode="r")
        block = 64
        chunk = tokens[0:block + 1]
        chunk_mask = masks[0:block + 1]

        x = chunk[:-1].astype(np.int64)
        labels = chunk[1:].astype(np.int64).copy()
        labels[chunk_mask[1:] == 0] = IGNORE_INDEX

        ignored = int((labels == IGNORE_INDEX).sum())
        check("shift: exactly prompt_len - 1 labels ignored",
              ignored == prompt_len - 1, f"{ignored} ignored, prompt {prompt_len}")

        # The first non-ignored label must be the first completion token, and
        # the input at that position must be the last prompt token.
        first_kept = int(np.argmax(labels != IGNORE_INDEX))
        check("shift: first predicted token is the first completion token",
              labels[first_kept] == ids[prompt_len],
              f"{ENC.decode([int(labels[first_kept])])!r} vs "
              f"{ENC.decode([ids[prompt_len]])!r}")
        check("shift: it is predicted FROM the last prompt token",
              x[first_kept] == ids[prompt_len - 1],
              f"{ENC.decode([int(x[first_kept])])!r}")

        # The separator must be predicted, since learning to stop is the point.
        sep_positions = [i for i, t in enumerate(labels) if t == ENC.eot_token]
        check("shift: the separator is a predicted label",
              bool(sep_positions) and labels[sep_positions[0]] == ENC.eot_token,
              str(sep_positions[:3]))


def test_describe_is_recorded():
    fmt = TrainingFormat(document_separator=True, template=True, mask_prompt=True)
    described = fmt.describe()
    check("describe: records every flag and the template",
          described["document_separator"] and described["template"]
          and described["mask_prompt"] and described["prompt_template"],
          json.dumps(described))


def main():
    test_arm1_is_unchanged()
    test_separator()
    test_template_and_mask()
    test_mask_survives_the_label_shift()
    test_describe_is_recorded()
    print(f"\n{len(PASSED)}/{len(PASSED) + len(FAILED)} passed")
    sys.exit(1 if FAILED else 0)


if __name__ == "__main__":
    main()
