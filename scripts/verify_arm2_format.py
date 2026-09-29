"""Phase 3: inspect REAL prepared arm-2 examples before spending GPU hours.

tests/test_training_format.py checks the logic on synthetic records. This runs
the actual preparation path - stream a dataset, clean it, tokenize it, write
shards, read them back through the training loader - and decodes what came out,
because the brief's instruction is right: do not rely on unit tests alone.

The eight things it verifies, from the brief:

  1. question tokens are masked
  2. the first answer token IS trained
  3. the final EOS is trained
  4. no answer token is accidentally masked
  5. no off-by-one error
  6. prompt loss is not being optimised
  7. EOS is included in the completion target
  8. document separator behaviour is correct

    python scripts/verify_arm2_format.py --tokens 20000
"""

import argparse
import glob
import json
import os
import shutil
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

SCRATCH = os.path.join("data", "shards_arm2_check")
FAILURES = []


def check(name, ok, detail=""):
    if not ok:
        FAILURES.append(name)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" - {detail}" if detail else ""))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="financial_instruction_alpaca")
    parser.add_argument("--tokens", type=int, default=20000)
    parser.add_argument("--keep", action="store_true", help="keep the scratch shards")
    args = parser.parse_args()

    from data_sources.prepare import prepare_dataset
    from data_sources.shard_writer import load_shard_index
    from data_sources.tokenizer import get_encoding
    from data_sources.training_format import IGNORE_INDEX, TrainingFormat
    from training.data_loader import ShardedDataset

    enc = get_encoding()
    if os.path.exists(SCRATCH):
        shutil.rmtree(SCRATCH, ignore_errors=True)

    fmt = TrainingFormat(document_separator=True, template=True, mask_prompt=True)
    print(f"preparing {args.dataset} with {json.dumps(fmt.describe())}\n")
    stats = prepare_dataset(args.dataset, max_tokens=args.tokens, seed=1234,
                            shards_root=SCRATCH, training_format=fmt)
    print(f"prepared {stats['train_tokens_used']:,} train tokens from "
          f"{stats['train_records_used']:,} records\n")

    shard_dir = os.path.join(SCRATCH, args.dataset, "train")
    index = load_shard_index(shard_dir)
    check("index records a mask file", bool(index.get("has_masks")), json.dumps(index)[:120])

    shard_path = os.path.join(shard_dir, index["shards"][0]["file"])
    mask_path = os.path.join(shard_dir, index["shards"][0]["mask_file"])
    tokens = np.fromfile(shard_path, dtype=np.uint16)
    masks = np.fromfile(mask_path, dtype=np.uint8)

    check("tokens and mask are the same length", len(tokens) == len(masks),
          f"{len(tokens)} vs {len(masks)}")
    separators = np.flatnonzero(tokens == enc.eot_token)
    check("8. separators present in real prepared data", len(separators) > 1,
          f"{len(separators)} separators in {len(tokens):,} tokens")

    # --- decode the first three real documents and show them ---------------
    print("=" * 72)
    bounds = [0] + (separators + 1).tolist()
    for n in range(min(3, len(bounds) - 1)):
        start, end = bounds[n], bounds[n + 1]
        doc_tokens, doc_mask = tokens[start:end], masks[start:end]
        text = enc.decode(doc_tokens.tolist())
        trained = enc.decode(doc_tokens[doc_mask == 1].tolist())
        ignored = enc.decode(doc_tokens[doc_mask == 0].tolist())
        print(f"\n--- real prepared document {n + 1} "
              f"({len(doc_tokens)} tokens, {int((doc_mask == 0).sum())} masked) ---")
        print(f"FULL     : {text[:240]!r}")
        print(f"IGNORED  : {ignored[:160]!r}")
        print(f"TRAINED  : {trained[:160]!r}")
    print("=" * 72)

    # --- the eight checks, on the first document ---------------------------
    first_end = int(separators[0]) + 1
    doc_tokens, doc_mask = tokens[:first_end], masks[:first_end]
    text = enc.decode(doc_tokens.tolist())

    check("template applied in real data",
          text.startswith("Question:") and "\nAnswer:" in text, text[:70])
    check("1. question tokens are masked", doc_mask[0] == 0 and 0 in set(doc_mask.tolist()),
          f"first mask values {doc_mask[:6].tolist()}")
    check("6. prompt loss is not optimised: the ignored span IS the question",
          enc.decode(doc_tokens[doc_mask == 0].tolist()).startswith("Question:"),
          enc.decode(doc_tokens[doc_mask == 0].tolist())[:60])

    boundary = int(np.flatnonzero(doc_mask == 1)[0])
    check("2. the first answer token is trained", doc_mask[boundary] == 1,
          f"boundary at {boundary}: {enc.decode([int(doc_tokens[boundary])])!r}")
    check("4. no answer token is masked after the boundary",
          bool((doc_mask[boundary:] == 1).all()),
          f"{int((doc_mask[boundary:] == 0).sum())} masked after the boundary")
    check("3/7. the final EOS is trained",
          doc_tokens[-1] == enc.eot_token and doc_mask[-1] == 1,
          f"last token {int(doc_tokens[-1])}, mask {int(doc_mask[-1])}")

    # --- 5. off-by-one, through the real training loader -------------------
    dataset = ShardedDataset([shard_dir])
    check("loader sees the masks", dataset.has_masks)

    block = min(256, first_end - 2)
    chunk, chunk_mask = tokens[:block + 1], masks[:block + 1]
    x = chunk[:-1].astype(np.int64)
    labels = chunk[1:].astype(np.int64).copy()
    labels[chunk_mask[1:] == 0] = IGNORE_INDEX

    ignored_count = int((labels == IGNORE_INDEX).sum())
    prompt_len = boundary
    check("5. exactly prompt_len - 1 labels ignored after the shift",
          ignored_count == prompt_len - 1,
          f"{ignored_count} ignored, prompt is {prompt_len} tokens")

    first_kept = int(np.argmax(labels != IGNORE_INDEX))
    check("5. first predicted label is the first answer token",
          int(labels[first_kept]) == int(doc_tokens[boundary]),
          f"{enc.decode([int(labels[first_kept])])!r} vs "
          f"{enc.decode([int(doc_tokens[boundary])])!r}")
    check("5. predicted FROM the last prompt token",
          int(x[first_kept]) == int(doc_tokens[boundary - 1]),
          f"input {enc.decode([int(x[first_kept])])!r}")

    # --- a non-instruction dataset must be separated but not masked --------
    print()
    plain_stats = prepare_dataset("fineweb_edu", max_tokens=8000, seed=1234,
                                  shards_root=SCRATCH, training_format=fmt)
    plain_dir = os.path.join(SCRATCH, "fineweb_edu", "train")
    plain_index = load_shard_index(plain_dir)
    plain_tokens = np.fromfile(os.path.join(plain_dir, plain_index["shards"][0]["file"]),
                               dtype=np.uint16)
    plain_mask = np.fromfile(os.path.join(plain_dir, plain_index["shards"][0]["mask_file"]),
                             dtype=np.uint8)
    check("plain text: separated", int((plain_tokens == enc.eot_token).sum()) > 0,
          f"{int((plain_tokens == enc.eot_token).sum())} separators in "
          f"{plain_stats['train_tokens_used']:,} tokens")
    check("plain text: nothing masked - there is no prompt to mask",
          int((plain_mask == 0).sum()) == 0, f"{int((plain_mask == 0).sum())} masked")

    del tokens, masks, plain_tokens, plain_mask, dataset
    if not args.keep:
        shutil.rmtree(SCRATCH, ignore_errors=True)

    print(f"\n{'ALL CHECKS PASSED' if not FAILURES else f'FAILED: {FAILURES}'}")
    sys.exit(1 if FAILURES else 0)


if __name__ == "__main__":
    main()
