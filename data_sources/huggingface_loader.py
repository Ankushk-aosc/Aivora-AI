"""Streaming loader for Hugging Face datasets.

Never materializes a full dataset in memory or on disk: it streams
records and stops as soon as max_records or max_tokens is reached,
whichever comes first.
"""

import logging

from datasets import load_dataset

from .dataset_registry import DatasetEntry
from .tokenizer import get_encoding as _get_encoding

logger = logging.getLogger(__name__)


def _extract_text(entry: DatasetEntry, example: dict) -> str:
    """Build a plain-text string from a record.

    Falls back to entry.fields_used (joined) when the record has no plain
    "text" field, which is the case for QA/instruction-style datasets.
    """
    if isinstance(example.get("text"), str) and example["text"]:
        return example["text"]

    parts = []
    for f in entry.fields_used:
        value = example.get(f)
        if isinstance(value, str) and value:
            parts.append(value)
    return "\n".join(parts)


# Field names holding the request and the response in instruction-style
# datasets. Used only to expose the structure; the joined "text" is unchanged.
_PROMPT_FIELDS = ("instruction", "question", "prompt", "query", "input", "context")
_COMPLETION_FIELDS = ("output", "response", "answer", "completion", "text_output")


def _extract_pair(entry: DatasetEntry, example: dict):
    """(prompt, completion) when the record has that shape, else (None, None).

    The old pipeline joined these fields with newlines and trained on the
    result, so nothing distinguished the question from the answer. Exposing the
    pair lets data_sources/training_format.py apply a template and a loss mask
    (experiment E1), while callers that only read "text" are unaffected.
    """
    fields = set(entry.fields_used or ())
    prompt_parts, completion = [], None
    for name in _PROMPT_FIELDS:
        if name in fields:
            value = example.get(name)
            if isinstance(value, str) and value.strip():
                prompt_parts.append(value.strip())
    for name in _COMPLETION_FIELDS:
        if name in fields:
            value = example.get(name)
            if isinstance(value, str) and value.strip():
                completion = value.strip()
                break
    if prompt_parts and completion:
        return "\n".join(prompt_parts), completion
    return None, None


def stream_records(
    entry: DatasetEntry,
    max_records=None,
    max_tokens=None,
):
    """Yield dicts {"text": str, "tokens": int} from `entry`, streaming.

    Stops as soon as either budget is exhausted. Either budget may be
    None to leave it unconstrained (not recommended for fineweb_edu-scale
    sources).
    """
    if entry.data_files:
        # Script-based repo with an auto-converted Parquet branch: read the
        # Parquet files directly rather than executing the loading script.
        logger.info("Streaming %s from Parquet files %s", entry.hf_id, entry.data_files)
        dataset = load_dataset(
            "parquet", data_files=entry.data_files, split="train", streaming=True
        )
    else:
        load_kwargs = {"split": entry.split, "streaming": True}
        if entry.subset:
            load_kwargs["name"] = entry.subset
        if entry.revision:
            load_kwargs["revision"] = entry.revision

        logger.info(
            "Streaming %s (subset=%s, split=%s, revision=%s)",
            entry.hf_id, entry.subset, entry.split, entry.revision,
        )
        dataset = load_dataset(entry.hf_id, **load_kwargs)

    enc = _get_encoding()
    records_seen = 0
    tokens_seen = 0

    for example in dataset:
        text = _extract_text(entry, example)
        if not text:
            continue

        n_tokens = len(enc.encode_ordinary(text))
        if n_tokens == 0:
            continue

        record = {"text": text, "tokens": n_tokens}
        prompt, completion = _extract_pair(entry, example)
        if prompt and completion:
            record["prompt"], record["completion"] = prompt, completion
        yield record

        records_seen += 1
        tokens_seen += n_tokens

        if max_records is not None and records_seen >= max_records:
            logger.info("Reached max_records=%s, stopping stream.", max_records)
            break
        if max_tokens is not None and tokens_seen >= max_tokens:
            logger.info("Reached max_tokens=%s, stopping stream.", max_tokens)
            break

    logger.info("Streamed %s records / %s tokens from %s", records_seen, tokens_seen, entry.hf_id)
