"""Phase 0: record exactly what the frozen baseline consists of.

Everything a later run would need to prove it compared against the same thing:
commit, checkpoint bytes, tokenizer identity, corpus identity, benchmark bytes,
and the inference and generation configuration actually used.

Hashes are over file bytes, so a silent edit anywhere shows up as a different
digest. Nothing here modifies anything.

    python scripts/freeze_manifest.py
"""

import hashlib
import json
import os
import platform
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

TAG = "AIVORA_PRE_TRAINING_BASELINE"
OUT = os.path.join("reports", "FREEZE_MANIFEST.json")

CHECKPOINT = os.path.join("checkpoints", "final", "checkpoint_247850.pt")
CHECKPOINT_META = os.path.join("checkpoints", "final", "checkpoint_247850.json")
BENCHMARKS = [os.path.join("data", "benchmark", f)
              for f in ("dev.jsonl", "test.jsonl", "blind.jsonl")]
CODE_UNDER_FREEZE = [
    "models/model.py", "training/trainer.py", "training/data_loader.py",
    "inference/generator.py", "data_sources/tokenizer.py",
    "data_sources/prepare.py", "data_sources/shard_writer.py",
    "data_sources/training_format.py",
    "app/backend/services/generation.py", "app/backend/services/chat_service.py",
    "app/backend/services/financial_router.py",
    "app/backend/services/financial_values.py",
    "app/backend/services/question_focus.py",
    "app/backend/services/analysis_patterns.py",
    "evaluation/benchmark.py", "evaluation/financial_metrics.py",
    "evaluation/external.py", "tools/financial_calculator.py",
    "tools/derived_calculations.py",
]


def sha256_file(path, chunk=8 * 1024 * 1024):
    if not os.path.exists(path):
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def git(*args):
    try:
        return subprocess.check_output(["git", *args], stderr=subprocess.DEVNULL,
                                       text=True).strip()
    except Exception:
        return None


def tokenizer_identity():
    from data_sources.tokenizer import ENCODING_NAME, get_encoding

    enc = get_encoding()
    # Hash the vocabulary itself, not just its name: a different BPE file with
    # the same name would silently change every token id.
    sample = "".join(enc.decode([i]) for i in range(0, 50257, 617))
    return {
        "encoding": ENCODING_NAME,
        "n_vocab": enc.n_vocab,
        "eot_token": enc.eot_token,
        "vocab_probe_sha256": hashlib.sha256(sample.encode("utf-8")).hexdigest(),
        "probe_description": "every 617th token decoded and concatenated",
    }


def corpus_identity():
    """The corpus the checkpoint was trained on, from its own embedded manifest."""
    if not os.path.exists(CHECKPOINT_META):
        return {"available": False}
    with open(CHECKPOINT_META, encoding="utf-8") as handle:
        meta = json.load(handle)
    manifest = meta.get("dataset_manifest") or {}
    datasets = manifest.get("datasets", manifest if isinstance(manifest, list) else [])
    unique = sum(d.get("train_tokens_used", 0) for d in datasets)
    processed = meta.get("tokens_processed")
    return {
        "available": True,
        "datasets": len(datasets),
        "unique_train_tokens": unique,
        "tokens_processed": processed,
        "epochs_over_corpus": round(processed / unique, 2) if unique and processed else None,
        "unique_tokens_per_parameter": round(unique / 101_723_264, 3) if unique else None,
        "manifest_sha256": hashlib.sha256(
            json.dumps(datasets, sort_keys=True).encode("utf-8")).hexdigest(),
        "dataset_mix": meta.get("dataset_config", {}).get("dataset_mix"),
        "effective_hparams": meta.get("effective_hparams"),
    }


def main():
    from app.backend.services.chat_service import DEFAULT_GENERATION

    manifest = {
        "tag": TAG,
        "tag_exists": bool(git("rev-parse", "-q", "--verify", f"refs/tags/{TAG}")),
        "tag_commit": git("rev-list", "-n", "1", TAG),
        "head_commit": git("rev-parse", "HEAD"),
        "head_describes_tag": git("describe", "--tags", "--abbrev=0") == TAG,
        "working_tree_clean": not git("status", "--porcelain"),
        "uncommitted_paths": (git("status", "--porcelain") or "").splitlines(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "checkpoint": {
            "path": CHECKPOINT,
            "sha256": sha256_file(CHECKPOINT),
            "bytes": os.path.getsize(CHECKPOINT) if os.path.exists(CHECKPOINT) else None,
            "metadata_sha256": sha256_file(CHECKPOINT_META),
        },
        "tokenizer": tokenizer_identity(),
        "training_data": corpus_identity(),
        "benchmarks": {os.path.basename(p): {"sha256": sha256_file(p),
                                             "bytes": os.path.getsize(p)
                                             if os.path.exists(p) else None}
                       for p in BENCHMARKS},
        "code_sha256": {p: sha256_file(p) for p in CODE_UNDER_FREEZE},
        "inference_configuration": {
            "device": "cpu (no local GPU)",
            "dtype": "float32",
            "kv_cache": False,
            "context_length": 1024,
            "prompt_template": "Question: {query}\\nAnswer:",
            "eos_token_id": 50256,
            "repetition_window_covers": "generated tokens only (fixed)",
        },
        "generation_configuration": dict(DEFAULT_GENERATION),
        "served_max_new_tokens": 40,
        "measured_baseline": {
            "dev": {"system": 100.0, "model_only": 1.64},
            "hidden": {"system": 96.08, "model_only": 3.43},
            "blind_prose": {"system": 13.89, "model_only": 7.64},
            "hidden_model_only": {
                "correct": 7, "total": 204, "hallucination_rate": 96.39,
                "abstention_accuracy": 0.0, "extraction": "0/25",
                "reasoning": "0/25", "eos_stops": 0, "max_token_stops": 201,
            },
            "four_group_hidden": {
                "system_correct_model_correct": 7,
                "system_correct_model_wrong": 189,
                "system_wrong_model_correct": 0,
                "system_wrong_model_wrong": 8,
            },
        },
        "rules": [
            "the baseline is never overwritten",
            "hidden/test data is never modified",
            "no benchmark answer is ever hardcoded",
            "calculator and retrieval output is never reported as model capability",
        ],
    }

    os.makedirs("reports", exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    print(json.dumps({k: v for k, v in manifest.items()
                      if k not in ("code_sha256", "uncommitted_paths")}, indent=2)[:2600])
    print(f"\nuncommitted paths preserved: {len(manifest['uncommitted_paths'])}")
    for path in manifest["uncommitted_paths"][:12]:
        print(f"  {path}")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
