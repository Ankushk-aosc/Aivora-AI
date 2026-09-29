"""Validate Aivora's inference path, and measure what the model can actually do.

Model-only accuracy measured 0.88% on the benchmark. That number is only
evidence about the MODEL if the path around it is correct, so this script
checks the path first and records everything needed to tell the two apart:

  1. checkpoint identity  - step, sha256, config, that it is 247850 and not an
                            earlier checkpoint
  2. tokenizer identity   - encoding, vocab size, the special token the model
                            was trained with
  3. a fixed 30-question diagnostic - 10 definitions, 10 calculations,
                            10 explanations - recorded with prompt, token ids,
                            decoded output, stop reason, latency and every
                            sampling parameter
  4. a sampling matrix    - temperature x top_p x max_new_tokens, reporting
                            output length, premature-stop rate, empty rate and
                            usable-answer rate

It changes no weights and tunes nothing against any benchmark: usable-answer
rate here is a coarse mechanical check (enough words, not a loop, not an echo of
the question), not a grade.

usage:
    python scripts/validate_inference.py                 # everything
    python scripts/validate_inference.py --skip-matrix    # 1-3 only
"""

import argparse
import hashlib
import json
import os
import re
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

CHECKPOINT = os.path.join("checkpoints", "final", "checkpoint_247850.pt")
EXPECTED_STEP = 247850
OUT_DIR = "reports"
SEED = 1234

DEFINITIONS = [
    "What is EBITDA?", "What is revenue?", "What is net income?", "What is ROE?",
    "What is a balance sheet?", "What is gross margin?", "What is working capital?",
    "What is free cash flow?", "What is debt-to-equity?", "What is operating income?",
]

CALCULATIONS = [
    "Revenue = 1,000\nNet income = 100\n\nCalculate net margin.",
    "Net income = 200\nEquity = 1,000\n\nCalculate ROE.",
    "Revenue = 5,000\nCOGS = 3,000\n\nCalculate gross margin.",
    "Current assets = 800\nCurrent liabilities = 400\n\nCalculate the current ratio.",
    "Total debt = 600\nEquity = 1,200\n\nCalculate debt-to-equity.",
    "EBITDA = 400\nRevenue = 2,000\n\nCalculate the EBITDA margin.",
    "Net income = 150\nTotal assets = 3,000\n\nCalculate ROA.",
    "Operating cash flow = 900\nCapital expenditure = 300\n\nCalculate free cash flow.",
    "Net income = 250\nShares outstanding = 100\n\nCalculate EPS.",
    "EBIT = 500\nInterest expense = 100\n\nCalculate interest coverage.",
]

EXPLANATIONS = [
    "Why can EBITDA increase while operating cash flow decreases?",
    "What does a rising debt-to-equity ratio indicate?",
    "What is the difference between EBIT and EBITDA?",
    "Why is depreciation a non-cash expense?",
    "Why might a profitable company run out of cash?",
    "What does a current ratio below 1 suggest?",
    "Why do companies hold inventory?",
    "How does leverage affect return on equity?",
    "Why does revenue growth not always improve profit?",
    "What does negative free cash flow mean?",
]

DIAGNOSTIC = ([("definition", q) for q in DEFINITIONS]
              + [("calculation", q) for q in CALCULATIONS]
              + [("explanation", q) for q in EXPLANATIONS])

# The configuration the server uses (app/backend/server.py load_checkpoint and
# chat_service.DEFAULT_GENERATION).
PRODUCTION = {"temperature": 0.7, "top_k": 40, "top_p": 0.9,
              "repetition_penalty": 1.3, "max_new_tokens": 40}


def sha256(path, chunk=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_identity(path):
    """Part 1: is this the completed checkpoint, and what was it trained as?"""
    import torch

    meta_path = path.rsplit(".pt", 1)[0] + ".json"
    metadata = {}
    if os.path.exists(meta_path):
        with open(meta_path, encoding="utf-8") as handle:
            metadata = json.load(handle)

    payload = torch.load(path, map_location="cpu")
    identity = {
        "path": path,
        "sha256": sha256(path),
        "step_in_checkpoint": payload.get("step"),
        "expected_step": EXPECTED_STEP,
        "step_matches": payload.get("step") == EXPECTED_STEP,
        "best_val_loss": payload.get("best_val_loss"),
        "tokens_processed": payload.get("tokens_processed"),
        "sidecar_present": bool(metadata),
        "model_config": metadata.get("model_config"),
        "effective_hparams": metadata.get("effective_hparams"),
        "state_dict_tensors": len(payload.get("model_state_dict", payload)),
    }
    del payload
    return identity


def tokenizer_identity(model_config):
    """Part 1: tokenizer, vocab and the special token the model actually has."""
    from data_sources.tokenizer import ENCODING_NAME, get_encoding

    enc = get_encoding()
    eot = enc.eot_token
    vocab_in_config = (model_config or {}).get("vocab_size")
    return {
        "encoding": ENCODING_NAME,
        "tokenizer_vocab_size": enc.n_vocab,
        "model_vocab_size": vocab_in_config,
        "vocab_matches": vocab_in_config == enc.n_vocab,
        "eot_token_id": eot,
        "eot_in_model_vocab": bool(vocab_in_config) and eot < vocab_in_config,
        # GPT-2 BPE has no BOS or PAD distinct from <|endoftext|>; training
        # concatenated documents separated by it, so it is the only special token.
        "bos_token": None,
        "pad_token": None,
        "special_tokens": {"<|endoftext|>": eot},
    }


LOOP_RE = re.compile(r"\b(\w+)\b(?:\s+\1\b){2,}", re.IGNORECASE)


def looks_usable(question, text):
    """A coarse mechanical check, deliberately not a grade.

    Usable means: it produced a sentence's worth of words, it is not a loop, and
    it is not simply echoing the question back.
    """
    if not text or not text.strip():
        return False, "empty"
    words = text.split()
    if len(words) < 5:
        return False, "too short"
    if LOOP_RE.search(text):
        return False, "repeats a word"
    if "Question:" in text:
        return False, "starts a new question"
    asked = {w.lower().strip("?.,") for w in question.split() if len(w) > 3}
    produced = {w.lower().strip("?.,") for w in words if len(w) > 3}
    if asked and produced and produced <= asked:
        return False, "echoes the question only"
    return True, "usable"


def generate_once(model, enc, question, config, device="cpu"):
    import torch

    prompt = f"Question: {question}\nAnswer:"
    ids = enc.encode_ordinary(prompt)
    context = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
    trace = {}
    torch.manual_seed(SEED)
    started = time.time()
    with torch.no_grad():
        out = model.generate(
            context, config["max_new_tokens"],
            temperature=config["temperature"], top_k=config.get("top_k"),
            top_p=config.get("top_p"),
            repetition_penalty=config.get("repetition_penalty", 1.0),
            stop_on_repetition=config.get("stop_on_repetition", True),
            trace=trace,
        )
    elapsed = time.time() - started
    generated_ids = out[0, len(ids):].tolist()
    text = enc.decode(generated_ids).strip()
    usable, reason = looks_usable(question, text)
    return {
        "question": question,
        "prompt": prompt,
        "prompt_token_ids": ids,
        "prompt_tokens": len(ids),
        "generated_token_ids": generated_ids,
        "generated_tokens": len(generated_ids),
        "decoded": text,
        "decoded_tokens": [enc.decode([t]) for t in generated_ids[:24]],
        "contains_eot": enc.eot_token in generated_ids,
        "stop_reason": trace.get("stop_reason"),
        "stop_step": trace.get("stop_step"),
        "stop_detail": trace.get("stop_detail"),
        "seconds": round(elapsed, 2),
        "config": dict(config),
        "seed": SEED,
        "usable": usable,
        "usable_reason": reason,
    }


def run_diagnostic(model, enc, config, label):
    records = []
    for index, (category, question) in enumerate(DIAGNOSTIC, 1):
        record = generate_once(model, enc, question, config)
        record["category"] = category
        records.append(record)
        print(f"  [{index:2}/{len(DIAGNOSTIC)}] {record['generated_tokens']:>3} tok "
              f"{record['stop_reason']:<16} {record['decoded'][:58]!r}")
    return {"label": label, "config": dict(config), "records": records,
            "summary": summarise_records(records)}


def summarise_records(records):
    total = len(records)
    lengths = [r["generated_tokens"] for r in records]
    premature = [r for r in records
                 if r["stop_reason"] in ("repeated_bigram", "repeated_token")
                 and r["generated_tokens"] < 20]
    return {
        "examples": total,
        "mean_generated_tokens": round(sum(lengths) / total, 1) if total else 0,
        "min_generated_tokens": min(lengths) if lengths else 0,
        "premature_stop_rate": round(100.0 * len(premature) / total, 1) if total else 0,
        "empty_rate": round(100.0 * sum(1 for r in records if not r["decoded"]) / total, 1)
        if total else 0,
        "usable_rate": round(100.0 * sum(1 for r in records if r["usable"]) / total, 1)
        if total else 0,
        "eot_emitted_rate": round(100.0 * sum(1 for r in records if r["contains_eot"]) / total, 1)
        if total else 0,
        "stop_reasons": {reason: sum(1 for r in records if r["stop_reason"] == reason)
                         for reason in {r["stop_reason"] for r in records}},
        "mean_seconds": round(sum(r["seconds"] for r in records) / total, 2) if total else 0,
    }


def run_matrix(model, enc, questions):
    """Part 4: does reasonable decoding produce meaningful output?"""
    results = []
    for temperature in (0.0, 0.3, 0.5, 0.7):
        for top_p in (0.8, 0.9, 0.95):
            config = {"temperature": max(temperature, 1e-5),
                      "top_k": 1 if temperature == 0.0 else 40,
                      "top_p": None if temperature == 0.0 else top_p,
                      "repetition_penalty": 1.3, "max_new_tokens": 128}
            records = [generate_once(model, enc, q, config) for q in questions]
            entry = {"temperature": temperature, "top_p": top_p,
                     "max_new_tokens": 128, "summary": summarise_records(records)}
            results.append(entry)
            print(f"  t={temperature} top_p={top_p}: "
                  f"len {entry['summary']['mean_generated_tokens']:>5}  "
                  f"premature {entry['summary']['premature_stop_rate']:>5}%  "
                  f"usable {entry['summary']['usable_rate']:>5}%")

    for max_new_tokens in (64, 128, 256):
        config = {"temperature": 0.7, "top_k": 40, "top_p": 0.9,
                  "repetition_penalty": 1.3, "max_new_tokens": max_new_tokens}
        records = [generate_once(model, enc, q, config) for q in questions]
        results.append({"temperature": 0.7, "top_p": 0.9,
                        "max_new_tokens": max_new_tokens,
                        "summary": summarise_records(records)})
        print(f"  max_new_tokens={max_new_tokens}: usable "
              f"{results[-1]['summary']['usable_rate']}%")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=CHECKPOINT)
    parser.add_argument("--skip-matrix", action="store_true")
    parser.add_argument("--matrix-questions", type=int, default=10)
    parser.add_argument("--out", default=OUT_DIR)
    parser.add_argument("--tag", default="", help="suffix for the output files")
    args = parser.parse_args()

    from data_sources.tokenizer import get_encoding
    from inference import load_model_for_inference

    print("checkpoint identity ...")
    identity = checkpoint_identity(args.checkpoint)
    print(json.dumps({k: v for k, v in identity.items()
                      if k not in ("model_config", "effective_hparams")}, indent=2))
    if not identity["step_matches"]:
        print(f"WARNING: checkpoint step is {identity['step_in_checkpoint']}, "
              f"expected {EXPECTED_STEP}")

    tokenizer = tokenizer_identity(identity.get("model_config"))
    print(json.dumps(tokenizer, indent=2))

    model, config = load_model_for_inference(args.checkpoint, device="cpu")
    enc = get_encoding()
    report = {
        "checkpoint": identity,
        "tokenizer": tokenizer,
        "model": {
            "parameters": sum(p.numel() for p in model.parameters()),
            "block_size": config.block_size,
            "n_layer": config.n_layer,
            "n_experts": config.n_experts,
            "dtype": str(next(model.parameters()).dtype),
            "device": "cpu",
            "kv_cache": False,
        },
        "production_config": PRODUCTION,
    }

    print("\n30-question diagnostic, production configuration ...")
    report["diagnostic_production"] = run_diagnostic(model, enc, PRODUCTION,
                                                     "production configuration")

    print("\n30-question diagnostic, greedy ...")
    greedy = {"temperature": 1e-5, "top_k": 1, "top_p": None,
              "repetition_penalty": 1.0, "max_new_tokens": 40}
    report["diagnostic_greedy"] = run_diagnostic(model, enc, greedy, "greedy")

    if not args.skip_matrix:
        print("\nsampling matrix ...")
        questions = [q for _, q in DIAGNOSTIC[:args.matrix_questions]]
        report["sampling_matrix"] = run_matrix(model, enc, questions)

    os.makedirs(args.out, exist_ok=True)
    tag = f"_{args.tag}" if args.tag else ""
    path = os.path.join(args.out, f"model_inference_validation{tag}.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(f"\nwrote {path}")

    for name in ("diagnostic_production", "diagnostic_greedy"):
        print(f"\n{name}: {json.dumps(report[name]['summary'], indent=2)}")


if __name__ == "__main__":
    main()
