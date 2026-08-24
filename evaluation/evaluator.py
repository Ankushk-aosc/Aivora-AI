"""Financial LLM POC Evaluation (Part 17 / §33).

Runs the model over held-out evaluation sets in data/evaluation/ and
reports measured accuracy per category. These files are never written
into data/shards/, so they are never part of the training mixture; a
leakage check against the prepared training shards is available via
check_leakage().

This is NOT a benchmark of general intelligence. It is labelled
"Financial LLM POC Evaluation" throughout.
"""

import json
import os

import torch

from data_sources.tokenizer import get_encoding
from evaluation.financial_metrics import (
    aggregate, exact_match, keyword_coverage, normalized_match, numeric_match,
)

EVAL_DIR = os.path.join("data", "evaluation")

EVAL_FILES = {
    "financial_qa": "financial_qa.jsonl",
    "terminology": "terminology.jsonl",
    "numerical": "numerical_questions.jsonl",
    "reasoning": "reasoning.jsonl",
}


def load_eval_set(category: str):
    path = os.path.join(EVAL_DIR, EVAL_FILES[category])
    if not os.path.exists(path):
        return []
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


@torch.no_grad()
def generate_answer(model, prompt: str, max_new_tokens: int = 48,
                    temperature: float = 0.7, top_k: int = 40, device: str = "cpu") -> str:
    enc = get_encoding()
    ids = enc.encode_ordinary(prompt)
    # Leave room for the generated continuation inside the context window.
    max_ctx = model.config.block_size - max_new_tokens
    if len(ids) > max_ctx:
        ids = ids[-max_ctx:]
    context = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
    generated = model.generate(context, max_new_tokens, temperature, top_k)
    completion = generated[0, len(ids):].tolist()
    return enc.decode(completion)


def score_item(category: str, item: dict, prediction: str) -> dict:
    """Score one prediction. Returns the full record, including the raw
    prediction so results are auditable."""
    record = {
        "id": item.get("id"),
        "category": category,
        "question": item.get("question"),
        "expected": item.get("answer"),
        "prediction": prediction,
    }

    if category in ("numerical", "reasoning") and item.get("numeric_answer") is not None:
        correct = numeric_match(prediction, item["numeric_answer"])
        record["match_type"] = "numeric"
    elif category == "terminology":
        correct = normalized_match(prediction, item["answer"], item.get("acceptable"))
        record["match_type"] = "normalized"
    else:
        coverage = keyword_coverage(prediction, item.get("keywords", []))
        record["keyword_coverage"] = round(coverage, 3)
        correct = coverage >= 0.5
        record["match_type"] = "keyword_coverage>=0.5"

    record["exact_match"] = exact_match(prediction, item.get("answer", ""))
    record["correct"] = bool(correct)
    return record


def evaluate_model(model, device: str = "cpu", categories=None, max_new_tokens: int = 48,
                    verbose: bool = False) -> dict:
    model.eval()
    categories = categories or list(EVAL_FILES)
    results = {"categories": {}, "details": []}

    for category in categories:
        items = load_eval_set(category)
        if not items:
            results["categories"][category] = {
                "total": 0, "correct": 0, "accuracy": None, "note": "No evaluation items found",
            }
            continue

        scored = []
        for item in items:
            prompt = f"Question: {item['question']}\nAnswer:"
            prediction = generate_answer(
                model, prompt, max_new_tokens=max_new_tokens, device=device
            )
            record = score_item(category, item, prediction)
            scored.append(record)
            results["details"].append(record)
            if verbose:
                mark = "OK " if record["correct"] else "XX "
                print(f"  {mark}{item['id']}: {prediction[:70]!r}")

        results["categories"][category] = aggregate(scored)

    all_scored = results["details"]
    results["overall"] = aggregate(all_scored)
    results["label"] = "Financial LLM POC Evaluation"
    return results


def check_leakage(shards_root: str = os.path.join("data", "shards")) -> dict:
    """Confirm no evaluation question text appears verbatim in the prepared
    training shards. Decodes shards back to text and does a substring check.
    """
    from data_sources.shard_writer import load_shard_index
    import numpy as np

    enc = get_encoding()
    questions = []
    for category in EVAL_FILES:
        for item in load_eval_set(category):
            questions.append((item["id"], item["question"]))

    if not os.path.isdir(shards_root):
        return {"checked": 0, "leaks": [], "note": "No shards directory found"}

    leaks = []
    checked_shards = 0
    for dataset_name in os.listdir(shards_root):
        train_dir = os.path.join(shards_root, dataset_name, "train")
        index = load_shard_index(train_dir)
        for shard in index.get("shards", []):
            path = os.path.join(train_dir, shard["file"])
            if not os.path.exists(path):
                continue
            arr = np.fromfile(path, dtype=np.uint16)
            text = enc.decode(arr.tolist()).lower()
            checked_shards += 1
            for qid, question in questions:
                if question.lower() in text:
                    leaks.append({"id": qid, "shard": path})

    return {
        "checked_shards": checked_shards,
        "eval_questions": len(questions),
        "leaks": leaks,
        "clean": len(leaks) == 0,
    }


def print_report(results: dict):
    print("=" * 62)
    print(results.get("label", "Financial LLM POC Evaluation"))
    print("=" * 62)
    print(f"{'Category':<22}{'Correct':>10}{'Total':>8}{'Accuracy':>12}")
    print("-" * 62)
    for category, stats in results["categories"].items():
        acc = "Not available" if stats["accuracy"] is None else f"{stats['accuracy']:.2f}%"
        print(f"{category:<22}{stats['correct']:>10}{stats['total']:>8}{acc:>12}")
    print("-" * 62)
    overall = results["overall"]
    acc = "Not available" if overall["accuracy"] is None else f"{overall['accuracy']:.2f}%"
    print(f"{'OVERALL':<22}{overall['correct']:>10}{overall['total']:>8}{acc:>12}")
    print("=" * 62)
