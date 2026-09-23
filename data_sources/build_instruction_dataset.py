"""Build the Stage B instruction-tuning JSONL from Hugging Face datasets.

The file this replaces, data/instruction/financial_instructions.jsonl, held
800 records - the first 800 rows of finance-alpaca. That is far too few to
teach answering.

Sources (all listed in data_sources/dataset_registry.py):

* gbharti/finance-alpaca - finance Q&A in instruction/input/output form; the
  main source.
* FinLang/investopedia-instruction-tuning-dataset - Investopedia Q&A. Its
  answers are written about a passage the model will not see, so rows whose
  answer refers to one ("the passage", "the table above", ...) are dropped.
* databricks/databricks-dolly-15k - general instruction following, capped
  small, so the model keeps answering non-financial instructions sensibly.

sujet-ai/Sujet-Finance-Instruct-177k is deliberately NOT used: sampling it
returns almost entirely sentiment-classification rows whose answer is one word
("neutral"), which teaches labels rather than explanations.

Every row is checked against data/evaluation/*.jsonl: an evaluation question
that leaked into training would make the evaluation score meaningless.

usage:
    python -m data_sources.build_instruction_dataset [--out PATH] [--limit N]
"""

import argparse
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

DEFAULT_OUT = os.path.join("data", "instruction", "financial_instructions.jsonl")
EVAL_DIR = os.path.join("data", "evaluation")

MIN_ANSWER_CHARS = 40
MAX_ANSWER_CHARS = 4000
MIN_QUESTION_CHARS = 12

# Investopedia answers that talk about source material the model will not have.
REFERS_TO_PASSAGE = re.compile(
    r"\b(the (given )?(passage|text|context|article|table|excerpt)|"
    r"according to the (passage|text|context)|as (stated|mentioned|shown) (in|above))\b",
    re.IGNORECASE)


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", " ", (text or "").lower()).strip()


def load_eval_questions() -> set:
    """Normalized evaluation questions, used to keep them out of training."""
    questions = set()
    if not os.path.isdir(EVAL_DIR):
        return questions
    for name in os.listdir(EVAL_DIR):
        if not name.endswith(".jsonl"):
            continue
        with open(os.path.join(EVAL_DIR, name), encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except ValueError:
                    continue
                q = _norm(item.get("question", ""))
                if q:
                    questions.add(q)
    return questions


def finance_alpaca(limit):
    from datasets import load_dataset
    ds = load_dataset("gbharti/finance-alpaca", split="train", streaming=True)
    for row in ds:
        if limit is not None and limit <= 0:
            return
        yield {"instruction": row.get("instruction", ""), "input": row.get("input", ""),
               "output": row.get("output", "")}
        limit = None if limit is None else limit - 1


def investopedia(limit):
    from datasets import load_dataset
    ds = load_dataset("FinLang/investopedia-instruction-tuning-dataset", split="train",
                      streaming=True)
    for row in ds:
        if limit is not None and limit <= 0:
            return
        answer = (row.get("Answer") or "").strip()
        if REFERS_TO_PASSAGE.search(answer):
            continue
        yield {"instruction": (row.get("Question") or "").strip(), "input": "",
               "output": answer}
        limit = None if limit is None else limit - 1


def dolly(limit):
    from datasets import load_dataset
    ds = load_dataset("databricks/databricks-dolly-15k", split="train", streaming=True)
    for row in ds:
        if limit is not None and limit <= 0:
            return
        yield {"instruction": row.get("instruction", ""), "input": row.get("context", ""),
               "output": row.get("response", "")}
        limit = None if limit is None else limit - 1


SOURCES = [
    ("finance_alpaca", finance_alpaca, 40000),
    ("investopedia", investopedia, 10000),
    ("dolly", dolly, 5000),
]


def build(out_path=DEFAULT_OUT, limit_scale=1.0):
    eval_questions = load_eval_questions()
    print(f"Evaluation questions to keep out of training: {len(eval_questions)}")

    seen = set()
    counts = {}
    dropped = {"short": 0, "long": 0, "duplicate": 0, "leaked": 0}
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    tmp = out_path + ".tmp"

    with open(tmp, "w", encoding="utf-8") as out:
        for name, fn, cap in SOURCES:
            cap = max(1, int(cap * limit_scale))
            kept = 0
            for rec in fn(cap):
                instruction = (rec["instruction"] or "").strip()
                output = (rec["output"] or "").strip()
                if len(instruction) < MIN_QUESTION_CHARS or len(output) < MIN_ANSWER_CHARS:
                    dropped["short"] += 1
                    continue
                if len(output) > MAX_ANSWER_CHARS:
                    dropped["long"] += 1
                    continue
                key = _norm(instruction)
                if key in seen:
                    dropped["duplicate"] += 1
                    continue
                if key in eval_questions:
                    dropped["leaked"] += 1
                    continue
                seen.add(key)
                out.write(json.dumps({"instruction": instruction,
                                      "input": (rec["input"] or "").strip(),
                                      "output": output,
                                      "source": name}) + "\n")
                kept += 1
            counts[name] = kept
            print(f"  {name}: kept {kept}")

    os.replace(tmp, out_path)
    total = sum(counts.values())
    print(f"\nWrote {total:,} records to {out_path}")
    print(f"Dropped: {dropped}")
    if dropped["leaked"]:
        print(f"NOTE: {dropped['leaked']} evaluation question(s) were found in the source "
              f"data and excluded.")
    return {"total": total, "by_source": counts, "dropped": dropped, "path": out_path}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--scale", type=float, default=1.0,
                    help="fraction of each source's cap to take (0.01 for a quick test)")
    args = ap.parse_args()
    build(args.out, args.scale)
