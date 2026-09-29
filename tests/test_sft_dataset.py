"""Validate the SFT dataset independently of the script that produced it.

scripts/build_sft_dataset.py verifies as it generates. That is not the same as
the dataset being verifiable: a bug in the generator would produce rows that are
wrong and self-consistent. This reads the file and checks it on its own terms.

Run: deepseek_env/Scripts/python.exe tests/test_sft_dataset.py
"""
import json
import os
import re
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.financial_metrics import abstained, extract_numbers  # noqa: E402

DATA = os.path.join("data", "sft", "financial_sft_v1.jsonl")
REQUIRED = ("id", "question", "context", "answer", "task_type", "domain",
            "difficulty", "source", "verification_status")

PASSED, FAILED = [], []


def check(name, ok, detail=""):
    (PASSED if ok else FAILED).append(name)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" - {detail}" if detail else ""))


def load():
    with open(DATA, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def benchmark_questions():
    questions = set()
    for name in ("dev.jsonl", "test.jsonl", "blind.jsonl"):
        path = os.path.join("data", "benchmark", name)
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    questions.add(json.loads(line)["question"].strip().lower())
    return questions


def main():
    rows = load()
    print(f"{len(rows)} rows in {DATA}\n")

    # --- schema -----------------------------------------------------------
    missing = [r.get("id", "?") for r in rows
               if any(f not in r for f in REQUIRED)]
    check("every row has the required fields", not missing, str(missing[:3]))
    ids = [r["id"] for r in rows]
    check("ids unique", len(ids) == len(set(ids)),
          f"{len(ids) - len(set(ids))} duplicates")
    check("difficulty is 1-5", all(1 <= r["difficulty"] <= 5 for r in rows))
    check("every row states a verification status",
          all(r["verification_status"] for r in rows))
    check("no empty answers", all(r["answer"].strip() for r in rows))

    # --- the promise that matters: no benchmark question is trained on ----
    benchmark = benchmark_questions()
    leaked = [r["id"] for r in rows if r["question"].strip().lower() in benchmark]
    check("no benchmark question appears in the training data", not leaked,
          f"{len(leaked)} leaked: {leaked[:3]}")

    # A weaker but useful check: no ANSWER is copied verbatim from a benchmark
    # item, which would be leakage through the target rather than the prompt.
    benchmark_answers = set()
    for name in ("dev.jsonl", "test.jsonl", "blind.jsonl"):
        path = os.path.join("data", "benchmark", name)
        if os.path.exists(path):
            with open(path, encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        answer = json.loads(line).get("answer")
                        if isinstance(answer, str) and len(answer) > 40:
                            benchmark_answers.add(answer.strip().lower())
    copied = [r["id"] for r in rows if r["answer"].strip().lower() in benchmark_answers]
    check("no benchmark reference answer is copied as a target", not copied,
          f"{len(copied)}: {copied[:3]}")

    # --- extraction: the answer must really be in the context -------------
    extraction = [r for r in rows if r["task_type"] == "extraction"]
    bad = [r["id"] for r in extraction if r["answer"].strip() not in r["context"]]
    check(f"extraction answers appear verbatim in context ({len(extraction)} rows)",
          not bad, f"{len(bad)}: {bad[:3]}")
    # And the asked field must be the one answered: the label in the question
    # must appear on the same line as the answer.
    mismatched = []
    for row in extraction[:400]:
        answer = row["answer"].strip()
        line = next((l for l in row["context"].splitlines() if answer in l), "")
        # Longest prefixes first, or "what is" strips before "what is the" and
        # leaves the article behind as the supposed field name.
        asked = re.sub(r"^(what is the|what was the|what is|what was|state)\s+", "",
                       row["question"].lower().rstrip("?.")).replace(" figure", "")
        asked = asked.replace("from the figures above", "").strip()
        if asked and asked.split()[0] not in line.lower():
            mismatched.append((row["id"], asked, line))
    check("the answered line is the asked field (400 sampled)", not mismatched,
          str(mismatched[:2]))

    # --- calculations: recompute independently where possible -------------
    from tools.financial_calculator import calculate

    recomputed, wrong = 0, []
    for row in rows:
        if row["task_type"] != "calculation" or row.get("numeric_answer") is None:
            continue
        detail = row.get("verification_detail", "")
        numbers = extract_numbers(row["question"])
        if "formula=net_profit_margin" in detail and len(numbers) >= 2:
            expected = 100.0 * numbers[0] / numbers[1]
        elif "formula=current_ratio" in detail and len(numbers) >= 2:
            expected = numbers[0] / numbers[1]
        elif "formula=free_cash_flow" in detail and len(numbers) >= 2:
            expected = numbers[0] - numbers[1]
        elif "formula=eps" in detail and len(numbers) >= 2:
            expected = numbers[0] / numbers[1]
        elif "formula=debt_to_equity" in detail and len(numbers) >= 2:
            expected = numbers[0] / numbers[1]
        elif "formula=working_capital" in detail and len(numbers) >= 2:
            expected = numbers[0] - numbers[1]
        elif "formula=equity_from_identity" in detail and len(numbers) >= 2:
            expected = numbers[0] - numbers[1]
        else:
            continue
        recomputed += 1
        if abs(expected - row["numeric_answer"]) > 0.02:
            wrong.append((row["id"], round(expected, 2), row["numeric_answer"]))
    check(f"calculation answers recompute from the question ({recomputed} checked)",
          recomputed > 300 and not wrong, str(wrong[:3]))

    # The stated numeric answer must also appear in the answer text, or the
    # model is being trained on prose that disagrees with the label.
    inconsistent = [r["id"] for r in rows if r.get("numeric_answer") is not None
                    and not any(abs(v - r["numeric_answer"]) < 0.02
                                for v in extract_numbers(r["answer"]))]
    check("numeric_answer appears in the answer text", not inconsistent,
          f"{len(inconsistent)}: {inconsistent[:3]}")

    # --- abstention: the target must actually decline ---------------------
    abstention = [r for r in rows if r["task_type"] == "abstention"]
    not_declining = [r["id"] for r in abstention if not abstained(r["answer"])]
    check(f"every abstention target declines ({len(abstention)} rows)",
          not not_declining, f"{len(not_declining)}: {not_declining[:3]}")
    # And no answerable row accidentally declines, which would teach refusal.
    answerable_declining = [r["id"] for r in rows
                            if r["task_type"] in ("extraction", "calculation",
                                                  "reasoning", "grounded")
                            and abstained(r["answer"])]
    check("no answerable target declines", not answerable_declining,
          f"{len(answerable_declining)}: {answerable_declining[:3]}")

    # --- interpretation must not assert a cause as fact -------------------
    interpretation = [r for r in rows if r["task_type"] == "interpretation"]
    asserting = [r["id"] for r in interpretation
                 if not any(h in r["answer"].lower()
                            for h in ("could", "may", "might", "possible", "usually"))]
    check(f"interpretation answers are hedged, not asserted ({len(interpretation)})",
          not asserting, f"{len(asserting)}: {asserting[:3]}")

    # --- coverage ---------------------------------------------------------
    by_task = Counter(r["task_type"] for r in rows)
    for task in ("definition", "explanation", "extraction", "calculation",
                 "interpretation", "reasoning", "abstention", "grounded"):
        check(f"task type present: {task}", by_task.get(task, 0) > 0,
              f"{by_task.get(task, 0)} rows")
    check("extraction is the largest or second-largest slice",
          by_task["extraction"] >= sorted(by_task.values())[-2],
          str(by_task.most_common(3)))

    domains = {r["domain"] for r in rows}
    for domain in ("accounting", "valuation", "statements", "cash_flow",
                   "corporate_finance", "risk", "ratios"):
        check(f"domain present: {domain}", domain in domains)

    print(f"\nby task type: {dict(by_task.most_common())}")
    print(f"\n{len(PASSED)}/{len(PASSED) + len(FAILED)} passed")
    sys.exit(1 if FAILED else 0)


if __name__ == "__main__":
    main()
