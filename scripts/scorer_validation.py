"""Scorer validation: sample 30 outputs for hand-labelling, then score agreement.

EXPERIMENT_RULES_v2 section 4. The scorer decides every number in this project,
and nobody has checked it against a human. This does that check.

  sample  writes 30 items with the scorer's verdict HIDDEN, so the labelling is
          not anchored by it
  score   reads the labelled file back and reports raw agreement, Cohen's kappa,
          and every disagreement in full

Acceptance: kappa >= 0.8 with no systematic disagreement in one category. Below
that, the scorer is fixed and any results produced with it are recomputed.

    python scripts/scorer_validation.py sample --from reports/heldout/aivora_baseline_frozen.json
    # ... you fill in "human_label": true/false in the produced file ...
    python scripts/scorer_validation.py score
"""

import argparse
import json
import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

OUT_DIR = os.path.join("reports", "scorer_validation")
SAMPLE = os.path.join(OUT_DIR, "sample_for_labelling.jsonl")
RESULT = os.path.join(OUT_DIR, "agreement.json")
SEED = 31337


def sample(sources, count):
    pools = []
    for path in sources:
        if not os.path.exists(path):
            print(f"  skipping missing {path}")
            continue
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
        label = payload["summary"]["label"]
        for record in payload["records"]:
            pools.append((label, record))
    if not pools:
        sys.exit("no scored runs found - score a system first")

    rng = random.Random(SEED)
    # Stratify by gate, and deliberately over-sample the scorer's WRONG verdicts:
    # if the scorer is broken, that is where it shows.
    by_gate = {}
    for entry in pools:
        by_gate.setdefault(entry[1]["gate"], []).append(entry)
    picked = []
    per_gate = max(1, count // max(len(by_gate), 1))
    for gate, entries in sorted(by_gate.items()):
        wrong = [e for e in entries if not e[1]["correct"]]
        right = [e for e in entries if e[1]["correct"]]
        rng.shuffle(wrong)
        rng.shuffle(right)
        take_wrong = min(len(wrong), per_gate // 2 + 1)
        take_right = min(len(right), per_gate - take_wrong)
        picked += wrong[:take_wrong] + right[:take_right]
    rng.shuffle(picked)
    picked = picked[:count]

    frozen = {}
    with open(os.path.join("data", "eval_heldout", "heldout_frozen.jsonl"),
              encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                item = json.loads(line)
                frozen[item["id"]] = item

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(SAMPLE, "w", encoding="utf-8") as handle:
        for index, (system, record) in enumerate(picked):
            item = frozen.get(record["id"], {})
            handle.write(json.dumps({
                "n": index + 1,
                "system": system,
                "item_id": record["id"],
                "gate": record["gate"],
                "question": item.get("question"),
                "context": item.get("context"),
                "expected": record["expected"],
                "model_answer": record["answer"],
                "human_label": None,
                "_instructions": ("set human_label to true if the model answer is "
                                  "correct for this question, false if not. The "
                                  "scorer's own verdict is deliberately not shown."),
            }) + "\n")
    print(f"wrote {len(picked)} items to {SAMPLE}")
    print("Label each one by setting \"human_label\" to true or false, then run:")
    print("  python scripts/scorer_validation.py score")


def kappa(agree_both_true, agree_both_false, a_true_b_false, a_false_b_true):
    n = agree_both_true + agree_both_false + a_true_b_false + a_false_b_true
    if n == 0:
        return None
    observed = (agree_both_true + agree_both_false) / n
    p_a_true = (agree_both_true + a_true_b_false) / n
    p_b_true = (agree_both_true + a_false_b_true) / n
    expected = p_a_true * p_b_true + (1 - p_a_true) * (1 - p_b_true)
    if expected == 1:
        return 1.0
    return (observed - expected) / (1 - expected)


def score():
    if not os.path.exists(SAMPLE):
        sys.exit(f"{SAMPLE} not found - run the sample step first")
    with open(SAMPLE, encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]

    unlabelled = [r["n"] for r in rows if r.get("human_label") is None]
    if unlabelled:
        sys.exit(f"{len(unlabelled)} items still unlabelled: {unlabelled[:10]}")

    # Re-score each answer with the CURRENT scorer, so this measures the scorer
    # rather than a stored verdict.
    from scripts.eval_heldout import score_item

    frozen = {}
    with open(os.path.join("data", "eval_heldout", "heldout_frozen.jsonl"),
              encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                item = json.loads(line)
                frozen[item["id"]] = item

    both_true = both_false = scorer_only = human_only = 0
    disagreements = []
    for row in rows:
        item = frozen[row["item_id"]]
        scorer_correct, category = score_item(item, row["model_answer"])
        human_correct = bool(row["human_label"])
        if scorer_correct and human_correct:
            both_true += 1
        elif not scorer_correct and not human_correct:
            both_false += 1
        elif scorer_correct and not human_correct:
            scorer_only += 1
            disagreements.append({**row, "scorer": True, "human": False,
                                  "category": category})
        else:
            human_only += 1
            disagreements.append({**row, "scorer": False, "human": True,
                                  "category": category})

    n = len(rows)
    agreement = (both_true + both_false) / n
    k = kappa(both_true, both_false, scorer_only, human_only)
    by_gate = {}
    for entry in disagreements:
        by_gate[entry["gate"]] = by_gate.get(entry["gate"], 0) + 1

    result = {
        "items": n,
        "raw_agreement_pct": round(100 * agreement, 2),
        "cohens_kappa": round(k, 4) if k is not None else None,
        "confusion": {"both_correct": both_true, "both_incorrect": both_false,
                      "scorer_says_correct_human_says_not": scorer_only,
                      "human_says_correct_scorer_says_not": human_only},
        "disagreements_by_gate": by_gate,
        "acceptance": {"threshold_kappa": 0.8,
                       "passed": bool(k is not None and k >= 0.8),
                       "systematic_concern": max(by_gate.values()) > n * 0.15
                       if by_gate else False},
        "disagreements": disagreements,
    }
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(RESULT, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)

    print(f"items labelled       {n}")
    print(f"raw agreement        {result['raw_agreement_pct']}%")
    print(f"Cohen's kappa        {result['cohens_kappa']}")
    print(f"confusion            {result['confusion']}")
    print(f"acceptance (k>=0.8)  "
          f"{'PASS' if result['acceptance']['passed'] else 'FAIL'}")
    if disagreements:
        print(f"\n{len(disagreements)} disagreement(s):")
        for entry in disagreements:
            print(f"\n  [{entry['gate']}] {entry['question']}")
            print(f"    expected: {entry['expected']!r}")
            print(f"    answer:   {entry['model_answer'][:100]!r}")
            print(f"    scorer says {entry['scorer']}, you say {entry['human']} "
                  f"(category {entry['category']})")
    print(f"\nwrote {RESULT}")
    if not result["acceptance"]["passed"]:
        print("\nScorer NOT accepted. Fix it, then recompute every result "
              "produced with it before drawing conclusions.")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("sample", "score"))
    parser.add_argument("--from", dest="sources", nargs="*",
                        default=[os.path.join("reports", "heldout",
                                              "aivora_baseline_frozen.json")])
    parser.add_argument("--count", type=int, default=30)
    args = parser.parse_args()

    if args.action == "sample":
        sample(args.sources, args.count)
    else:
        score()


if __name__ == "__main__":
    main()
