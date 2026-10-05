"""Re-score saved raw outputs with the current scorer, and report what moved.

The rules require raw outputs to be preserved. This is what that buys: when a
scorer defect is found, every system can be re-measured from the outputs it
already produced, with no model re-run, no GPU, and no new look at the frozen
set. The model's answers are fixed; only the judgement of them changes.

Applied to EVERY saved system, not only the one that prompted the fix, so a
scorer change cannot quietly favour whichever system found the bug.

    python scripts/rescore_saved.py
    python scripts/rescore_saved.py --out reports/phase4/rescore_2026-10-05.json
"""

import argparse
import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts.eval_heldout import FROZEN, SELECTION, score_item  # noqa: E402


def load_items(split):
    path = FROZEN if split == "frozen" else SELECTION
    with open(path, encoding="utf-8") as handle:
        return {item["id"]: item for item in
                (json.loads(line) for line in handle if line.strip())}


def rescore(path, items_by_split):
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    records = payload.get("records")
    summary = payload.get("summary", {})
    if not records:
        return None

    split = summary.get("split") or "frozen"
    items = items_by_split.setdefault(split, load_items(split))

    changed, missing = [], 0
    was_k = now_k = scored = 0
    for record in records:
        item = items.get(record["id"])
        if item is None:
            missing += 1
            continue
        # Abstention controls are excluded from the overall figure by the same
        # rule evaluate() applies, so the two numbers stay comparable.
        if record["gate"] == "abstention" and item.get("expected") is False:
            continue
        scored += 1
        before = bool(record["correct"])
        correct, category = score_item(item, record["answer"])
        was_k += before
        now_k += bool(correct)
        if bool(correct) != before:
            changed.append({"id": record["id"], "gate": record["gate"],
                            "was": before, "now": bool(correct),
                            "was_category": record.get("category"),
                            "now_category": category,
                            "expected": item.get("expected"),
                            "answer": str(record["answer"])[:160]})

    return {
        "file": os.path.basename(path),
        "label": summary.get("label", "?"),
        "split": split,
        "scored_items": scored,
        "items_not_found_in_split": missing,
        "before_pct": round(100.0 * was_k / max(scored, 1), 2),
        "after_pct": round(100.0 * now_k / max(scored, 1), 2),
        "delta_pp": round(100.0 * (now_k - was_k) / max(scored, 1), 2),
        "changed_count": len(changed),
        "newly_correct": sum(1 for c in changed if c["now"]),
        "newly_wrong": sum(1 for c in changed if not c["now"]),
        "changed": changed,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--glob", default="reports/heldout/*.json")
    parser.add_argument("--out", default="reports/phase4/rescore_report.json")
    args = parser.parse_args()

    items_by_split = {}
    results = []
    for path in sorted(glob.glob(args.glob)):
        try:
            result = rescore(path, items_by_split)
        except Exception as error:                      # noqa: BLE001
            print(f"  skipped {os.path.basename(path)}: "
                  f"{type(error).__name__}: {error}")
            continue
        if result:
            results.append(result)

    header = (f"{'system':<46}{'split':<11}{'n':>5}{'before':>9}{'after':>8}"
              f"{'delta':>8}{'+':>5}{'-':>4}")
    print(header)
    print("-" * len(header))
    for r in sorted(results, key=lambda r: -abs(r["delta_pp"])):
        print(f"{r['label'][:45]:<46}{r['split']:<11}{r['scored_items']:>5}"
              f"{r['before_pct']:>8.2f}%{r['after_pct']:>7.2f}%"
              f"{r['delta_pp']:>+7.2f}{r['newly_correct']:>5}{r['newly_wrong']:>4}")

    moved = [r for r in results if r["changed_count"]]
    print(f"\n{len(moved)} of {len(results)} systems moved.")
    for r in moved:
        print(f"\n{r['label']} ({r['split']}): {r['changed_count']} items changed")
        for change in r["changed"][:6]:
            direction = "now correct" if change["now"] else "NOW WRONG"
            print(f"  {change['id']:<18}{direction:<12} "
                  f"was {change['was_category']} -> {change['now_category']}")
            print(f"    expected {change['expected']!r}  answer "
                  f"{change['answer'][:88]!r}")
        if r["changed_count"] > 6:
            print(f"  ... {r['changed_count'] - 6} more in the JSON")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump({"scorer_change": "numbers_in rejects letter-adjacent tokens "
                                    "instead of truncating them, and accepts a "
                                    "ratio's x unit",
                   "systems": results}, handle, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
