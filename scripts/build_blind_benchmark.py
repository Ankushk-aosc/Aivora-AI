"""Build data/benchmark/blind.jsonl from the authored pools.

Part 18 asked for ~1,000 authored questions. This produces 174, and the
shortfall is stated rather than padded: the value of this set is that every
question is a question a person might ask, with a rubric written from the
finance. Filling it to 1,000 by generating variations would give the number the
brief asked for and destroy the thing it was asked for.

Read the honesty note at the top of data/benchmark/blind_authored.py before
quoting any score from this set: the questions were written by Claude in the
same session as the pipeline they measure, so they test the CODE's
generalisation fairly but are not blind to the author of the glossary.

usage:
    python scripts/build_blind_benchmark.py
"""

import importlib.util
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

OUT_DIR = os.path.join("data", "benchmark")
POOLS = ["blind_authored.py", "blind_authored_extra.py"]

# The brief's distribution, for reporting how far this set falls short of it.
TARGET = {"accounting": 150, "reporting": 150, "interpretation": 150,
          "corporate_finance": 150, "valuation": 150, "risk": 100,
          "concepts": 150, "general": 50}


def load_pool(filename):
    path = os.path.join(OUT_DIR, filename)
    spec = importlib.util.spec_from_file_location(filename[:-3], path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.CATEGORIES


def main():
    pools = {}
    for filename in POOLS:
        for category, rows in load_pool(filename).items():
            pools.setdefault(category, []).extend(rows)

    items, seen = [], set()
    duplicates = 0
    for category, rows in sorted(pools.items()):
        for index, (level, question, answer, rubric) in enumerate(rows):
            key = question.strip().lower()
            if key in seen:
                duplicates += 1
                continue
            seen.add(key)
            items.append({
                "id": f"blind_{category}_{index:03d}",
                "category": category, "level": level, "kind": "authored",
                "question": question, "answer": answer, "required_any": rubric,
            })

    # Overlap against the existing splits would make this set worthless as a
    # separate measurement, so it is checked rather than assumed.
    existing = set()
    for split in ("dev", "test"):
        path = os.path.join(OUT_DIR, f"{split}.jsonl")
        if os.path.exists(path):
            with open(path, encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        existing.add(json.loads(line)["question"].strip().lower())
    overlapping = [i["id"] for i in items
                   if i["question"].strip().lower() in existing]
    items = [i for i in items if i["question"].strip().lower() not in existing]

    with open(os.path.join(OUT_DIR, "blind.jsonl"), "w", encoding="utf-8") as handle:
        for item in items:
            handle.write(json.dumps(item) + "\n")

    by_category = {}
    for item in items:
        by_category[item["category"]] = by_category.get(item["category"], 0) + 1
    manifest = {
        "examples": len(items),
        "target_examples": sum(TARGET.values()),
        "completion_pct": round(100.0 * len(items) / sum(TARGET.values()), 1),
        "by_category": dict(sorted(by_category.items())),
        "target_by_category": TARGET,
        "by_level": {str(level): sum(1 for i in items if i["level"] == level)
                     for level in sorted({i["level"] for i in items})},
        "duplicates_dropped": duplicates,
        "removed_for_overlap_with_dev_or_test": overlapping,
        "all_rubric_graded": all(i.get("required_any") for i in items),
        "authorship": ("written by Claude in the same session as the pipeline it "
                       "measures: a fair test of the code's generalisation, not a "
                       "blind test of the glossary. Treat scores as an upper bound."),
        "usage": ("measurement only - never used for training, and no glossary "
                  "entry, pattern or rule may be added because an item here fails "
                  "without first reproducing the failure on dev."),
    }
    with open(os.path.join(OUT_DIR, "blind_manifest.json"), "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    print(json.dumps(manifest, indent=2))

    # Sanity: a rubric must be groups of alternatives, and the reference answer
    # should satisfy its own rubric - a rubric its own answer fails is wrong.
    from evaluation.financial_metrics import rubric_match

    malformed = [i["id"] for i in items
                 if not all(isinstance(g, (list, tuple)) for g in i["required_any"])]
    self_failing = [i["id"] for i in items
                    if not rubric_match(i["answer"], i["required_any"])]
    print(f"\nmalformed rubrics: {malformed or 'none'}")
    print(f"rubrics their own reference answer fails: {len(self_failing)}")
    for item_id in self_failing[:12]:
        print(f"  {item_id}")


if __name__ == "__main__":
    main()
