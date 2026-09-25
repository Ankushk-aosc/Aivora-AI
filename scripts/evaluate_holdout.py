"""Score the pipeline on HELD-OUT questions it was never tuned against.

The 95.56% on the project's own 45 questions was reached after reading their
failures, so it is optimistic by construction. These 18 questions were written
afterwards and never looked at while fixing anything.
"""
import json
import os
import sys
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "hf_export"))

g = types.ModuleType("gradio")


class _W:
    def __init__(self, *a, **k):
        pass


class _I:
    def __init__(self, **k):
        self.fn = k.get("fn")

    def launch(self, *a, **k):
        pass


g.Textbox = g.Slider = g.Markdown = _W
g.Interface = _I
sys.modules["gradio"] = g

import app  # noqa: E402

from evaluation.evaluator import score_item  # noqa: E402
from evaluation.financial_metrics import aggregate  # noqa: E402

items = [json.loads(line) for line in
         open(os.path.join(ROOT, "data", "evaluation", "holdout.jsonl"), encoding="utf-8")
         if line.strip()]

scored, fails = [], []
for item in items:
    prediction = app.answer(item["question"], 48, 0.7)
    record = score_item(item["category"], item, prediction)
    scored.append(record)
    mark = "OK " if record["correct"] else "XX "
    print(f"{mark}{item['id']}: {item['question'][:58]}")
    if not record["correct"]:
        fails.append((item, prediction))

print("")
print("HELD-OUT RESULT:", aggregate(scored))
from collections import Counter
by_cat = {}
for r in scored:
    by_cat.setdefault(r["category"], []).append(r)
for cat, rows in sorted(by_cat.items()):
    print(f"   {cat:14} {sum(1 for r in rows if r['correct']):>3}/{len(rows):<3} "
          f"({100*sum(1 for r in rows if r['correct'])/len(rows):.0f}%)")
print("   graded by:", dict(Counter(r.get("match_type") for r in scored)))
for item, prediction in fails:
    print("")
    print(f"[{item['id']}] {item['question']}")
    print(f"   expected: {item['answer']}")
    print(f"   got     : {prediction[:170]}")
