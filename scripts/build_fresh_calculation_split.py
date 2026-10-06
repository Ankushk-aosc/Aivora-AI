"""Task 5 — a calculation split built independently of the pipeline.

Why. Operand resolution raised calculation from 83.3% to 100% on the frozen set,
and the captions that did it were written AFTER the eight frozen failures were
inspected (commit ac16f13, 2026-10-05). The 100% is therefore not a clean
held-out measurement. This builds a split the vocabulary has never seen so the
same capability can be measured honestly.

Three rules were followed in building it:

1. **Expected answers are computed here, in plain Python, from the operands** -
   never by calling pipeline.tool_pipeline. If the pipeline's formula table is
   wrong, this split disagrees with it rather than inheriting the error.
2. **Captions come from standard statement terminology**, written from
   accounting usage rather than read off the pipeline's SYNONYMS table. Which of
   them the table happens to cover is then a *measurement*, reported as a
   covered/uncovered split, instead of something arranged in advance.
3. **No context is reused** from the frozen set or the training corpus; every
   figure is generated fresh from a seeded RNG.

    python scripts/build_fresh_calculation_split.py
"""

import json
import os
import random

OUT_DIR = os.path.join("data", "eval_fresh")
OUT = os.path.join(OUT_DIR, "calculation_fresh.jsonl")
SEED = 20261006

# (question wording, [(caption, role)], how to compute, unit)
# Captions are ordinary statement wordings - Companies Act formats, IFRS and
# common commercial usage. Written from terminology, not from the table.
TEMPLATES = [
    ("What is the gross margin?",
     [("Sales revenue", "a"), ("Cost of sales", "b")],
     lambda a, b: 100.0 * (a - b) / a, "%"),
    ("What is the gross margin?",
     [("Turnover", "a"), ("Cost of goods sold", "b")],
     lambda a, b: 100.0 * (a - b) / a, "%"),
    ("What is gross profit?",
     [("Revenue from contracts with customers", "a"), ("Cost of sales", "b")],
     lambda a, b: a - b, ""),
    ("What is the net profit margin?",
     [("Profit for the year", "a"), ("Revenue", "b")],
     lambda a, b: 100.0 * a / b, "%"),
    ("What is the net profit margin?",
     [("Profit attributable to shareholders", "a"), ("Turnover", "b")],
     lambda a, b: 100.0 * a / b, "%"),
    ("What is the operating margin?",
     [("Operating profit", "a"), ("Revenue", "b")],
     lambda a, b: 100.0 * a / b, "%"),
    ("What is the operating margin?",
     [("Profit from operations", "a"), ("Sales", "b")],
     lambda a, b: 100.0 * a / b, "%"),
    ("What is the current ratio?",
     [("Current assets", "a"), ("Current liabilities", "b")],
     lambda a, b: a / b, "x"),
    ("What is the current ratio?",
     [("Total current assets", "a"),
      ("Creditors: amounts falling due within one year", "b")],
     lambda a, b: a / b, "x"),
    ("What is working capital?",
     [("Current assets", "a"), ("Current liabilities", "b")],
     lambda a, b: a - b, ""),
    ("What is the debt-to-equity ratio?",
     [("Total debt", "a"), ("Total equity", "b")],
     lambda a, b: a / b, "x"),
    ("What is the gearing ratio?",
     [("Loans and borrowings", "a"), ("Capital and reserves", "b")],
     lambda a, b: a / b, "x"),
    ("What is return on equity as a percentage?",
     [("Profit for the year", "a"), ("Shareholders' funds", "b")],
     lambda a, b: 100.0 * a / b, "%"),
    ("What is return on assets as a percentage?",
     [("Net profit", "a"), ("Total assets", "b")],
     lambda a, b: 100.0 * a / b, "%"),
    ("What is the asset turnover?",
     [("Revenue", "a"), ("Total assets", "b")],
     lambda a, b: a / b, "x"),
    ("What is the interest cover?",
     [("Operating profit", "a"), ("Finance costs", "b")],
     lambda a, b: a / b, "x"),
    ("What is the interest cover?",
     [("Earnings before interest and tax", "a"), ("Interest payable", "b")],
     lambda a, b: a / b, "x"),
    ("What is free cash flow?",
     [("Net cash from operating activities", "a"),
      ("Purchases of property, plant and equipment", "b")],
     lambda a, b: a - b, ""),
    ("What is earnings per share?",
     [("Profit for the year", "a"), ("Ordinary shares in issue", "b")],
     lambda a, b: a / b, ""),
    ("What is the revenue growth rate?",
     [("Revenue this year", "a"), ("Revenue last year", "b")],
     lambda a, b: 100.0 * (a - b) / b, "%"),
]

# Distractor lines, so the right figure must be found rather than guessed.
DISTRACTORS = ["Employees", "Number of branches", "Dividend per share",
               "Corporation tax charge", "Depreciation", "Other income",
               "Share price at year end", "Directors' remuneration"]


def money(rng):
    return round(rng.uniform(500, 90000), 2)


# Both operands drawn independently from one range produced gross margins of
# -35.8% and net margins of 1045% - arithmetically correct and financially
# absurd. A split that does not look like a statement is not evidence about
# reading statements, so the second operand is scaled to a plausible ratio of
# the first.
RATIO = {
    "margin_cost": (0.45, 0.80),    # cost of sales as a share of revenue
    "margin_profit": (0.02, 0.22),  # profit as a share of revenue
    "ratio_小": (0.30, 0.90),
}


def related(rng, base, low, high):
    return round(base * rng.uniform(low, high), 2)


def render(value, unit):
    if unit == "%":
        return f"{value:.2f}%"
    if unit == "x":
        return f"{value:.2f}x"
    return f"{value:,.2f}"


def build():
    rng = random.Random(SEED)
    rows = []
    for index in range(44):
        question, operands, compute, unit = TEMPLATES[index % len(TEMPLATES)]
        # The first operand is drawn freely; the second is scaled to a
        # plausible relationship with it, chosen from the question's shape.
        lowered = question.lower()
        if "gross margin" in lowered or "gross profit" in lowered:
            span = (0.45, 0.80)          # cost below revenue
        elif "margin" in lowered or "return on" in lowered:
            span = (3.0, 25.0)           # denominator far above the profit
        elif "growth" in lowered:
            span = (0.75, 0.95)          # prior year below this year
        elif "cover" in lowered or "turnover" in lowered:
            span = (0.05, 0.40)
        elif "per share" in lowered:
            span = (50.0, 400.0)         # many shares against one profit figure
        else:
            span = (0.35, 0.85)

        values, lines = {}, []
        first = money(rng)
        for position, (caption, role) in enumerate(operands):
            value = first if position == 0 else related(rng, first, *span)
            values[role] = value
            lines.append((caption, value))
        # a denominator of zero or a negative margin is not what is being tested
        if values.get("b", 1) == 0:
            continue
        for _ in range(rng.randint(1, 3)):
            lines.append((rng.choice(DISTRACTORS), round(rng.uniform(1, 900), 2)))
        rng.shuffle(lines)
        context = "\n".join(f"{c}: {v:,.2f}" for c, v in lines)

        # Expected answer computed HERE, from the operands, in plain Python.
        expected_value = compute(*[values[r] for _, r in operands])
        rows.append({
            "id": f"fresh_calc_{index:03d}",
            "gate": "calculation",
            "origin": "generated_fresh",
            "question": question,
            "context": context,
            "expected": render(expected_value, unit),
            "expected_numeric": round(expected_value, 6),
            "operand_captions": [c for c, _ in operands],
            "unit": unit,
            "note": "expected computed independently of pipeline.tool_pipeline",
        })

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    # Coverage against the pipeline's vocabulary - measured after the fact.
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from pipeline.tool_pipeline import ToolPipeline

    covered = uncovered = 0
    unseen_captions = set()
    for row in rows:
        resolvable = True
        for caption in row["operand_captions"]:
            # does ANY operand field claim this caption?
            hit = any(ToolPipeline.resolve_field(f"{caption}: 1.00", field)
                      for field in __import__(
                          "pipeline.tool_pipeline", fromlist=["SYNONYMS"]).SYNONYMS)
            if not hit:
                resolvable = False
                unseen_captions.add(caption)
        covered += resolvable
        uncovered += not resolvable

    print(f"wrote {OUT}: {len(rows)} calculation items")
    print(f"  seed {SEED}, contexts generated fresh, no reuse from frozen set")
    print(f"  expected answers computed in this file, not by the pipeline")
    print()
    print(f"  captions the vocabulary covers    : {covered} items")
    print(f"  captions it does NOT cover        : {uncovered} items")
    if unseen_captions:
        print(f"  uncovered captions: {sorted(unseen_captions)}")
    manifest = {"items": len(rows), "seed": SEED, "source": "generated fresh",
                "expected_computed_by": "scripts/build_fresh_calculation_split.py",
                "covered_items": covered, "uncovered_items": uncovered,
                "uncovered_captions": sorted(unseen_captions)}
    with open(os.path.join(OUT_DIR, "manifest.json"), "w", encoding="utf-8") as h:
        json.dump(manifest, h, indent=2)
    return rows


if __name__ == "__main__":
    build()
