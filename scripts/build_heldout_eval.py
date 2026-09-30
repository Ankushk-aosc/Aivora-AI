"""Phase 1: build, de-duplicate and FREEZE the held-out evaluation set.

Rules this enforces rather than assumes:

  * >= 60 items per gate for copy, extraction and wording; >= 30 for calculation
    and abstention
  * >= 20 HAND-WRITTEN items per gate, so no gate rests on templates alone
  * every item checked against ALL training data - exact match and 8-gram
    near-duplicate - and removed if it overlaps, with the count reported
  * a separate SELECTION split for choosing checkpoints, so the frozen set is
    touched only for final reporting
  * every calculation answer recomputed here; a disagreement is a build failure
  * frozen with SHA-256

    python scripts/build_heldout_eval.py
"""

import hashlib
import json
import os
import random
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

OUT_DIR = os.path.join("data", "eval_heldout")
FROZEN = os.path.join(OUT_DIR, "heldout_frozen.jsonl")
SELECTION = os.path.join(OUT_DIR, "selection.jsonl")
MANIFEST = os.path.join(OUT_DIR, "heldout_manifest.json")

SEED = 424242          # used nowhere else in this repository
NGRAM = 8

TRAINING_FILES = [
    "data/sft/financial_sft_v1.jsonl", "data/sft/financial_sft_v2.jsonl",
    "data/sft/financial_sft_chat.jsonl",
    "data/tiny_overfit/tiny_financial.jsonl",
    "data/instruction/financial_instructions.jsonl",
]

GATE_MINIMUMS = {"copy": 60, "extraction": 60, "wording": 60,
                 "calculation": 30, "abstention": 30}
HANDWRITTEN_MINIMUM = 20


def ngrams(text, n=NGRAM):
    words = re.findall(r"[a-z0-9.]+", text.lower())
    return {" ".join(words[i:i + n]) for i in range(max(0, len(words) - n + 1))}


def training_fingerprints():
    """Exact texts and n-grams from every training file."""
    exact, grams = set(), set()
    for path in TRAINING_FILES:
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                pieces = []
                for key in ("question", "context", "answer", "raw_text",
                            "instruction", "input", "output"):
                    value = row.get(key)
                    if isinstance(value, str) and value:
                        pieces.append(value)
                for message in (row.get("messages") or []):
                    if isinstance(message, dict) and message.get("content"):
                        pieces.append(message["content"])
                text = "\n".join(pieces)
                if text:
                    exact.add(text.strip().lower())
                    grams |= ngrams(text)
    return exact, grams


# --------------------------------------------------------------- generated

def generated_copy(rng, count):
    subjects = ["consignment code", "audit reference", "permit number",
                "badge id", "sample code", "route number", "berth number",
                "dispatch code"]
    alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
    items = []
    for _ in range(count):
        subject = rng.choice(subjects)
        code = "-".join("".join(rng.choice(alphabet) for _ in range(rng.randint(2, 4)))
                        for _ in range(rng.randint(1, 3)))
        others = []
        for _ in range(rng.randint(1, 3)):
            other = rng.choice([s for s in subjects if s != subject])
            value = "-".join("".join(rng.choice(alphabet)
                                     for _ in range(rng.randint(2, 4)))
                             for _ in range(rng.randint(1, 3)))
            others.append(f"The {other} is {value}.")
        lines = [f"The {subject} is {code}."] + others
        rng.shuffle(lines)
        items.append({"question": f"What is the {subject}?",
                      "context": " ".join(lines), "expected": code,
                      "note": f"{len(others)} distractor codes", "origin": "generated"})
    return items


FIELD_SETS = [
    ("Revenue", "Revenues from contracts", "Other revenue"),
    ("Operating profit", "Operating profit before exceptionals", "Operating cash flow"),
    ("Net debt", "Net debt excluding leases", "Gross debt"),
    ("Total assets", "Total current assets", "Total non-current assets"),
    ("Employee costs", "Employee benefit expense", "Other operating costs"),
    ("Deferred tax asset", "Deferred tax liability", "Current tax liability"),
    ("Cash generated from operations", "Cash and cash equivalents", "Cash used in financing"),
    ("Basic earnings per share", "Diluted earnings per share", "Dividend per share"),
]


def generated_extraction(rng, count, unseen_wording=False):
    templates = ["What is {label_l}?", "What figure is given for {label_l}?",
                 "State {label_l}."]
    unseen_templates = [
        "Looking at the extract, what value sits against {label_l}?",
        "From the numbers supplied, what is recorded for {label_l}?",
        "Reading the statement, how much is shown as {label_l}?",
    ]
    items = []
    for _ in range(count):
        labels = list(rng.choice(FIELD_SETS))
        extra = rng.sample([label for group in FIELD_SETS for label in group
                            if label not in labels], rng.randint(2, 5))
        labels += extra
        rng.shuffle(labels)
        values, seen = {}, set()
        for label in labels:
            while True:
                value = round(rng.uniform(15, 9950), 2)
                if value not in seen:
                    seen.add(value)
                    break
            values[label] = value
        asked = rng.choice(labels)
        lines = [f"{label}: {values[label]:,.2f}" for label in labels]
        template = rng.choice(unseen_templates if unseen_wording else templates)
        items.append({
            "question": template.format(label_l=asked.lower()),
            "context": "\n".join(lines),
            "expected": f"{values[asked]:,.2f}",
            "note": (f"{len(labels)} fields, similar labels, asked field at "
                     f"position {labels.index(asked) + 1}"),
            "origin": "generated"})
    return items


CALC_FORMS = [
    ("gross margin as a percentage", ("Revenue", "Cost of sales"),
     lambda a, b: 100.0 * (a - b) / a),
    ("the current ratio", ("Current assets", "Current liabilities"),
     lambda a, b: a / b),
    ("return on equity as a percentage", ("Net income", "Shareholders' equity"),
     lambda a, b: 100.0 * a / b),
    ("free cash flow", ("Operating cash flow", "Capital expenditure"),
     lambda a, b: a - b),
    ("earnings per share", ("Net income", "Shares outstanding"), lambda a, b: a / b),
    ("the debt-to-equity ratio", ("Total debt", "Shareholders' equity"),
     lambda a, b: a / b),
]


def generated_calculation(rng, count):
    items = []
    for index in range(count):
        name, (label_a, label_b), fn = CALC_FORMS[index % len(CALC_FORMS)]
        a = round(rng.uniform(200, 9000), 2)
        b = round(a / rng.choice([1.5, 2.0, 2.5, 4.0, 5.0]), 2)
        try:
            expected = fn(a, b)
        except ZeroDivisionError:
            continue
        items.append({
            "question": f"What is {name}?",
            "context": f"{label_a}: {a:,.2f}\n{label_b}: {b:,.2f}",
            "expected": round(expected, 2),
            "note": f"{name} from two stated figures", "origin": "generated"})
    return items


def generated_abstention(rng, count):
    missing = [("the net profit margin", "Revenue", "net income"),
               ("return on assets", "Net income", "total assets"),
               ("the quick ratio", "Current assets", "current liabilities"),
               ("the payout ratio", "Dividends paid", "net income"),
               ("asset turnover", "Revenue", "total assets")]
    items = []
    for index in range(count):
        metric, supplied, withheld = missing[index % len(missing)]
        value = round(rng.uniform(100, 9000), 2)
        if index % 4 == 3:
            # Answerable control.
            second = round(value / rng.choice([2.0, 4.0, 5.0]), 2)
            items.append({
                "question": f"What is {metric}?",
                "context": f"{supplied}: {value:,.2f}\n{withheld.capitalize()}: "
                           f"{second:,.2f}",
                "expected": False,
                "note": "control: answerable, must not abstain",
                "origin": "generated"})
        else:
            items.append({
                "question": f"What is {metric}?",
                "context": f"{supplied}: {value:,.2f}",
                "expected": None,
                "note": f"{withheld} withheld", "origin": "generated"})
    return items


def main():
    from data.eval_heldout.handwritten_items import (
        ABSTENTION, CALCULATION, COPY, EXTRACTION, WORDING,
    )

    rng = random.Random(SEED)
    print("collecting hand-written items...")
    gates = {
        "copy": [{"question": q, "context": c, "expected": e, "note": n,
                  "origin": "handwritten"} for q, c, e, n in COPY],
        "extraction": [{"question": q, "context": c, "expected": e, "note": n,
                        "origin": "handwritten"} for q, c, e, n in EXTRACTION],
        "wording": [{"question": q, "context": c, "expected": e, "note": n,
                     "origin": "handwritten"} for q, c, e, n in WORDING],
        "calculation": [{"question": q, "context": c, "expected": e, "note": n,
                         "origin": "handwritten"} for q, c, e, n in CALCULATION],
        "abstention": [{"question": q, "context": c, "expected": e, "note": n,
                        "origin": "handwritten"} for q, c, e, n in ABSTENTION],
    }
    handwritten_counts = {gate: len(items) for gate, items in gates.items()}

    print("generating the remainder...")
    gates["copy"] += generated_copy(rng, 60)
    gates["extraction"] += generated_extraction(rng, 60)
    gates["wording"] += generated_extraction(rng, 70, unseen_wording=True)
    gates["calculation"] += generated_calculation(rng, 40)
    gates["abstention"] += generated_abstention(rng, 32)

    # --- recompute every calculation answer -----------------------------
    # Operands are looked up BY LABEL. A positional checker got this wrong:
    # "EBITDA: 750 / Revenue: 3,000" and "Revenue: X / Net income: Y" put the
    # numerator in opposite positions, and the first version computed an EBITDA
    # margin of 400%.
    def labelled_values(context):
        values = {}
        for line in context.splitlines():
            match = re.match(r"\s*([A-Za-z][A-Za-z' &/-]*?)\s*:\s*"
                             r"\(?(-?[\d,]+\.?\d*)\)?\s*%?\s*$", line)
            if match:
                values[match.group(1).strip().lower()] = float(
                    match.group(2).replace(",", ""))
        return values

    def pick(values, *names):
        for name in names:
            if name in values:
                return values[name]
        return None

    CHECKS = {
        "gross margin as a percentage":
            lambda v: 100.0 * (pick(v, "revenue") - pick(v, "cost of sales"))
            / pick(v, "revenue"),
        "net profit margin as a percentage":
            lambda v: 100.0 * pick(v, "net income") / pick(v, "revenue"),
        "ebitda margin as a percentage":
            lambda v: 100.0 * pick(v, "ebitda") / pick(v, "revenue"),
        "current ratio":
            lambda v: pick(v, "current assets") / pick(v, "current liabilities"),
        "quick ratio":
            lambda v: (pick(v, "current assets") - pick(v, "inventory"))
            / pick(v, "current liabilities"),
        "return on equity as a percentage":
            lambda v: 100.0 * pick(v, "net income") / pick(v, "shareholders' equity"),
        "return on assets as a percentage":
            lambda v: 100.0 * pick(v, "net income") / pick(v, "total assets"),
        "debt-to-equity ratio":
            lambda v: pick(v, "total debt") / pick(v, "shareholders' equity"),
        "debt-to-ebitda ratio":
            lambda v: pick(v, "total debt") / pick(v, "ebitda"),
        "free cash flow":
            lambda v: pick(v, "operating cash flow") - pick(v, "capital expenditure"),
        "working capital":
            lambda v: pick(v, "current assets") - pick(v, "current liabilities"),
        "earnings per share":
            lambda v: pick(v, "net income") / pick(v, "shares outstanding"),
        "interest coverage ratio":
            lambda v: pick(v, "ebit") / pick(v, "interest expense"),
        "asset turnover":
            lambda v: pick(v, "revenue") / pick(v, "total assets"),
        "shareholders' equity":
            lambda v: pick(v, "total assets") - pick(v, "total liabilities"),
        "gross profit":
            lambda v: pick(v, "revenue") - pick(v, "cost of sales"),
        "revenue growth rate as a percentage":
            lambda v: 100.0 * (pick(v, "revenue this year")
                               - pick(v, "revenue last year"))
            / pick(v, "revenue last year"),
        "dividend payout ratio as a percentage":
            lambda v: 100.0 * pick(v, "dividends paid") / pick(v, "net income"),
        "enterprise value":
            lambda v: pick(v, "market capitalisation") + pick(v, "total debt")
            - pick(v, "cash"),
        "price-to-earnings ratio":
            lambda v: pick(v, "share price") / pick(v, "earnings per share"),
        # The alternative label wordings used by the later hand-written items,
        # added after the build reported 10 of 70 items going unverified. An
        # unverified expected value is exactly the thing this check exists for.
        "gearing ratio":
            lambda v: pick(v, "borrowings") / pick(v, "owners' funds"),
        "interest cover":
            lambda v: pick(v, "profit before interest and tax")
            / pick(v, "finance charges"),
        "operating margin as a percentage":
            lambda v: 100.0 * pick(v, "trading profit")
            / pick(v, "sales for the half year"),
        "asset turnover":
            lambda v: pick(v, "turnover for the year", "revenue")
            / pick(v, "total resources employed", "total assets"),
    }

    # Label synonyms, so one formula covers several phrasings.
    SYNONYMS = {
        "revenue": ["sales for the half year", "turnover for the year", "turnover"],
        "cost of sales": ["direct production costs"],
        "current assets": ["assets falling due within one year",
                           "stocks and debtors and cash"],
        "current liabilities": ["liabilities falling due within one year",
                                "creditors due within one year"],
        "net income": ["profit attributable to owners", "profit after taxation"],
        "shareholders' equity": ["total owners' funds", "owners' funds"],
        "operating cash flow": ["net cash inflow from trading"],
        "capital expenditure": ["payments for fixed assets"],
        "shares outstanding": ["ordinary shares in issue"],
        "total assets": ["total resources employed"],
        "total debt": ["borrowings"],
        "ebit": ["profit before interest and tax"],
        "interest expense": ["finance charges"],
    }

    bad, checked = [], 0
    for item in gates["calculation"]:
        question = item["question"].lower().rstrip("?")
        values = labelled_values(item["context"])
        for canonical, alternatives in SYNONYMS.items():
            if canonical not in values:
                for alternative in alternatives:
                    if alternative in values:
                        values[canonical] = values[alternative]
                        break
        for name, formula in CHECKS.items():
            if name in question:
                try:
                    expected = formula(values)
                except (TypeError, ZeroDivisionError):
                    expected = None
                if expected is None:
                    break
                checked += 1
                if abs(expected - float(item["expected"])) > 0.02:
                    bad.append((item["question"],
                                item["context"].replace("\n", " | "),
                                item["expected"], round(expected, 2)))
                break
    print(f"recomputed {checked} of {len(gates['calculation'])} calculation items")
    if checked < len(gates["calculation"]):
        unchecked = len(gates["calculation"]) - checked
        print(f"\nBUILD FAILED: {unchecked} calculation items could not be "
              f"recomputed, so their expected values are unverified. Add the "
              f"formula or the label synonym rather than shipping them.")
        sys.exit(1)
    if bad:
        print("\nCALCULATION ANSWERS DISAGREE WITH RECOMPUTATION:")
        for entry in bad[:8]:
            print(f"  {entry}")
        sys.exit(1)

    # --- overlap with training data -------------------------------------
    print("checking overlap with every training file...")
    exact, grams = training_fingerprints()
    removed = {"exact": 0, "ngram": 0, "by_gate": {}}
    cleaned = {}
    for gate, items in gates.items():
        kept, dropped = [], 0
        for item in items:
            text = f"{item['question']}\n{item['context']}".strip().lower()
            if text in exact:
                removed["exact"] += 1
                dropped += 1
                continue
            overlap = ngrams(text) & grams
            if overlap:
                removed["ngram"] += 1
                dropped += 1
                continue
            kept.append(item)
        removed["by_gate"][gate] = dropped
        cleaned[gate] = kept

    # --- split: selection set kept separate from the frozen set ---------
    frozen, selection = [], []
    for gate, items in cleaned.items():
        rng.shuffle(items)
        # The selection split is drawn from GENERATED items only. Its job is to
        # choose checkpoints, and hand-written items are the scarce, deliberate
        # part of the evaluation - they belong in the frozen set where the
        # reporting happens.
        generated = [i for i in items if i["origin"] == "generated"]
        handwritten = [i for i in items if i["origin"] == "handwritten"]
        cut = max(6, len(generated) // 4)
        for index, item in enumerate(generated[:cut]):
            selection.append({"id": f"{gate}_sel_{index:03d}", "gate": gate, **item})
        for index, item in enumerate(handwritten + generated[cut:]):
            frozen.append({"id": f"{gate}_{index:03d}", "gate": gate, **item})

    # --- the minimums the brief specified --------------------------------
    problems = []
    frozen_by_gate = {}
    for gate in GATE_MINIMUMS:
        items = [r for r in frozen if r["gate"] == gate]
        frozen_by_gate[gate] = len(items)
        if len(items) < GATE_MINIMUMS[gate]:
            problems.append(f"{gate}: {len(items)} frozen items, "
                            f"minimum {GATE_MINIMUMS[gate]}")
        hand = sum(1 for r in items if r["origin"] == "handwritten")
        if hand < HANDWRITTEN_MINIMUM:
            problems.append(f"{gate}: {hand} hand-written items in the frozen "
                            f"set, minimum {HANDWRITTEN_MINIMUM}")
    if problems:
        print("\nBUILD FAILED - minimums not met:")
        for problem in problems:
            print(f"  - {problem}")
        sys.exit(1)

    os.makedirs(OUT_DIR, exist_ok=True)
    for path, rows in ((FROZEN, frozen), (SELECTION, selection)):
        with open(path, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")

    manifest = {
        "frozen": {"path": FROZEN, "items": len(frozen),
                   "sha256": hashlib.sha256(open(FROZEN, "rb").read()).hexdigest(),
                   "by_gate": frozen_by_gate,
                   "handwritten_by_gate": {
                       gate: sum(1 for r in frozen if r["gate"] == gate
                                 and r["origin"] == "handwritten")
                       for gate in GATE_MINIMUMS}},
        "selection": {"path": SELECTION, "items": len(selection),
                      "sha256": hashlib.sha256(open(SELECTION, "rb").read()).hexdigest(),
                      "purpose": "checkpoint choice only; never used for reporting"},
        "seed": SEED,
        "overlap_removed": removed,
        "overlap_method": f"exact text match plus {NGRAM}-gram intersection against "
                          f"{len(TRAINING_FILES)} training files",
        "training_files_checked": TRAINING_FILES,
        "rules": [
            "never train on the frozen set",
            "never select checkpoints with the frozen set - use selection.jsonl",
            "frozen items are reported with k/n and a confidence interval",
        ],
    }
    with open(MANIFEST, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    print(f"\nfrozen: {len(frozen)} items, sha256 "
          f"{manifest['frozen']['sha256'][:16]}...")
    print(f"selection: {len(selection)} items (checkpoint choice only)")
    print(f"\n{'gate':<14}{'frozen':>8}{'hand':>7}{'removed':>9}")
    for gate in GATE_MINIMUMS:
        print(f"{gate:<14}{frozen_by_gate[gate]:>8}"
              f"{manifest['frozen']['handwritten_by_gate'][gate]:>7}"
              f"{removed['by_gate'][gate]:>9}")
    print(f"\noverlap removed: {removed['exact']} exact, {removed['ngram']} "
          f"{NGRAM}-gram near-duplicates")
    print(f"\nwrote {FROZEN}\nwrote {SELECTION}\nwrote {MANIFEST}")


if __name__ == "__main__":
    main()
