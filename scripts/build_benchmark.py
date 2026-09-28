"""Build the Aivora financial benchmark: data/benchmark/{dev,test}.jsonl.

Two kinds of item, and the manifest records how many of each, because the
distinction matters when reading a score:

* GENERATED - a template with randomised figures whose answer is computed
  exactly in this file (margins, ratios, statement arithmetic, extraction,
  valuation). These are verifiable by construction: the expected value is
  arithmetic, not opinion.
* AUTHORED - written here with a grading rubric (concept groups that an answer
  must express). Used where the answer is prose: concepts, reporting/SEC
  knowledge, interpretation, and the abstention cases.

Splits are disjoint BY CONSTRUCTION, not by shuffling: dev and test draw from
separate random streams and separate authored pools, so no test item is a
re-rolled twin of a dev item. The test split is the hidden set - nothing in
the training or tuning pipeline may read data/benchmark/test.jsonl.

usage:
    python scripts/build_benchmark.py            # writes dev + test + manifest
    python scripts/build_benchmark.py --check    # validate an existing build
"""

import argparse
import hashlib
import json
import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

OUT_DIR = os.path.join("data", "benchmark")

# The grading markers live with the scorer, not here, so the two can never
# drift apart. Imported only to record them in the manifest.
from evaluation.financial_metrics import ABSTENTION_MARKERS  # noqa: E402


# --------------------------------------------------------------------------
# generated items: the answer is computed here, so it is right by construction
# --------------------------------------------------------------------------

def _money(rng, low=50, high=5000, step=5):
    return float(rng.randrange(low, high, step))


def gen_ratio_items(rng, n, split):
    """Single-formula ratio and margin questions."""
    items = []
    makers = [
        ("gross_margin", "Revenue is {revenue:,.2f} and gross profit is {gross:,.2f}. "
                         "What is the gross margin?", "%"),
        ("net_profit_margin", "Net income is {net:,.2f} and revenue is {revenue:,.2f}. "
                              "What is the net profit margin?", "%"),
        ("current_ratio", "Current assets are {ca:,.2f} and current liabilities are "
                          "{cl:,.2f}. What is the current ratio?", "x"),
        ("debt_to_equity", "Total debt is {debt:,.2f} and shareholders' equity is "
                           "{equity:,.2f}. What is the debt-to-equity ratio?", "x"),
        ("roa", "Net income is {net:,.2f} and total assets are {assets:,.2f}. "
                "What is ROA?", "%"),
        ("roic", "NOPAT is {nopat:,.2f} and invested capital is {ic:,.2f}. "
                 "What is ROIC?", "%"),
        ("ebitda_margin", "EBITDA is {ebitda:,.2f} and revenue is {revenue:,.2f}. "
                          "What is the EBITDA margin?", "%"),
        ("quick_ratio", "Current assets are {ca:,.2f}, inventory is {inv:,.2f}, and "
                        "current liabilities are {cl:,.2f}. What is the quick ratio?", "x"),
        ("asset_turnover", "Revenue is {revenue:,.2f} and total assets are {assets:,.2f}. "
                           "What is asset turnover?", "x"),
        ("interest_coverage", "EBIT is {ebit:,.2f} and interest expense is {interest:,.2f}. "
                              "What is the interest coverage ratio?", "x"),
    ]
    for i in range(n):
        name, template, unit = makers[i % len(makers)]
        revenue = _money(rng, 500, 5000)
        values = {
            "revenue": revenue,
            "gross": round(revenue * rng.choice([0.2, 0.25, 0.3, 0.4, 0.5]), 2),
            "net": round(revenue * rng.choice([0.05, 0.1, 0.12, 0.15, 0.2]), 2),
            "ebitda": round(revenue * rng.choice([0.1, 0.2, 0.25, 0.3]), 2),
            "assets": _money(rng, 1000, 8000),
            "ca": _money(rng, 200, 2000),
            "cl": _money(rng, 100, 1000),
            "inv": _money(rng, 50, 400),
            "debt": _money(rng, 200, 3000),
            "equity": _money(rng, 200, 3000),
            "nopat": _money(rng, 50, 900),
            "ic": _money(rng, 500, 6000),
            "ebit": _money(rng, 100, 1200),
            "interest": _money(rng, 10, 200, 5),
        }
        answers = {
            "gross_margin": 100 * values["gross"] / values["revenue"],
            "net_profit_margin": 100 * values["net"] / values["revenue"],
            "current_ratio": values["ca"] / values["cl"],
            "debt_to_equity": values["debt"] / values["equity"],
            "roa": 100 * values["net"] / values["assets"],
            "roic": 100 * values["nopat"] / values["ic"],
            "ebitda_margin": 100 * values["ebitda"] / values["revenue"],
            "quick_ratio": (values["ca"] - values["inv"]) / values["cl"],
            "asset_turnover": values["revenue"] / values["assets"],
            "interest_coverage": values["ebit"] / values["interest"],
        }
        value = round(answers[name], 2)
        items.append({
            "id": f"{split}_ratio_{i:04d}", "category": "ratios", "level": 2,
            "kind": "generated", "formula": name,
            "question": template.format(**values),
            "answer": f"{value}{unit}" if unit == "%" else f"{value}",
            "numeric_answer": value, "tolerance": 0.05,
        })
    return items


def gen_statement_items(rng, n, split):
    """A small income statement / balance sheet, then extraction and derivation."""
    items = []
    for i in range(n):
        revenue = _money(rng, 1000, 9000)
        cogs = round(revenue * rng.choice([0.5, 0.55, 0.6, 0.65]), 2)
        opex = round(revenue * rng.choice([0.1, 0.15, 0.2]), 2)
        da = round(revenue * rng.choice([0.02, 0.04, 0.05]), 2)
        interest = _money(rng, 10, 200, 5)
        tax_rate = rng.choice([0.2, 0.25, 0.3])
        gross = revenue - cogs
        ebitda = gross - opex
        ebit = ebitda - da
        pbt = ebit - interest
        tax = round(pbt * tax_rate, 2)
        net = round(pbt - tax, 2)
        assets = _money(rng, 4000, 20000)
        liabilities = round(assets * rng.choice([0.3, 0.4, 0.5, 0.6]), 2)
        equity = round(assets - liabilities, 2)

        statement = (
            f"Income statement (in $000s):\n"
            f"  Revenue {revenue:,.2f}\n  Cost of goods sold {cogs:,.2f}\n"
            f"  Operating expenses {opex:,.2f}\n  Depreciation & amortization {da:,.2f}\n"
            f"  Interest expense {interest:,.2f}\n  Tax rate {tax_rate:.0%}\n"
            f"Balance sheet (in $000s):\n"
            f"  Total assets {assets:,.2f}\n  Total liabilities {liabilities:,.2f}\n"
        )
        asks = [
            ("What is gross profit?", gross, "statements", 2),
            ("What is EBITDA?", ebitda, "statements", 3),
            ("What is EBIT?", ebit, "statements", 3),
            ("What is net income?", net, "statements", 4),
            ("What is shareholders' equity?", equity, "accounting", 2),
            ("What is the gross margin as a percentage?", round(100 * gross / revenue, 2),
             "statements", 3),
        ]
        question, value, category, level = asks[i % len(asks)]
        items.append({
            "id": f"{split}_stmt_{i:04d}", "category": category, "level": level,
            "kind": "generated", "question": f"{statement}\n{question}",
            "answer": f"{value:,.2f}", "numeric_answer": round(value, 2), "tolerance": 0.5,
        })
    return items


def gen_extraction_items(rng, n, split):
    """Read one figure back out of a statement - field-level accuracy."""
    items = []
    fields = ["Revenue", "Net income", "Total assets", "Total liabilities", "Cash",
              "Total debt", "EPS", "EBITDA", "Free cash flow"]
    for i in range(n):
        values = {
            "Revenue": _money(rng, 1000, 9000), "Net income": _money(rng, 50, 900),
            "Total assets": _money(rng, 4000, 20000),
            "Total liabilities": _money(rng, 1000, 9000),
            "Cash": _money(rng, 100, 3000), "Total debt": _money(rng, 200, 4000),
            "EPS": round(rng.uniform(0.5, 12), 2), "EBITDA": _money(rng, 200, 2500),
            "Free cash flow": _money(rng, 50, 1500),
        }
        body = "\n".join(f"  {k}: {v:,.2f}" for k, v in values.items())
        field = fields[i % len(fields)]
        items.append({
            "id": f"{split}_extract_{i:04d}", "category": "extraction", "level": 1,
            "kind": "generated",
            "question": f"Financial summary (in $000s):\n{body}\n\nWhat is {field}?",
            "answer": f"{values[field]:,.2f}", "numeric_answer": round(values[field], 2),
            "tolerance": 0.01, "field": field,
        })
    return items


def gen_valuation_items(rng, n, split):
    items = []
    for i in range(n):
        kind = i % 5
        if kind == 0:
            mcap = _money(rng, 1000, 20000)
            debt = _money(rng, 200, 5000)
            cash = _money(rng, 50, 2000)
            value = mcap + debt - cash
            q = (f"Market capitalisation is {mcap:,.2f}, total debt is {debt:,.2f}, "
                 f"and cash is {cash:,.2f}. What is enterprise value?")
        elif kind == 1:
            ev = _money(rng, 2000, 20000)
            ebitda = _money(rng, 200, 2000)
            value = round(ev / ebitda, 2)
            q = (f"Enterprise value is {ev:,.2f} and EBITDA is {ebitda:,.2f}. "
                 f"What is the EV/EBITDA multiple?")
        elif kind == 2:
            price = round(rng.uniform(10, 300), 2)
            eps = round(rng.uniform(0.5, 15), 2)
            value = round(price / eps, 2)
            q = (f"The share price is {price:,.2f} and EPS is {eps:,.2f}. "
                 f"What is the P/E ratio?")
        elif kind == 3:
            begin = _money(rng, 100, 1000)
            years = rng.choice([2, 3, 4, 5])
            growth = rng.choice([0.05, 0.1, 0.15, 0.2])
            end = round(begin * (1 + growth) ** years, 2)
            value = round(100 * ((end / begin) ** (1 / years) - 1), 2)
            q = (f"An investment grew from {begin:,.2f} to {end:,.2f} over {years} years. "
                 f"What is the CAGR?")
        else:
            net = _money(rng, 100, 900)
            shares = _money(rng, 50, 500, 5)
            value = round(net / shares, 2)
            q = (f"Net income is {net:,.2f} and there are {shares:,.2f} shares "
                 f"outstanding. What is EPS?")
        items.append({
            "id": f"{split}_val_{i:04d}", "category": "valuation", "level": 3,
            "kind": "generated", "question": q, "answer": f"{value:,.2f}",
            "numeric_answer": round(value, 2), "tolerance": 0.05,
        })
    return items


def gen_corporate_finance_items(rng, n, split):
    items = []
    for i in range(n):
        kind = i % 4
        if kind == 0:
            e_w = rng.choice([0.4, 0.5, 0.6, 0.7])
            re = rng.choice([0.08, 0.1, 0.12])
            rd = rng.choice([0.03, 0.04, 0.05])
            tax = rng.choice([0.2, 0.25, 0.3])
            value = round(100 * (e_w * re + (1 - e_w) * rd * (1 - tax)), 2)
            q = (f"Equity is {e_w:.0%} of capital at a cost of {re:.0%}, debt is the rest "
                 f"at {rd:.0%}, and the tax rate is {tax:.0%}. What is the WACC?")
        elif kind == 1:
            net = _money(rng, 200, 2000)
            dividends = round(net * rng.choice([0.2, 0.3, 0.4, 0.5]), 2)
            value = round(100 * dividends / net, 2)
            q = (f"Net income is {net:,.2f} and dividends paid are {dividends:,.2f}. "
                 f"What is the dividend payout ratio?")
        elif kind == 2:
            ocf = _money(rng, 300, 3000)
            capex = _money(rng, 50, 1200)
            value = round(ocf - capex, 2)
            q = (f"Operating cash flow is {ocf:,.2f} and capital expenditure is "
                 f"{capex:,.2f}. What is free cash flow?")
        else:
            ebitda = _money(rng, 200, 2000)
            debt = _money(rng, 500, 8000)
            value = round(debt / ebitda, 2)
            q = (f"Total debt is {debt:,.2f} and EBITDA is {ebitda:,.2f}. "
                 f"What is the debt-to-EBITDA ratio?")
        items.append({
            "id": f"{split}_corp_{i:04d}", "category": "corporate_finance", "level": 3,
            "kind": "generated", "question": q, "answer": f"{value:,.2f}",
            "numeric_answer": round(value, 2), "tolerance": 0.05,
        })
    return items


def gen_multistep_items(rng, n, split):
    """Level 5: two or more steps before the answer."""
    items = []
    for i in range(n):
        kind = i % 4
        if kind == 0:
            assets = _money(rng, 2000, 12000)
            liabilities = round(assets * rng.choice([0.3, 0.45, 0.6]), 2)
            net = _money(rng, 100, 900)
            value = round(100 * net / (assets - liabilities), 2)
            q = (f"Total assets are {assets:,.2f}, total liabilities are {liabilities:,.2f}, "
                 f"and net income is {net:,.2f}. What is ROE?")
        elif kind == 1:
            net = _money(rng, 100, 900)
            shares = _money(rng, 50, 400, 5)
            price = round(rng.uniform(10, 200), 2)
            value = round(price / (net / shares), 2)
            q = (f"Net income is {net:,.2f}, shares outstanding are {shares:,.2f}, and the "
                 f"share price is {price:,.2f}. What is the P/E ratio?")
        elif kind == 2:
            ebitda = _money(rng, 300, 3000)
            dep = _money(rng, 20, 400)
            amort = _money(rng, 10, 200)
            value = round(ebitda - dep - amort, 2)
            q = (f"EBITDA is {ebitda:,.2f}, depreciation is {dep:,.2f}, and amortization is "
                 f"{amort:,.2f}. What is EBIT?")
        else:
            revenue = _money(rng, 1000, 8000)
            cogs = round(revenue * rng.choice([0.5, 0.6, 0.7]), 2)
            opex = round(revenue * rng.choice([0.1, 0.15]), 2)
            value = round(100 * (revenue - cogs - opex) / revenue, 2)
            q = (f"Revenue is {revenue:,.2f}, COGS is {cogs:,.2f}, and operating expenses "
                 f"are {opex:,.2f}. What is the operating margin?")
        items.append({
            "id": f"{split}_multi_{i:04d}", "category": "reasoning", "level": 5,
            "kind": "generated", "question": q, "answer": f"{value:,.2f}",
            "numeric_answer": round(value, 2), "tolerance": 0.05,
        })
    return items


def gen_abstention_items(rng, n, split):
    """The correct answer is a refusal: the data is missing or not knowable."""
    missing = [
        ("Revenue is {a:,.2f}. What is the net profit margin?",
         "net income is not given"),
        ("Total assets are {a:,.2f}. What is ROE?", "equity and net income are not given"),
        ("EBITDA is {a:,.2f}. What is the EBITDA margin?", "revenue is not given"),
        ("Current assets are {a:,.2f}. What is the current ratio?",
         "current liabilities are not given"),
        ("Net income is {a:,.2f}. What is EPS?", "the share count is not given"),
        ("Operating cash flow is {a:,.2f}. What is free cash flow?",
         "capital expenditure is not given"),
    ]
    unknowable = [
        "What was Apple's revenue yesterday?",
        "What is Tesla's share price right now?",
        "How much did Microsoft earn this quarter?",
        "What will the S&P 500 close at tomorrow?",
        "What is the current Federal Reserve interest rate today?",
        "What were Amazon's exact operating expenses last month?",
    ]
    items = []
    for i in range(n):
        if i % 2 == 0:
            template, why = missing[(i // 2) % len(missing)]
            question = template.format(a=_money(rng, 100, 5000))
            note = f"insufficient information: {why}"
        else:
            question = unknowable[(i // 2) % len(unknowable)]
            note = "requires live market data the system does not have"
        items.append({
            "id": f"{split}_abstain_{i:04d}", "category": "hallucination", "level": 4,
            "kind": "generated", "question": question,
            "answer": f"Insufficient information - {note}.",
            "must_abstain": True, "reason": note,
        })
    return items


# --------------------------------------------------------------------------
# authored items: prose answers, graded by rubric (concept groups)
# --------------------------------------------------------------------------

def _load_authored():
    """data/benchmark/authored_pool.py holds the written questions and rubrics.

    Loaded by path, not imported as a package: data/ is a data directory and
    should not become one."""
    import importlib.util

    pool_path = os.path.join(OUT_DIR, "authored_pool.py")
    spec = importlib.util.spec_from_file_location("authored_pool", pool_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.DEV, module.TEST


AUTHORED_DEV, AUTHORED_TEST = _load_authored()


def authored_items(pool, split):
    items = []
    for i, (category, level, question, answer, rubric) in enumerate(pool):
        items.append({
            "id": f"{split}_auth_{i:04d}", "category": category, "level": level,
            "kind": "authored", "question": question, "answer": answer,
            "required_any": rubric,
        })
    return items


# --------------------------------------------------------------------------

def build_split(split, seed, counts, authored_pool):
    rng = random.Random(seed)
    items = []
    items += gen_ratio_items(rng, counts["ratios"], split)
    items += gen_statement_items(rng, counts["statements"], split)
    items += gen_extraction_items(rng, counts["extraction"], split)
    items += gen_valuation_items(rng, counts["valuation"], split)
    items += gen_corporate_finance_items(rng, counts["corporate_finance"], split)
    items += gen_multistep_items(rng, counts["reasoning"], split)
    items += gen_abstention_items(rng, counts["hallucination"], split)
    items += authored_items(authored_pool, split)
    return items


def dedup(items):
    seen, out, dropped = set(), [], 0
    for item in items:
        key = hashlib.sha256(item["question"].strip().lower().encode()).hexdigest()
        if key in seen:
            dropped += 1
            continue
        seen.add(key)
        out.append(item)
    return out, dropped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="validate an existing build")
    args = ap.parse_args()

    # Sized so that generated + authored lands near the planned 800 / 200 after
    # duplicate templates are dropped.
    dev_counts = {"ratios": 140, "statements": 140, "extraction": 100, "valuation": 100,
                  "corporate_finance": 100, "reasoning": 110, "hallucination": 60}
    test_counts = {"ratios": 35, "statements": 35, "extraction": 25, "valuation": 25,
                   "corporate_finance": 20, "reasoning": 25, "hallucination": 20}

    if not args.check:
        os.makedirs(OUT_DIR, exist_ok=True)
        # Separate seeds AND separate authored pools: the hidden split is not a
        # re-roll of the same templates with the same numbers.
        dev = build_split("dev", 20260928, dev_counts, AUTHORED_DEV)
        test = build_split("test", 777001, test_counts, AUTHORED_TEST)
        dev, dev_dropped = dedup(dev)
        test, test_dropped = dedup(test)

        dev_questions = {i["question"].strip().lower() for i in dev}
        overlap = [i["id"] for i in test if i["question"].strip().lower() in dev_questions]
        test = [i for i in test if i["question"].strip().lower() not in dev_questions]

        # Shuffled deterministically so that evaluating a PREFIX (a --limit run
        # during development) samples every category, not just the first
        # generator's output.
        random.Random(11).shuffle(dev)
        random.Random(12).shuffle(test)

        for name, rows in (("dev", dev), ("test", test)):
            with open(os.path.join(OUT_DIR, f"{name}.jsonl"), "w", encoding="utf-8") as f:
                for row in rows:
                    f.write(json.dumps(row) + "\n")

        manifest = {
            "dev_examples": len(dev),
            "test_examples": len(test),
            "duplicates_dropped": {"dev": dev_dropped, "test": test_dropped},
            "test_items_removed_for_overlapping_dev": len(overlap),
            "by_kind": {
                "generated": sum(1 for i in dev + test if i["kind"] == "generated"),
                "authored": sum(1 for i in dev + test if i["kind"] == "authored"),
            },
            "by_category": {},
            "by_level": {},
            "graded_by": {
                "numeric_tolerance": sum(1 for i in dev + test if i.get("numeric_answer") is not None),
                "rubric": sum(1 for i in dev + test if i.get("required_any")),
                "abstention": sum(1 for i in dev + test if i.get("must_abstain")),
            },
            "abstention_markers": list(ABSTENTION_MARKERS),
            "note": ("test.jsonl is the hidden split: never used for training, tuning, "
                     "prompt design or glossary edits."),
        }
        for item in dev + test:
            manifest["by_category"][item["category"]] = \
                manifest["by_category"].get(item["category"], 0) + 1
            manifest["by_level"][str(item["level"])] = \
                manifest["by_level"].get(str(item["level"]), 0) + 1
        with open(os.path.join(OUT_DIR, "manifest.json"), "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        print(json.dumps(manifest, indent=2))

    # validation, on whatever is on disk
    rows = {}
    for name in ("dev", "test"):
        with open(os.path.join(OUT_DIR, f"{name}.jsonl"), encoding="utf-8") as f:
            rows[name] = [json.loads(line) for line in f if line.strip()]
    ids = [r["id"] for r in rows["dev"] + rows["test"]]
    assert len(ids) == len(set(ids)), "duplicate ids"
    dev_q = {r["question"].strip().lower() for r in rows["dev"]}
    leaked = [r["id"] for r in rows["test"] if r["question"].strip().lower() in dev_q]
    assert not leaked, f"test questions also in dev: {leaked[:5]}"
    for r in rows["dev"] + rows["test"]:
        assert r.get("numeric_answer") is not None or r.get("required_any") \
            or r.get("must_abstain"), f"{r['id']} has no gradable target"
    print(f"\nvalidated: dev {len(rows['dev'])}, test {len(rows['test'])}, "
          f"no id or question overlap, every item gradable")


if __name__ == "__main__":
    main()
