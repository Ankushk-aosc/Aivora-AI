"""The benchmark and its scorer must be trustworthy before any score is quoted.

Two things are checked. First the dataset: sizes, split hygiene (the hidden
test split must share no question or id with dev), every item gradable, and
that generated answers really are arithmetically right - a benchmark whose own
answers are wrong is worse than none. Second the scorer: a right answer, a
wrong assertion, a refusal on an answerable question and a refusal on an
unanswerable one are four different outcomes and must not be conflated.

Run: deepseek_env/Scripts/python.exe tests/test_benchmark.py
"""
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.benchmark import (  # noqa: E402
    load_benchmark, score_item, summarise,
)

PASSED, FAILED = [], []


def check(name, ok, detail=""):
    (PASSED if ok else FAILED).append(name)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" - {detail}" if detail else ""))


def check_dataset():
    dev = load_benchmark("dev")
    test = load_benchmark("test")
    check("dev split is 750-850 items", 750 <= len(dev) <= 850, f"{len(dev)}")
    check("test split is 180-230 items", 180 <= len(test) <= 230, f"{len(test)}")
    check("total is about 1000", 950 <= len(dev) + len(test) <= 1050,
          f"{len(dev) + len(test)}")

    ids = [i["id"] for i in dev + test]
    check("ids unique", len(ids) == len(set(ids)),
          f"{len(ids) - len(set(ids))} duplicates")

    dev_questions = {i["question"].strip().lower() for i in dev}
    leaked = [i["id"] for i in test if i["question"].strip().lower() in dev_questions]
    check("hidden split shares no question with dev", not leaked, str(leaked[:3]))

    # The authored pools are disjoint by construction; verify it held.
    dev_auth = {i["question"] for i in dev if i["kind"] == "authored"}
    test_auth = {i["question"] for i in test if i["kind"] == "authored"}
    check("authored pools disjoint", not (dev_auth & test_auth),
          str(sorted(dev_auth & test_auth)[:2]))

    ungradable = [i["id"] for i in dev + test
                  if i.get("numeric_answer") is None and not i.get("required_any")
                  and not i.get("must_abstain")]
    check("every item gradable", not ungradable, str(ungradable[:3]))

    # Coverage the plan asks for.
    categories = {i["category"] for i in dev + test}
    for needed in ("ratios", "statements", "accounting", "valuation",
                   "corporate_finance", "reasoning", "extraction",
                   "hallucination", "concepts", "reporting", "interpretation",
                   "general"):
        check(f"category present: {needed}", needed in categories)
    levels = {i["level"] for i in dev + test}
    check("difficulty levels 1-5 present", levels == {1, 2, 3, 4, 5}, str(sorted(levels)))
    check("has abstention items", sum(1 for i in dev + test if i.get("must_abstain")) >= 40)
    check("has rubric items", sum(1 for i in dev + test if i.get("required_any")) >= 90)

    # Rubrics must be groups of alternatives, not a flat keyword list.
    malformed = [i["id"] for i in dev + test if i.get("required_any")
                 and not all(isinstance(g, (list, tuple)) for g in i["required_any"])]
    check("rubrics are concept groups", not malformed, str(malformed[:3]))

    # Answers must agree with the figures the question actually prints. This
    # caught a real defect: figures were displayed rounded to whole numbers
    # while the answer was computed from the unrounded value, so the stated
    # numbers did not produce the stated answer.
    FORMULAS = {
        "gross_margin": lambda n: 100 * n[1] / n[0],
        "net_profit_margin": lambda n: 100 * n[0] / n[1],
        "current_ratio": lambda n: n[0] / n[1],
        "debt_to_equity": lambda n: n[0] / n[1],
        "roa": lambda n: 100 * n[0] / n[1],
        "roic": lambda n: 100 * n[0] / n[1],
        "ebitda_margin": lambda n: 100 * n[0] / n[1],
        "quick_ratio": lambda n: (n[0] - n[1]) / n[2],
        "asset_turnover": lambda n: n[0] / n[1],
        "interest_coverage": lambda n: n[0] / n[1],
    }
    NUMBER_RE = r"-?\d{1,3}(?:,\d{3})+(?:\.\d+)?|-?\d+(?:\.\d+)?"
    recomputed, mismatched = 0, []
    for item in dev + test:
        formula = item.get("formula")
        if formula not in FORMULAS:
            continue
        numbers = [float(n.replace(",", ""))
                   for n in re.findall(NUMBER_RE, item["question"])]
        expected = round(FORMULAS[formula](numbers), 2)
        recomputed += 1
        if abs(expected - item["numeric_answer"]) > 0.011:
            mismatched.append((item["id"], formula, expected, item["numeric_answer"]))
    check(f"ratio answers recompute from the printed figures ({recomputed} checked)",
          recomputed > 150 and not mismatched, str(mismatched[:3]))

    # Same for the statement items: derive net income from the printed lines.
    stmt_checked, stmt_bad = 0, []
    for item in dev + test:
        if not item["id"].split("_", 1)[1].startswith("stmt"):
            continue
        if not item["question"].rstrip().endswith("What is net income?"):
            continue
        text = item["question"]

        def figure(label):
            line = next(l for l in text.splitlines() if l.strip().startswith(label))
            return float(re.findall(NUMBER_RE, line)[-1].replace(",", ""))

        revenue, cogs = figure("Revenue"), figure("Cost of goods sold")
        opex, da = figure("Operating expenses"), figure("Depreciation")
        interest = figure("Interest expense")
        rate = figure("Tax rate") / 100.0
        pbt = revenue - cogs - opex - da - interest
        expected = round(pbt - round(pbt * rate, 2), 2)
        stmt_checked += 1
        if abs(expected - item["numeric_answer"]) > 0.02:
            stmt_bad.append((item["id"], expected, item["numeric_answer"]))
    check(f"statement net income recomputes ({stmt_checked} checked)",
          stmt_checked > 10 and not stmt_bad, str(stmt_bad[:3]))

    # The tolerance must be tight enough to fail a genuinely wrong answer.
    loose = [i["id"] for i in dev + test if i.get("numeric_answer") is not None
             and i.get("tolerance", 0.01) > max(0.5, abs(i["numeric_answer"]) * 0.01)]
    check("numeric tolerances tight", not loose, f"{len(loose)} loose: {loose[:3]}")
    return dev, test


def check_scorer(dev):
    numeric = next(i for i in dev if i.get("numeric_answer") is not None)
    rubric = next(i for i in dev if i.get("required_any"))
    abstain = next(i for i in dev if i.get("must_abstain"))

    exact = score_item(numeric, f"The answer is {numeric['numeric_answer']}.")
    check("numeric: right value passes", exact["correct"], exact["outcome"])
    wrong = score_item(numeric, f"The answer is {numeric['numeric_answer'] + 99}.")
    check("numeric: wrong value is a hallucination",
          not wrong["correct"] and wrong["outcome"] == "hallucination", wrong["outcome"])
    ducked = score_item(numeric, "There is insufficient information to answer.")
    check("numeric: refusing an answerable question is over-abstention",
          ducked["outcome"] == "over_abstention" and not ducked["correct"],
          ducked["outcome"])
    # Commas must not break parsing (they did until extract_numbers was fixed).
    comma = score_item(numeric, f"About {numeric['numeric_answer']:,.2f} in total.")
    check("numeric: comma-formatted answer passes", comma["correct"], comma["prediction"])

    rubric_pass = score_item(rubric, rubric["answer"])
    check("rubric: reference answer passes", rubric_pass["correct"],
          f"{rubric['id']}: {rubric['answer'][:50]}")
    rubric_fail = score_item(rubric, "It is a financial term used in business.")
    check("rubric: vague answer fails", not rubric_fail["correct"],
          rubric_fail["outcome"])

    good_abstain = score_item(abstain, "Insufficient information to compute that.")
    check("abstention: declining is correct", good_abstain["correct"]
          and good_abstain["outcome"] == "correct_abstention", good_abstain["outcome"])
    bad_abstain = score_item(abstain, "The margin is 42%.")
    check("abstention: inventing a figure is a hallucination",
          not bad_abstain["correct"] and bad_abstain["outcome"] == "hallucination",
          bad_abstain["outcome"])

    # A system that refuses everything must score 0 and show a 100%
    # over-abstention rate, not a flattering 0% hallucination rate alone.
    records = [score_item(i, "I don't know.") for i in dev[:200]]
    summary = summarise(records)
    answerable = summary["answerable"]["total"]
    check("refuse-everything scores near zero",
          summary["overall"]["accuracy"] < 5, str(summary["overall"]))
    check("refuse-everything shows 100% over-abstention",
          summary["over_abstention_rate"] == 100.0 and answerable > 0,
          str(summary["over_abstention_rate"]))
    check("refuse-everything shows 0% hallucination",
          summary["hallucination_rate"] == 0.0, str(summary["hallucination_rate"]))

    # A system that always asserts the same wrong thing: 0% correct, high
    # hallucination, and abstention accuracy 0.
    records = [score_item(i, "The answer is 7.") for i in dev[:200]]
    summary = summarise(records)
    check("assert-always shows high hallucination",
          summary["hallucination_rate"] > 90, str(summary["hallucination_rate"]))
    check("assert-always fails every abstention item",
          summary["abstention_accuracy"] in (0, 0.0, None),
          str(summary["abstention_accuracy"]))


def check_manifest():
    path = os.path.join("data", "benchmark", "manifest.json")
    check("manifest exists", os.path.exists(path))
    if not os.path.exists(path):
        return
    manifest = json.load(open(path, encoding="utf-8"))
    for key in ("dev_examples", "test_examples", "by_kind", "by_category",
                "by_level", "graded_by"):
        check(f"manifest records {key}", key in manifest)
    check("manifest states the hidden-split rule",
          "never used for training" in manifest.get("note", ""))


def main():
    dev, _ = check_dataset()
    check_scorer(dev)
    check_manifest()
    print(f"\n{len(PASSED)}/{len(PASSED) + len(FAILED)} passed")
    sys.exit(1 if FAILED else 0)


if __name__ == "__main__":
    main()
