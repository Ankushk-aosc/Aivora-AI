"""Regression tests for every failure group the baseline measured.

reports/baseline_analysis.md grouped 293 of 331 dev-split failures into 13
causes. Each cause gets a test here, with the concrete question that failed, so
a future change that reintroduces one is caught rather than rediscovered by a
three-hour evaluation run.

The rules these encode:

* a question whose answer is STATED must be read back, never replaced by a
  formula computed from the neighbouring figures;
* a question whose figures are present must be computed, not declined;
* a question asking for a CURRENT value must be refused - a definition of X is
  never an answer to the current value of X;
* a question missing a figure must still be declined, not guessed. Fixing
  over-abstention must not turn into a licence to guess.

Run: deepseek_env/Scripts/python.exe tests/test_intent_routing.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.backend.services.chat_service import FinancialChat  # noqa: E402
from app.backend.services.financial_router import Route, classify  # noqa: E402
from app.backend.services.financial_values import (  # noqa: E402
    asked_field, parse_financial_values, values_dict,
)
from evaluation.financial_metrics import abstained, numeric_match  # noqa: E402

PASSED, FAILED = [], []

SUMMARY = """Financial summary (in $000s):
  Revenue: 8,075.00
  Net income: 695.00
  Total assets: 14,000.00
  Total liabilities: 5,000.00
  Cash: 2,700.00
  Total debt: 3,735.00
  EPS: 4.25
  EBITDA: 1,655.00
  Free cash flow: 520.00
"""

STATEMENT = """Income statement (in $000s):
  Revenue 4,000.00
  Cost of goods sold 2,400.00
  Operating expenses 600.00
  Depreciation & amortization 160.00
  Interest expense 90.00
  Tax rate 25%
Balance sheet (in $000s):
  Total assets 9,000.00
  Total liabilities 4,000.00
"""


def check(name, ok, detail=""):
    (PASSED if ok else FAILED).append(name)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" - {detail}" if detail else ""))


def numeric(chat, question, expected, tolerance=0.02):
    answer = chat.ask(question).answer
    return numeric_match(answer, expected, tolerance), answer.replace("\n", " | ")[:100]


def test_interpretation_outranks_figures():
    """Part 6: what was requested outranks what the text contains.

    A rule of the form "two numbers -> calculator" sent "Debt/EBITDA rose from
    2x to 5x. What does that imply?" to the calculator, which answered
    "Debt/EBITDA = 5.00" from the 2 and the 5. Measured on the hidden split.
    """
    cases = [
        "Debt/EBITDA rose from 2x to 5x in a year. What does that imply?",
        "Days sales outstanding rose from 30 to 75. What happened?",
        "Operating margin improved while gross margin fell. How is that possible?",
        "Gross margin is 60% but operating margin is 2%. What does that tell you?",
        "Interest coverage fell from 8x to 1.5x. What is the implication?",
    ]
    chat = FinancialChat(model=None)
    for question in cases:
        route = classify(question).route
        check(f"routes INTERPRETATION despite figures: {question[:40]}",
              route == Route.INTERPRETATION, f"got {route.value}")
        answer = chat.ask(question).answer
        check(f"  and is not answered with arithmetic: {question[:34]}",
              "Formula / Breakdown" not in answer, answer[:70])

    # The converse must still hold: an explicit calculation stays a calculation.
    for question in ("Calculate ROE from net income of 60.00 and equity of 400.00.",
                     "Revenue is 1,000.00 and net income is 120.00. What is the net profit margin?"):
        route = classify(question).route
        check(f"still NUMERICAL: {question[:44]}", route == Route.NUMERICAL,
              f"got {route.value}")


def test_definition_is_not_an_explanation():
    """Part 11: an entry about the right subject can still be the wrong answer."""
    chat = FinancialChat(model=None)
    # Asks about a property the entry does not discuss -> must not be answered
    # with that entry.
    for question, forbidden in (
            ("Why do interest rate rises usually reduce equity valuations?",
             "price of borrowing"),
            ("What does a high inventory turnover suggest?",
             "how many times inventory is sold")):
        answer = chat.ask(question).answer
        check(f"withholds a bare definition: {question[:40]}",
              forbidden not in answer, answer[:70])

    # Asks about a property the entry DOES state -> the entry is a fair answer.
    answer = chat.ask("Why is depreciation called a non-cash expense?").answer
    check("an entry that states the asked property may answer",
          "non-cash" in answer.lower(), answer[:70])


def test_pattern_selection_by_evidence():
    """Parts 13/14: the best-supported pattern wins, not the first declared."""
    from app.backend.services.analysis_patterns import ANALYSIS_PATTERNS, match_pattern

    specific = match_pattern("Operating margin improved while gross margin fell. "
                             "How is that possible?")
    check("specific pattern beats the general one",
          specific is not None and "fell by more than" in specific["text"],
          (specific or {}).get("text", "")[:70])

    # One keyword must never trigger a diagnostic.
    for weak in ("What is profit?", "Tell me about margins", "inventory"):
        check(f"no pattern fires on a single keyword: {weak[:30]}",
              match_pattern(weak) is None, str(match_pattern(weak))[:60])

    # Every pattern carries at least two required groups, so evidence is needed.
    thin = [i for i, (groups, _) in enumerate(ANALYSIS_PATTERNS) if len(groups) < 2]
    check("every pattern requires >= 2 signal groups", not thin, str(thin))


def test_routing():
    """The intent must be decided before any tool runs."""
    cases = [
        (SUMMARY + "\nWhat is Total debt?", Route.EXTRACTION),
        (SUMMARY + "\nWhat is Cash?", Route.EXTRACTION),
        ("Calculate ROE from net income of 60.00 and equity of 400.00.", Route.NUMERICAL),
        ("Revenue is 1,000.00 and net income is 120.00. What is the net profit margin?",
         Route.NUMERICAL),
        ("What is EBITDA?", Route.FINANCIAL_KNOWLEDGE),
        ("What is working capital?", Route.FINANCIAL_KNOWLEDGE),
        ("What is the Fed rate today?", Route.CURRENT_DATA),
        ("What is Tesla's share price right now?", Route.CURRENT_DATA),
        ("What was Apple's revenue yesterday?", Route.CURRENT_DATA),
        ("How much did Microsoft earn this quarter?", Route.CURRENT_DATA),
    ]
    for question, expected in cases:
        got = classify(question).route
        check(f"routes {expected.value}: {question.splitlines()[-1][:44]}",
              got == expected, f"got {got.value}")

    # A stated figure is an input, not the subject: this must NOT be extraction.
    got = classify("Total debt is 3,000.00 and equity is 1,500.00. "
                   "What is the debt-to-equity ratio?").route
    check("a stated figure used as an input stays NUMERICAL",
          got == Route.NUMERICAL, f"got {got.value}")


def test_extraction():
    chat = FinancialChat(model=None)
    for question, expected in (("What is total debt?", 3735.00),
                               ("What is cash?", 2700.00),
                               ("What is net income?", 695.00),
                               ("What is total liabilities?", 5000.00),
                               ("What is EPS?", 4.25),
                               ("What is free cash flow?", 520.00),
                               ("What is total assets?", 14000.00)):
        ok, answer = numeric(chat, SUMMARY + "\n" + question, expected, 0.01)
        check(f"extracts {question[:34]} -> {expected:,.2f}", ok, answer)

    # The answer must cite where it read the figure, not just assert it.
    detail = chat.ask(SUMMARY + "\nWhat is Total debt?").detail
    check("extraction cites its source span",
          "3,735" in str(detail.get("source_span", "")), str(detail)[:90])


def test_parser_gaps():
    """The three phrasings that produced 68 declined-but-answerable questions."""
    cases = [
        ("Net income is 905.00 and dividends paid are 362.00.",
         {"net_income": 905.0, "dividends": 362.0}),
        ("Net income is 220.00 and there are 360.00 shares outstanding.",
         {"net_income": 220.0, "shares_outstanding": 360.0}),
        ("Equity is 70% of capital at a cost of 8%, debt is the rest at 5%, "
         "and the tax rate is 20%.",
         {"equity_weight": 0.7, "cost_of_equity": 0.08, "cost_of_debt": 0.05,
          "tax_rate": 0.2}),
        ("An investment grew from 400.00 to 644.20 over 3 years.",
         {"beginning_value": 400.0, "ending_value": 644.2, "years": 3.0}),
        ("Market capitalisation is 12,000.00, total debt is 3,000.00, and cash is 500.00.",
         {"market_cap": 12000.0, "total_debt": 3000.0, "cash": 500.0}),
        ("Current assets are 1,060.00, inventory is 200.00, and current liabilities "
         "are 400.00.",
         {"current_assets": 1060.0, "inventory": 200.0, "current_liabilities": 400.0}),
    ]
    for text, expected in cases:
        parsed = values_dict(parse_financial_values(text))
        missing = {k: v for k, v in expected.items()
                   if abs(parsed.get(k, float("nan")) - v) > 0.001}
        check(f"parses {text[:46]}", not missing,
              f"wrong/missing {missing}, got {parsed}")

    # Provenance is kept for every figure.
    values = parse_financial_values("Total debt is $3,735.00 million.")
    debt = next(v for v in values if v.field == "total_debt")
    check("keeps unit, currency, scale and source span",
          debt.value == 3735e6 and debt.currency == "USD" and debt.scale == "million"
          and "3,735" in debt.source_span, str(debt.to_dict()))


def test_calculations():
    chat = FinancialChat(model=None)
    cases = [
        ("Net income is 905.00 and dividends paid are 362.00. "
         "What is the dividend payout ratio?", 40.00),
        ("Net income is 220.00 and there are 360.00 shares outstanding. What is EPS?", 0.61),
        ("Equity is 70% of capital at a cost of 8%, debt is the rest at 5%, and the "
         "tax rate is 20%. What is the WACC?", 6.80),
        ("Market capitalisation is 12,000.00, total debt is 3,000.00, and cash is "
         "500.00. What is enterprise value?", 14500.00),
        ("An investment grew from 400.00 to 644.20 over 3 years. What is the CAGR?", 17.23),
        ("Current assets are 1,060.00, inventory is 200.00, and current liabilities "
         "are 400.00. What is the quick ratio?", 2.15),
        ("Revenue is 8,075.00 and total assets are 4,000.00. What is asset turnover?", 2.02),
        ("EBIT is 900.00 and interest expense is 150.00. What is the interest "
         "coverage ratio?", 6.00),
        # Must still work: these were already correct before the change.
        ("Current assets are 1,060.00 and current liabilities are 400.00. "
         "What is the current ratio?", 2.65),
        ("Revenue is 1,000.00 and net income is 120.00. What is the net profit margin?", 12.00),
        ("Total assets are 9,000.00, total liabilities are 4,000.00, and net income "
         "is 500.00. What is ROE?", 10.00),
    ]
    for question, expected in cases:
        ok, answer = numeric(chat, question, expected, 0.05)
        check(f"computes {question[:50]} -> {expected:,.2f}", ok, answer)

    # Statement derivations.
    for question, expected in (("What is shareholders' equity?", 5000.00),
                               ("What is EBIT?", 840.00),
                               ("What is net income?", 562.50)):
        ok, answer = numeric(chat, STATEMENT + "\n" + question, expected, 0.5)
        check(f"derives {question[:32]} -> {expected:,.2f}", ok, answer)


def test_concepts_still_answered():
    chat = FinancialChat(model=None)
    for question, must_contain in (("What is EBITDA?", "Earnings Before Interest"),
                                   ("What is working capital?", "current"),
                                   ("What is depreciation?", "tangible")):
        answer = chat.ask(question).answer
        check(f"concept answered: {question}", must_contain.lower() in answer.lower(),
              answer[:70])


def test_current_data_abstains():
    chat = FinancialChat(model=None)
    for question in ("What is the Fed rate today?",
                     "What is the current Federal Reserve interest rate today?",
                     "What is Tesla's share price right now?",
                     "What was Apple's revenue yesterday?",
                     "What will the S&P 500 close at tomorrow?"):
        answer = chat.ask(question).answer
        declined = abstained(answer)
        # The specific failure mode: a DEFINITION standing in for a current value.
        definitional = ("is the price of borrowing" in answer
                        or "expressed as a percentage" in answer
                        or "day-to-day costs" in answer)
        check(f"abstains on current data: {question[:44]}",
              declined and not definitional, answer[:90])


def test_missing_information_still_abstains():
    """Fixing over-abstention must not make the system guess."""
    chat = FinancialChat(model=None)
    for question in ("Revenue is 2,420.00. What is the net profit margin?",
                     "Total assets are 4,590.00. What is ROE?",
                     "Current assets are 1,060.00. What is the current ratio?",
                     "Net income is 3,830.00. What is EPS?",
                     "Operating cash flow is 3,340.00. What is free cash flow?",
                     "Calculate ROE."):
        answer = chat.ask(question).answer
        check(f"abstains when a figure is missing: {question[:46]}",
              abstained(answer), answer[:90])


def main():
    test_routing()
    test_interpretation_outranks_figures()
    test_definition_is_not_an_explanation()
    test_pattern_selection_by_evidence()
    test_extraction()
    test_parser_gaps()
    test_calculations()
    test_concepts_still_answered()
    test_current_data_abstains()
    test_missing_information_still_abstains()
    print(f"\n{len(PASSED)}/{len(PASSED) + len(FAILED)} passed")
    sys.exit(1 if FAILED else 0)


if __name__ == "__main__":
    main()
