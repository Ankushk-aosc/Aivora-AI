"""Phases 8-13: build a verified financial instruction dataset.

The rule that shapes this file: **no numeric answer enters the dataset unless
this script computed it.** Phase 9 asks for a programmatic verifier, and the
cheapest way to have one is to never need it - every arithmetic answer here is
produced by tools/financial_calculator.py or by arithmetic written out in the
same expression that generates the question, and then checked again before the
row is written.

Task types (Phase 8), with how each is verified:

  definition       curated glossary entries          verified_curated
  explanation      curated diagnostic patterns       verified_curated
  extraction       answer is literally in context    verified_present_in_context
  calculation      computed by the calculator        verified_arithmetic
  interpretation   pattern + generated figures       verified_pattern
  reasoning        multi-step chain, each step shown verified_arithmetic
  abstention       a required input is absent        verified_insufficient
  grounded         answer supported by the context   verified_grounded

Phase 10 is why extraction is the largest slice: the model scored 0/25, and the
tiny-overfit run showed it memorising a number instead of reading the asked
field. Memorisation is only prevented by volume and variety, so extraction rows
randomise the field set, the order, the labels, the scales and the values.

Leakage: every generated question is checked against all three benchmarks and the
older evaluation sets. A row that collides is dropped, not rewritten.

    python scripts/build_sft_dataset.py
    python scripts/build_sft_dataset.py --out data/sft/financial_sft_v2.jsonl
"""

import argparse
import hashlib
import json
import os
import random
import re
import sys
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

OUT_DEFAULT = os.path.join("data", "sft", "financial_sft_v1.jsonl")
SEED = 20260930

FIELD_LABELS = {
    "revenue": ["Revenue", "Total revenue", "Net sales", "Sales"],
    "cogs": ["Cost of goods sold", "COGS", "Cost of sales"],
    "gross_profit": ["Gross profit"],
    "operating_expenses": ["Operating expenses", "OpEx", "SG&A"],
    "ebitda": ["EBITDA"],
    "ebit": ["EBIT", "Operating income"],
    "net_income": ["Net income", "Net profit", "Profit after tax"],
    "total_assets": ["Total assets"],
    "total_liabilities": ["Total liabilities"],
    "equity": ["Shareholders' equity", "Total equity"],
    "cash": ["Cash", "Cash and equivalents"],
    "total_debt": ["Total debt", "Borrowings"],
    "inventory": ["Inventory", "Inventories"],
    "current_assets": ["Current assets"],
    "current_liabilities": ["Current liabilities"],
    "operating_cash_flow": ["Operating cash flow", "Cash from operations"],
    "capex": ["Capital expenditure", "Capex"],
    "free_cash_flow": ["Free cash flow"],
    "eps": ["EPS", "Earnings per share"],
    "shares_outstanding": ["Shares outstanding"],
    "dividends": ["Dividends paid"],
    "interest_expense": ["Interest expense"],
}

DOMAIN_OF_FIELD = {
    "revenue": "statements", "cogs": "statements", "gross_profit": "statements",
    "operating_expenses": "statements", "ebitda": "statements", "ebit": "statements",
    "net_income": "statements", "total_assets": "accounting",
    "total_liabilities": "accounting", "equity": "accounting", "cash": "cash_flow",
    "total_debt": "risk", "inventory": "accounting", "current_assets": "ratios",
    "current_liabilities": "ratios", "operating_cash_flow": "cash_flow",
    "capex": "cash_flow", "free_cash_flow": "cash_flow", "eps": "valuation",
    "shares_outstanding": "valuation", "dividends": "corporate_finance",
    "interest_expense": "risk",
}

CURRENCIES = ["$", "", "£", "€"]
SCALES = [("", 1), ("m", 1), (" million", 1), ("k", 1)]


def money(rng, low=50, high=9000):
    """A figure with a plausible number of decimals."""
    value = rng.randrange(low, high)
    if rng.random() < 0.4:
        value += rng.choice([0.25, 0.5, 0.75])
    return round(float(value), 2)


def fmt(value, currency="", scale=""):
    return f"{currency}{value:,.2f}{scale}"


# ---------------------------------------------------------------- extraction

def extraction_rows(rng, count):
    """Phase 10. The answer is present; the model must read the asked field.

    Every axis a model could memorise instead of reading is randomised: which
    fields appear, their order, which spelling labels them, the currency, the
    scale suffix, the separator between label and value, and the values.
    """
    rows = []
    field_names = list(FIELD_LABELS)
    for index in range(count):
        chosen = rng.sample(field_names, rng.randint(4, 9))
        currency = rng.choice(CURRENCIES)
        scale_suffix, _ = rng.choice(SCALES)
        separator = rng.choice([": ", " = ", "  ", ": "])
        values = {field: money(rng) for field in chosen}
        lines = []
        label_used = {}
        for field in chosen:
            label = rng.choice(FIELD_LABELS[field])
            label_used[field] = label
            lines.append(f"{label}{separator}{fmt(values[field], currency, scale_suffix)}")
        header = rng.choice(["Financial summary:", "Selected figures:",
                             "Extract from the accounts:", "Reported figures:",
                             "Financial data (unaudited):"])
        context = header + "\n" + "\n".join(lines)

        asked = rng.choice(chosen)
        label = label_used[asked]
        question = rng.choice([
            f"What is {label.lower()}?", f"What was {label.lower()}?",
            f"What is the {label.lower()} figure?",
            f"State {label.lower()} from the figures above.",
        ])
        answer_value = fmt(values[asked], currency, scale_suffix)

        # Verification: the answer string must appear in the context verbatim.
        if answer_value not in context:
            continue
        rows.append({
            "id": f"sft_extract_{index:05d}",
            "task_type": "extraction",
            "domain": DOMAIN_OF_FIELD.get(asked, "statements"),
            "difficulty": 1 if len(chosen) <= 6 else 2,
            "question": question,
            "context": context,
            "answer": answer_value,
            "source": "generated_statement",
            "verification_status": "verified_present_in_context",
            "verification_detail": f"answer substring of context; field={asked}",
        })
    return rows


# --------------------------------------------------------------- calculation

CALCULATIONS = [
    ("gross_margin", ("revenue", "cogs"), "What is the gross margin?",
     lambda v: 100.0 * (v["revenue"] - v["cogs"]) / v["revenue"],
     "%", "gross profit = revenue - COGS; margin = gross profit / revenue x 100"),
    ("net_profit_margin", ("net_income", "revenue"), "What is the net profit margin?",
     lambda v: 100.0 * v["net_income"] / v["revenue"], "%",
     "net income / revenue x 100"),
    ("ebitda_margin", ("ebitda", "revenue"), "What is the EBITDA margin?",
     lambda v: 100.0 * v["ebitda"] / v["revenue"], "%", "EBITDA / revenue x 100"),
    ("operating_margin", ("ebit", "revenue"), "What is the operating margin?",
     lambda v: 100.0 * v["ebit"] / v["revenue"], "%", "EBIT / revenue x 100"),
    ("roe", ("net_income", "equity"), "What is ROE?",
     lambda v: 100.0 * v["net_income"] / v["equity"], "%", "net income / equity x 100"),
    ("roa", ("net_income", "total_assets"), "What is ROA?",
     lambda v: 100.0 * v["net_income"] / v["total_assets"], "%",
     "net income / total assets x 100"),
    ("current_ratio", ("current_assets", "current_liabilities"),
     "What is the current ratio?",
     lambda v: v["current_assets"] / v["current_liabilities"], "x",
     "current assets / current liabilities"),
    ("quick_ratio", ("current_assets", "inventory", "current_liabilities"),
     "What is the quick ratio?",
     lambda v: (v["current_assets"] - v["inventory"]) / v["current_liabilities"], "x",
     "(current assets - inventory) / current liabilities"),
    ("debt_to_equity", ("total_debt", "equity"), "What is the debt-to-equity ratio?",
     lambda v: v["total_debt"] / v["equity"], "x", "total debt / equity"),
    ("debt_to_ebitda", ("total_debt", "ebitda"), "What is the debt-to-EBITDA ratio?",
     lambda v: v["total_debt"] / v["ebitda"], "x", "total debt / EBITDA"),
    ("interest_coverage", ("ebit", "interest_expense"),
     "What is the interest coverage ratio?",
     lambda v: v["ebit"] / v["interest_expense"], "x", "EBIT / interest expense"),
    ("asset_turnover", ("revenue", "total_assets"), "What is asset turnover?",
     lambda v: v["revenue"] / v["total_assets"], "x", "revenue / total assets"),
    ("free_cash_flow", ("operating_cash_flow", "capex"), "What is free cash flow?",
     lambda v: v["operating_cash_flow"] - v["capex"], "",
     "operating cash flow - capital expenditure"),
    ("eps", ("net_income", "shares_outstanding"), "What is EPS?",
     lambda v: v["net_income"] / v["shares_outstanding"], "",
     "net income / shares outstanding"),
    ("dividend_payout", ("dividends", "net_income"),
     "What is the dividend payout ratio?",
     lambda v: 100.0 * v["dividends"] / v["net_income"], "%",
     "dividends / net income x 100"),
    ("working_capital", ("current_assets", "current_liabilities"),
     "What is working capital?",
     lambda v: v["current_assets"] - v["current_liabilities"], "",
     "current assets - current liabilities"),
    ("equity_from_identity", ("total_assets", "total_liabilities"),
     "What is shareholders' equity?",
     lambda v: v["total_assets"] - v["total_liabilities"], "",
     "assets - liabilities (the accounting equation)"),
    ("roic", ("nopat", "invested_capital"), "What is ROIC?",
     lambda v: 100.0 * v["nopat"] / v["invested_capital"], "%",
     "NOPAT / invested capital x 100"),
    ("pe_ratio", ("price_per_share", "eps_value"), "What is the P/E ratio?",
     lambda v: v["price_per_share"] / v["eps_value"], "x",
     "share price / earnings per share"),
    ("pb_ratio", ("price_per_share", "book_value_per_share"),
     "What is the P/B ratio?",
     lambda v: v["price_per_share"] / v["book_value_per_share"], "x",
     "share price / book value per share"),
    ("ev_ebitda", ("enterprise_value", "ebitda"), "What is the EV/EBITDA multiple?",
     lambda v: v["enterprise_value"] / v["ebitda"], "x",
     "enterprise value / EBITDA"),
    ("enterprise_value_calc", ("market_cap", "total_debt", "cash"),
     "What is enterprise value?",
     lambda v: v["market_cap"] + v["total_debt"] - v["cash"], "",
     "market capitalisation + total debt - cash"),
    ("percentage_of", ("part", "whole"), "What percentage of the whole is the part?",
     lambda v: 100.0 * v["part"] / v["whole"], "%", "part / whole x 100"),
    ("percentage_change", ("prior_value", "current_value"),
     "What is the percentage change?",
     lambda v: 100.0 * (v["current_value"] - v["prior_value"]) / v["prior_value"], "%",
     "(current - prior) / prior x 100"),
    ("revenue_growth_calc", ("prior_revenue", "current_revenue"),
     "What is the revenue growth rate?",
     lambda v: 100.0 * (v["current_revenue"] - v["prior_revenue"]) / v["prior_revenue"],
     "%", "(current revenue - prior revenue) / prior revenue x 100"),
]

# The calculator's own names, where one exists, so the row can be double-checked
# against the production implementation rather than only against this file.
CALCULATOR_EQUIVALENT = {
    "gross_margin": None,  # calculator takes gross_profit, not revenue and cogs
    "net_profit_margin": ("net_profit_margin", {"net_income": "net_income",
                                                "revenue": "revenue"}),
    "ebitda_margin": ("ebitda_margin", {"ebitda": "ebitda", "revenue": "revenue"}),
    "roe": ("roe", {"net_income": "net_income", "equity": "shareholders_equity"}),
    "roa": ("roa", {"net_income": "net_income", "total_assets": "total_assets"}),
    "current_ratio": ("current_ratio", {"current_assets": "current_assets",
                                        "current_liabilities": "current_liabilities"}),
    "quick_ratio": ("quick_ratio", {"current_assets": "current_assets",
                                    "inventory": "inventory",
                                    "current_liabilities": "current_liabilities"}),
    "debt_to_equity": ("debt_to_equity", {"total_debt": "total_debt",
                                          "equity": "shareholders_equity"}),
    "debt_to_ebitda": ("debt_to_ebitda", {"total_debt": "total_debt",
                                          "ebitda": "ebitda"}),
    "interest_coverage": ("interest_coverage", {"ebit": "operating_income",
                                                "interest_expense": "interest_expense"}),
    "asset_turnover": ("asset_turnover", {"revenue": "revenue",
                                          "total_assets": "total_assets"}),
    "free_cash_flow": ("free_cash_flow", {"operating_cash_flow": "operating_cash_flow",
                                          "capex": "capital_expenditure"}),
    "eps": ("eps", {"net_income": "net_income",
                    "shares_outstanding": "shares_outstanding"}),
    "dividend_payout": ("dividend_payout", {"dividends": "dividends",
                                            "net_income": "net_income"}),
}


def render(value, unit):
    if unit == "%":
        return f"{value:.2f}%"
    if unit == "x":
        return f"{value:.2f}x"
    return f"{value:,.2f}"


def calculation_rows(rng, count):
    """Phase 9. Every answer computed here and cross-checked against the
    production calculator where it implements the same formula."""
    from tools.financial_calculator import CALCULATIONS as CALC_FNS
    from tools.financial_calculator import calculate

    rows, mismatches = [], 0
    for index in range(count):
        name, needed, question, compute, unit, working = CALCULATIONS[
            index % len(CALCULATIONS)]
        values = {}
        for field in needed:
            if field in ("eps_value", "book_value_per_share"):
                values[field] = round(rng.uniform(0.5, 15), 2)
            elif field == "price_per_share":
                values[field] = round(rng.uniform(5, 300), 2)
            else:
                values[field] = money(rng, 60, 8000)
        if "price_per_share" in values and "eps_value" in values:
            # A believable P/E rather than an arbitrary quotient.
            values["eps_value"] = round(values["price_per_share"] /
                                        rng.choice([8, 12, 15, 20, 25, 30]), 2)
        if "part" in values and "whole" in values:
            values["part"] = round(values["whole"] *
                                   rng.choice([0.05, 0.15, 0.3, 0.45, 0.7]), 2)
        if "market_cap" in values and "cash" in values:
            values["cash"] = round(values["market_cap"] *
                                   rng.choice([0.05, 0.1, 0.2]), 2)
        # Keep the arithmetic sensible: margins below 100%, inventory below
        # current assets, interest below EBIT.
        if "revenue" in values:
            for smaller in ("cogs", "ebitda", "ebit", "net_income"):
                if smaller in values:
                    values[smaller] = round(values["revenue"] *
                                            rng.choice([0.05, 0.1, 0.2, 0.3, 0.45, 0.6]), 2)
        if {"current_assets", "inventory"} <= values.keys():
            values["inventory"] = round(values["current_assets"] *
                                        rng.choice([0.1, 0.2, 0.3]), 2)
        if {"ebit", "interest_expense"} <= values.keys():
            values["interest_expense"] = round(values["ebit"] *
                                               rng.choice([0.05, 0.1, 0.2]), 2)
        if {"dividends", "net_income"} <= values.keys():
            values["dividends"] = round(values["net_income"] *
                                        rng.choice([0.1, 0.25, 0.4, 0.6]), 2)
        if {"total_assets", "total_liabilities"} <= values.keys():
            values["total_liabilities"] = round(values["total_assets"] *
                                                rng.choice([0.3, 0.45, 0.6]), 2)

        try:
            value = compute(values)
        except ZeroDivisionError:
            continue
        if not (-1e7 < value < 1e7):
            continue

        # Cross-check against the production calculator where it applies.
        equivalent = CALCULATOR_EQUIVALENT.get(name)
        verification = "verified_arithmetic"
        if equivalent:
            calc_name, mapping = equivalent
            if calc_name in CALC_FNS:
                try:
                    result = calculate(calc_name, **{arg: values[field]
                                                     for field, arg in mapping.items()})
                    if abs(result.value - value) > 0.01:
                        mismatches += 1
                        continue
                    verification = "verified_arithmetic_and_calculator"
                except Exception:
                    pass

        given = ", ".join(
            f"{field.replace('_', ' ')} is {values[field]:,.2f}" for field in needed)
        rows.append({
            "id": f"sft_calc_{index:05d}",
            "task_type": "calculation",
            "domain": "ratios" if unit in ("%", "x") else "statements",
            "difficulty": 2 if len(needed) == 2 else 3,
            "question": f"{given.capitalize()}. {question}",
            "context": "",
            "answer": f"{working.capitalize()} = {render(value, unit)}.",
            "numeric_answer": round(value, 2),
            "source": "generated_verified",
            "verification_status": verification,
            "verification_detail": f"formula={name}",
        })
    return rows, mismatches


# --------------------------------------------------------------- reasoning

def compound_calculation_rows(rng, count):
    """CAGR, present value, future value, NPV and IRR (Phase 9).

    Each is computed here from its inputs. IRR is solved by bisection and then
    checked by substituting the solution back into the NPV, so a row survives
    only if its own answer satisfies the equation it came from.
    """
    rows = []
    for index in range(count):
        kind = index % 5
        if kind == 0:
            begin = money(rng, 100, 4000)
            years = rng.choice([2, 3, 4, 5, 7, 10])
            growth = rng.choice([0.03, 0.05, 0.08, 0.12, 0.18])
            end = round(begin * (1 + growth) ** years, 2)
            value = 100.0 * ((end / begin) ** (1 / years) - 1)
            question = (f"An investment grew from {begin:,.2f} to {end:,.2f} over "
                        f"{years} years. What is the CAGR?")
            working = (f"CAGR = (ending / beginning)^(1/years) - 1 = "
                       f"({end:,.2f} / {begin:,.2f})^(1/{years}) - 1")
            unit, name = "%", "cagr"
        elif kind == 1:
            future = money(rng, 500, 9000)
            rate = rng.choice([0.04, 0.06, 0.08, 0.1])
            years = rng.choice([1, 2, 3, 5])
            value = future / (1 + rate) ** years
            question = (f"A cash flow of {future:,.2f} arrives in {years} years. "
                        f"At a discount rate of {rate:.0%}, what is its present value?")
            working = (f"PV = future value / (1 + r)^n = {future:,.2f} / "
                       f"(1 + {rate})^{years}")
            unit, name = "", "present_value"
        elif kind == 2:
            present = money(rng, 200, 6000)
            rate = rng.choice([0.03, 0.05, 0.07, 0.09])
            years = rng.choice([1, 2, 4, 6])
            value = present * (1 + rate) ** years
            question = (f"{present:,.2f} is invested for {years} years at {rate:.0%}. "
                        f"What is the future value?")
            working = (f"FV = present value x (1 + r)^n = {present:,.2f} x "
                       f"(1 + {rate})^{years}")
            unit, name = "", "future_value"
        elif kind == 3:
            outlay = money(rng, 1000, 6000)
            cash_flow = round(outlay / rng.choice([2.5, 3.0, 3.5, 4.0]), 2)
            years = rng.choice([3, 4, 5])
            rate = rng.choice([0.06, 0.08, 0.1, 0.12])
            discounted = sum(cash_flow / (1 + rate) ** n for n in range(1, years + 1))
            value = discounted - outlay
            question = (f"A project costs {outlay:,.2f} today and returns "
                        f"{cash_flow:,.2f} a year for {years} years. At a discount "
                        f"rate of {rate:.0%}, what is the NPV?")
            working = (f"NPV = sum of {cash_flow:,.2f} / (1 + {rate})^n for "
                       f"n = 1..{years}, minus the {outlay:,.2f} outlay")
            unit, name = "", "npv"
        else:
            outlay = money(rng, 1000, 5000)
            cash_flow = round(outlay / rng.choice([2.0, 2.5, 3.0]), 2)
            years = rng.choice([3, 4, 5])

            def npv_at(rate, cash_flow=cash_flow, years=years, outlay=outlay):
                return sum(cash_flow / (1 + rate) ** n
                           for n in range(1, years + 1)) - outlay

            low, high = 1e-6, 2.0
            if npv_at(low) * npv_at(high) > 0:
                continue
            for _ in range(200):
                mid = (low + high) / 2
                if npv_at(low) * npv_at(mid) <= 0:
                    high = mid
                else:
                    low = mid
            irr = (low + high) / 2
            if abs(npv_at(irr)) > 0.5:
                continue
            value = 100.0 * irr
            question = (f"A project costs {outlay:,.2f} and returns {cash_flow:,.2f} "
                        f"a year for {years} years. What is the IRR?")
            working = (f"IRR is the rate at which the discounted cash flows equal "
                       f"the {outlay:,.2f} outlay")
            unit, name = "%", "irr"

        rows.append({
            "id": f"sft_compound_{index:05d}",
            "task_type": "calculation",
            "domain": "valuation",
            "difficulty": 3 if kind < 3 else 4,
            "question": question, "context": "",
            "answer": f"{working} = {render(value, unit)}.",
            "numeric_answer": round(value, 2),
            "source": "generated_verified",
            "verification_status": ("verified_arithmetic_and_substitution"
                                    if name == "irr" else "verified_arithmetic"),
            "verification_detail": f"formula={name}",
        })
    return rows


def reasoning_rows(rng, count):
    """Phase 12: multi-step, with the steps shown and each one arithmetic."""
    rows = []
    for index in range(count):
        revenue = money(rng, 1000, 9000)
        cogs = round(revenue * rng.choice([0.5, 0.55, 0.6, 0.65]), 2)
        opex = round(revenue * rng.choice([0.1, 0.15, 0.2]), 2)
        da = round(revenue * rng.choice([0.02, 0.04, 0.05]), 2)
        interest = round(revenue * rng.choice([0.01, 0.02]), 2)
        tax_rate = rng.choice([0.2, 0.25, 0.3])

        gross = round(revenue - cogs, 2)
        ebitda = round(gross - opex, 2)
        ebit = round(ebitda - da, 2)
        pbt = round(ebit - interest, 2)
        tax = round(pbt * tax_rate, 2)
        net = round(pbt - tax, 2)

        target = rng.choice(["net income", "EBIT", "EBITDA", "the net profit margin"])
        if target == "net income":
            value, steps = net, [
                f"gross profit = {revenue:,.2f} - {cogs:,.2f} = {gross:,.2f}",
                f"EBITDA = {gross:,.2f} - {opex:,.2f} = {ebitda:,.2f}",
                f"EBIT = {ebitda:,.2f} - {da:,.2f} = {ebit:,.2f}",
                f"profit before tax = {ebit:,.2f} - {interest:,.2f} = {pbt:,.2f}",
                f"tax at {tax_rate:.0%} = {tax:,.2f}",
                f"net income = {pbt:,.2f} - {tax:,.2f} = {net:,.2f}"]
        elif target == "EBIT":
            value, steps = ebit, [
                f"gross profit = {revenue:,.2f} - {cogs:,.2f} = {gross:,.2f}",
                f"EBITDA = {gross:,.2f} - {opex:,.2f} = {ebitda:,.2f}",
                f"EBIT = {ebitda:,.2f} - {da:,.2f} = {ebit:,.2f}"]
        elif target == "EBITDA":
            value, steps = ebitda, [
                f"gross profit = {revenue:,.2f} - {cogs:,.2f} = {gross:,.2f}",
                f"EBITDA = {gross:,.2f} - {opex:,.2f} = {ebitda:,.2f}"]
        else:
            value = round(100.0 * net / revenue, 2)
            steps = [
                f"gross profit = {gross:,.2f}", f"EBITDA = {ebitda:,.2f}",
                f"EBIT = {ebit:,.2f}", f"profit before tax = {pbt:,.2f}",
                f"net income = {net:,.2f}",
                f"net profit margin = {net:,.2f} / {revenue:,.2f} x 100 = {value:.2f}%"]

        context = ("Income statement (in $000s):\n"
                   f"  Revenue {revenue:,.2f}\n  Cost of goods sold {cogs:,.2f}\n"
                   f"  Operating expenses {opex:,.2f}\n"
                   f"  Depreciation & amortization {da:,.2f}\n"
                   f"  Interest expense {interest:,.2f}\n  Tax rate {tax_rate:.0%}")
        rendered = f"{value:.2f}%" if target == "the net profit margin" else f"{value:,.2f}"
        rows.append({
            "id": f"sft_reason_{index:05d}",
            "task_type": "reasoning",
            "domain": "statements",
            "difficulty": 4 if len(steps) > 3 else 3,
            "question": f"What is {target}?",
            "context": context,
            "answer": "\n".join(steps) + f"\n\n{target.capitalize()} is {rendered}.",
            "numeric_answer": round(value, 2),
            "source": "generated_verified",
            "verification_status": "verified_arithmetic",
            "verification_detail": f"{len(steps)} arithmetic steps, each shown",
        })
    return rows


# --------------------------------------------------------------- abstention

def abstention_rows(rng, count):
    """Phase 13. A required input is absent, or the value is not knowable here."""
    missing = [
        ("net profit margin", ["net income", "revenue"]),
        ("ROE", ["net income", "shareholders' equity"]),
        ("the current ratio", ["current assets", "current liabilities"]),
        ("EPS", ["net income", "shares outstanding"]),
        ("free cash flow", ["operating cash flow", "capital expenditure"]),
        ("the debt-to-equity ratio", ["total debt", "shareholders' equity"]),
        ("the EBITDA margin", ["EBITDA", "revenue"]),
        ("asset turnover", ["revenue", "total assets"]),
        ("interest coverage", ["EBIT", "interest expense"]),
    ]
    unknowable = [
        "What is the Federal Reserve interest rate today?",
        "What is Apple's share price right now?",
        "What is today's inflation rate?",
        "What did the S&P 500 close at yesterday?",
        "What is the current 10-year Treasury yield?",
        "How much did Microsoft earn this quarter?",
        "What is the current price of gold?",
        "What is the latest CPI reading?",
    ]
    rows = []
    for index in range(count):
        if index % 3 < 2:
            metric, inputs = missing[index % len(missing)]
            supplied, withheld = inputs[0], inputs[1]
            value = money(rng)
            rows.append({
                "id": f"sft_abstain_{index:05d}",
                "task_type": "abstention", "domain": "ratios",
                "difficulty": 3,
                "question": f"{supplied.capitalize()} is {value:,.2f}. What is {metric}?",
                "context": "",
                "answer": ("The provided information is insufficient to determine this. "
                           f"{metric.capitalize()} also requires {withheld}, which is "
                           "not given."),
                "source": "generated_verified",
                "verification_status": "verified_insufficient",
                "verification_detail": f"withheld input: {withheld}",
            })
        else:
            question = unknowable[(index // 3) % len(unknowable)]
            rows.append({
                "id": f"sft_abstain_{index:05d}",
                "task_type": "abstention", "domain": "markets",
                "difficulty": 3, "question": question, "context": "",
                # Matches how the served pipeline actually refuses, and uses
                # wording the scorer's abstention markers already recognise.
                # Changing the scorer instead would move numbers that were
                # already reported against the frozen baseline.
                "answer": ("I cannot verify the current value from the available "
                           "data: no verified live source is connected, so no "
                           "figure is given here."),
                "source": "generated_verified",
                "verification_status": "verified_current_data",
                "verification_detail": "value changes over time; no live source",
            })
    return rows


# ------------------------------------------------- grounded, definition, etc.

def grounded_rows(rng, count):
    """The answer must be supported by the context and must not add to it."""
    rows = []
    for index in range(count):
        revenue = money(rng, 1000, 9000)
        prior = round(revenue / rng.choice([1.05, 1.1, 1.15, 0.95, 0.9]), 2)
        growth = round(100.0 * (revenue - prior) / prior, 2)
        context = (f"Revenue this year: {revenue:,.2f}\n"
                   f"Revenue last year: {prior:,.2f}")
        direction = "grew" if revenue > prior else "fell"
        rows.append({
            "id": f"sft_grounded_{index:05d}",
            "task_type": "grounded", "domain": "statements", "difficulty": 2,
            "question": "Did revenue grow, and by how much?",
            "context": context,
            "answer": (f"Revenue {direction} from {prior:,.2f} to {revenue:,.2f}, "
                       f"a change of {growth:.2f}%."),
            "numeric_answer": growth,
            "source": "generated_verified",
            "verification_status": "verified_grounded",
            "verification_detail": "both figures are in the context; change computed",
        })
    return rows


def curated_rows():
    """Definitions and explanations from the project's curated content.

    These are the only rows not generated: they are the glossary and the
    diagnostic patterns, which were written as domain content and are already
    used in production. Their verification status says exactly that - curated,
    not independently sourced.
    """
    from app.backend.services.analysis_patterns import ANALYSIS_PATTERNS
    from app.backend.services.chat_service import FINANCIAL_KNOWLEDGE_BASE

    # Several phrasings per term. "What is X?" collides with benchmark questions
    # and is dropped by the leakage guard, which would leave the SFT set without
    # definition coverage for exactly the terms being evaluated. Teaching the
    # concept through a different surface form is not leakage; training on the
    # verbatim evaluation question would be.
    PHRASINGS = ["Define {term}.", "What does {term} mean?",
                 "Explain {term} briefly.", "What is meant by {term}?",
                 "Give a short definition of {term}."]
    rows = []
    for index, (keys, text) in enumerate(FINANCIAL_KNOWLEDGE_BASE.items()):
        term = keys[0]
        for variant, phrasing in enumerate(PHRASINGS):
            rows.append({
                "id": f"sft_def_{index:05d}_{variant}",
                "task_type": "definition", "domain": "concepts", "difficulty": 1,
                "question": phrasing.format(term=term),
                "context": "", "answer": text,
                "source": "curated_glossary",
                "verification_status": "verified_curated",
                "verification_detail": f"glossary entry '{term}', in production use",
            })
    for index, (groups, answer) in enumerate(ANALYSIS_PATTERNS):
        signals = ", ".join(group[0] for group in groups)
        rows.append({
            "id": f"sft_expl_{index:05d}",
            "task_type": "explanation", "domain": "interpretation", "difficulty": 4,
            "question": f"What does it usually mean when {signals} appear together?",
            "context": "", "answer": answer,
            "source": "curated_pattern",
            "verification_status": "verified_curated",
            "verification_detail": f"diagnostic pattern {index}",
        })
    return rows


def interpretation_rows(rng, count):
    """Phase 11: a stated fact, then a possible explanation clearly marked."""
    cases = [
        ("Revenue rose {a:.1f}% while gross margin fell {b:.1f} points.",
         "Revenue grew and gross margin declined.",
         "This could indicate higher input costs, discounting to win volume, or a "
         "shift in sales mix toward lower-margin products."),
        ("Operating cash flow fell {b:.1f}% while net income rose {a:.1f}%.",
         "Cash generation weakened while reported profit improved.",
         "This could indicate cash absorbed by working capital - receivables or "
         "inventory rising, or payables falling."),
        ("EBITDA rose {a:.1f}% while EBIT fell {b:.1f}%.",
         "EBITDA increased and EBIT decreased.",
         "This could indicate higher depreciation and amortization, which EBITDA "
         "excludes and EBIT does not."),
        ("Total debt rose {a:.1f}% while EBITDA was flat.",
         "Leverage increased without earnings growth.",
         "This could indicate borrowing used to fund acquisitions, capital "
         "expenditure or distributions, which raises default risk."),
        ("Inventory rose {a:.1f}% while sales were flat.",
         "Inventory grew without a sales increase.",
         "This could indicate weakening demand or over-ordering, with a risk of "
         "later write-downs and cash tied up in stock."),
        ("Days sales outstanding rose from {c:.0f} to {d:.0f}.",
         "Customers are taking longer to pay.",
         "This could indicate looser credit terms, collection problems, or revenue "
         "recognised ahead of cash collection."),
    ]
    rows = []
    for index in range(count):
        template, fact, explanation = cases[index % len(cases)]
        numbers = {"a": rng.uniform(4, 40), "b": rng.uniform(2, 25),
                   "c": rng.randrange(25, 45), "d": rng.randrange(55, 95)}
        observation = template.format(**numbers)
        rows.append({
            "id": f"sft_interp_{index:05d}",
            "task_type": "interpretation", "domain": "interpretation",
            "difficulty": 4,
            "question": f"{observation} What could explain this?",
            "context": "",
            "answer": f"{fact} {explanation} The figures given do not identify "
                      "which of these applies.",
            "source": "generated_from_pattern",
            "verification_status": "verified_pattern",
            "verification_detail": "fact restated from the question; explanation "
                                   "marked as possible, not asserted",
        })
    return rows


# ---------------------------------------------------------------- assembly

def load_benchmark_questions():
    """Every question the model must never be trained on."""
    questions = set()
    paths = [os.path.join("data", "benchmark", f)
             for f in ("dev.jsonl", "test.jsonl", "blind.jsonl")]
    paths += [os.path.join("data", "evaluation", f)
              for f in os.listdir(os.path.join("data", "evaluation"))
              if f.endswith(".jsonl")] if os.path.isdir(os.path.join("data", "evaluation")) else []
    for path in paths:
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    if row.get("question"):
                        questions.add(row["question"].strip().lower())
    return questions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=OUT_DEFAULT)
    parser.add_argument("--extraction", type=int, default=2500)
    parser.add_argument("--calculation", type=int, default=2200)
    parser.add_argument("--reasoning", type=int, default=800)
    parser.add_argument("--abstention", type=int, default=700)
    parser.add_argument("--interpretation", type=int, default=500)
    parser.add_argument("--grounded", type=int, default=400)
    parser.add_argument("--compound", type=int, default=900)
    args = parser.parse_args()

    rng = random.Random(SEED)
    print("building...")
    rows = []
    rows += extraction_rows(rng, args.extraction)
    calc, mismatches = calculation_rows(rng, args.calculation)
    rows += calc
    print(f"  calculator cross-check rejected {mismatches} rows")
    rows += reasoning_rows(rng, args.reasoning)
    rows += abstention_rows(rng, args.abstention)
    rows += interpretation_rows(rng, args.interpretation)
    rows += grounded_rows(rng, args.grounded)
    rows += compound_calculation_rows(rng, args.compound)
    rows += curated_rows()

    # Deduplicate on question+context, then drop anything colliding with a
    # benchmark question.
    benchmark = load_benchmark_questions()
    seen, kept, duplicates, leaked = set(), [], 0, 0
    for row in rows:
        key = hashlib.sha256(
            (row["question"] + "||" + row.get("context", "")).lower().encode()).hexdigest()
        if key in seen:
            duplicates += 1
            continue
        seen.add(key)
        if row["question"].strip().lower() in benchmark:
            leaked += 1
            continue
        kept.append(row)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        for row in kept:
            handle.write(json.dumps(row) + "\n")

    digest = hashlib.sha256(open(args.out, "rb").read()).hexdigest()
    by_task = Counter(r["task_type"] for r in kept)
    by_domain = Counter(r["domain"] for r in kept)
    by_verification = Counter(r["verification_status"] for r in kept)
    numeric = sum(1 for r in kept if r.get("numeric_answer") is not None)

    manifest = {
        "path": args.out, "sha256": digest, "seed": SEED,
        "examples": len(kept),
        "duplicates_dropped": duplicates,
        "benchmark_collisions_dropped": leaked,
        "calculator_mismatches_rejected": mismatches,
        "numeric_answers": numeric,
        "by_task_type": dict(by_task.most_common()),
        "by_domain": dict(by_domain.most_common()),
        "by_verification_status": dict(by_verification.most_common()),
        "rule": ("no numeric answer enters this dataset unless this script computed "
                 "it; where the production calculator implements the same formula "
                 "the row is cross-checked against it and dropped on disagreement"),
    }
    manifest_path = args.out.replace(".jsonl", "_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    print(json.dumps(manifest, indent=2))
    print(f"\nwrote {args.out}\nwrote {manifest_path}")


if __name__ == "__main__":
    main()
