"""Multi-step financial calculations: values a question implies rather than states.

tools/financial_calculator.py computes one formula from inputs that are handed
to it. Real questions often supply the *components* instead:

    "Total assets are 800, total liabilities are 500, net income is 60.
     What is ROE?"        -> equity = assets - liabilities, then ROE

    "Net income 60, shares 20, share price 45. What is the P/E ratio?"
                          -> EPS = net income / shares, then P/E

    "Revenue rose from 200 to 250 while net income rose from 20 to 30.
     Did the net profit margin improve?"   -> use the LATER pair

Each rule here is a standard finance identity, stdlib-only, and returns the
working so an answer can show how it got there rather than asserting a number.
"""

import re
from dataclasses import dataclass, field


@dataclass
class Derived:
    name: str
    value: float
    unit: str
    formula: str
    inputs: dict = field(default_factory=dict)
    steps: list = field(default_factory=list)

    def formatted(self):
        if self.unit == "%":
            return f"{self.value:.2f}%"
        if self.unit == "x":
            return f"{self.value:.2f}x"
        return f"{self.value:,.2f}"


def _pairs_over_time(question):
    """{'revenue': (200.0, 250.0)} for 'revenue rose from 200 to 250'.

    Without this, a trend question is answered from the OPENING figures,
    which silently reports last period's margin as if it were this one's."""
    out = {}
    pattern = re.compile(
        r"(revenue|sales|net income|profit|earnings)\s*"
        r"(?:rose|grew|increased|went|fell|dropped|decreased|declined|changed)?\s*"
        r"from\s*[₹$€£]?\s*([\d,]+(?:\.\d+)?)\s*(?:to|->)\s*[₹$€£]?\s*([\d,]+(?:\.\d+)?)",
        re.IGNORECASE)
    for label, first, second in pattern.findall(question):
        key = label.lower().replace(" ", "_")
        key = {"sales": "revenue", "profit": "net_income",
               "earnings": "net_income"}.get(key, key)
        out[key] = (float(first.replace(",", "")), float(second.replace(",", "")))
    return out


def solve(question, values):
    """Return a Derived result if the question implies one, else None.

    `values` is financial_router.extract_financial_values(question)."""
    q = (question or "").lower()
    trend = _pairs_over_time(question)

    # Trend questions: answer for the CURRENT period, not the opening one.
    if trend and ("margin" in q or "improve" in q):
        revenue = trend.get("revenue", (None, values.get("revenue")))[-1]
        income = trend.get("net_income", (None, values.get("net_income")))[-1]
        if revenue and income:
            before = None
            if "revenue" in trend and "net_income" in trend:
                before = 100.0 * trend["net_income"][0] / trend["revenue"][0]
            margin = 100.0 * income / revenue
            steps = [f"current net profit margin = {income:,.2f} / {revenue:,.2f} x 100 = {margin:.2f}%"]
            if before is not None:
                steps.insert(0, f"previous net profit margin = {before:.2f}%")
                steps.append("improved" if margin > before else "did not improve")
            return Derived("Net Profit Margin (current period)", margin, "%",
                           "Net Income / Revenue x 100",
                           {"net_income": income, "revenue": revenue}, steps)

    # Operating margin from its components.
    if "operating margin" in q and {"revenue", "cogs", "operating_expenses"} <= values.keys():
        revenue, cogs, opex = values["revenue"], values["cogs"], values["operating_expenses"]
        operating_income = revenue - cogs - opex
        return Derived("Operating Margin", 100.0 * operating_income / revenue, "%",
                       "(Revenue - COGS - Operating Expenses) / Revenue x 100",
                       {"revenue": revenue, "cogs": cogs, "operating_expenses": opex},
                       [f"operating income = {revenue:,.2f} - {cogs:,.2f} - {opex:,.2f} "
                        f"= {operating_income:,.2f}"])

    # ROE when equity is not given but assets and liabilities are.
    if ("roe" in q or "return on equity" in q) and "equity" not in values \
            and {"total_assets", "liabilities", "net_income"} <= values.keys():
        assets, liabilities = values["total_assets"], values["liabilities"]
        equity = assets - liabilities
        if equity:
            return Derived("ROE", 100.0 * values["net_income"] / equity, "%",
                           "Net Income / (Total Assets - Total Liabilities) x 100",
                           {"net_income": values["net_income"], "total_assets": assets,
                            "liabilities": liabilities},
                           [f"equity = {assets:,.2f} - {liabilities:,.2f} = {equity:,.2f}"])

    # P/E from net income + shares + price (EPS is implied, not stated).
    if ("p/e" in q or "pe ratio" in q or "price to earnings" in q) \
            and {"net_income", "shares", "price"} <= values.keys() | {"shares", "price"} \
            and {"net_income"} <= values.keys():
        shares = values.get("shares") or values.get("shares_outstanding")
        price = values.get("price") or values.get("price_per_share")
        if shares and price:
            eps = values["net_income"] / shares
            if eps:
                return Derived("P/E Ratio", price / eps, "x",
                               "Share Price / (Net Income / Shares Outstanding)",
                               {"net_income": values["net_income"], "shares": shares,
                                "price_per_share": price},
                               [f"EPS = {values['net_income']:,.2f} / {shares:,.2f} = {eps:,.2f}"])

    # Free cash flow expressed as a share of revenue.
    if "free cash flow" in q and "percent" in q.replace("%", "percent") \
            and {"operating_cash_flow", "capex", "revenue"} <= values.keys():
        ocf, capex, revenue = values["operating_cash_flow"], values["capex"], values["revenue"]
        fcf = ocf - capex
        return Derived("Free Cash Flow Margin", 100.0 * fcf / revenue, "%",
                       "(Operating Cash Flow - CapEx) / Revenue x 100",
                       {"operating_cash_flow": ocf, "capex": capex, "revenue": revenue},
                       [f"free cash flow = {ocf:,.2f} - {capex:,.2f} = {fcf:,.2f}"])

    return None
