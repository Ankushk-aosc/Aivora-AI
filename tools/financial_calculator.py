"""Deterministic financial calculations (Part 19 / §35).

The language model is never responsible for exact arithmetic: these
functions compute the numbers, and the model explains the result.

Every function returns a CalcResult carrying the value, the formula that
was applied, and the inputs used, so an answer can cite exactly how it
was derived. Division-by-zero and missing inputs raise CalculationError
rather than returning a fabricated number.
"""

from dataclasses import dataclass, field
from typing import Optional


class CalculationError(ValueError):
    """Raised when a calculation cannot be performed with the given inputs."""


@dataclass
class CalcResult:
    name: str
    value: float
    unit: str
    formula: str
    inputs: dict = field(default_factory=dict)

    def formatted(self, places: int = 2) -> str:
        if self.unit == "%":
            return f"{self.value:.{places}f}%"
        if self.unit == "x":
            return f"{self.value:.{places}f}x"
        return f"{self.value:,.{places}f}"

    def explain(self) -> str:
        inputs = ", ".join(f"{k}={v:,}" for k, v in self.inputs.items())
        return f"{self.name} = {self.formatted()}  [{self.formula}; {inputs}]"


def _require_nonzero(value: float, label: str) -> float:
    if value is None:
        raise CalculationError(f"{label} is required")
    if value == 0:
        raise CalculationError(f"{label} must be non-zero (got 0)")
    return value


def _ratio_pct(numerator: float, denominator: float, denom_label: str) -> float:
    _require_nonzero(denominator, denom_label)
    return (numerator / denominator) * 100.0


# ----------------------------------------------------------------------
# Growth
# ----------------------------------------------------------------------

def revenue_growth(current_revenue: float, prior_revenue: float) -> CalcResult:
    value = _ratio_pct(current_revenue - prior_revenue, prior_revenue, "prior_revenue")
    return CalcResult(
        "Revenue Growth", value, "%",
        "(Current Revenue - Prior Revenue) / Prior Revenue x 100",
        {"current_revenue": current_revenue, "prior_revenue": prior_revenue},
    )


def cagr(beginning_value: float, ending_value: float, years: float) -> CalcResult:
    _require_nonzero(beginning_value, "beginning_value")
    _require_nonzero(years, "years")
    if beginning_value < 0 or ending_value < 0:
        raise CalculationError("CAGR is undefined for negative values")
    value = ((ending_value / beginning_value) ** (1.0 / years) - 1.0) * 100.0
    return CalcResult(
        "CAGR", value, "%",
        "((Ending / Beginning)^(1/Years) - 1) x 100",
        {"beginning_value": beginning_value, "ending_value": ending_value, "years": years},
    )


# ----------------------------------------------------------------------
# Margins
# ----------------------------------------------------------------------

def simple_profit(revenue: float, expenses: float) -> CalcResult:
    return CalcResult(
        "Profit", revenue - expenses, "",
        "Revenue - Expenses",
        {"revenue": revenue, "expenses": expenses},
    )


def gross_margin(gross_profit: float, revenue: float) -> CalcResult:
    return CalcResult(
        "Gross Margin", _ratio_pct(gross_profit, revenue, "revenue"), "%",
        "Gross Profit / Revenue x 100",
        {"gross_profit": gross_profit, "revenue": revenue},
    )


def operating_margin(operating_income: float, revenue: float) -> CalcResult:
    return CalcResult(
        "Operating Margin", _ratio_pct(operating_income, revenue, "revenue"), "%",
        "Operating Income / Revenue x 100",
        {"operating_income": operating_income, "revenue": revenue},
    )


def ebitda_margin(ebitda: float, revenue: float) -> CalcResult:
    return CalcResult(
        "EBITDA Margin", _ratio_pct(ebitda, revenue, "revenue"), "%",
        "EBITDA / Revenue x 100",
        {"ebitda": ebitda, "revenue": revenue},
    )


def net_profit_margin(net_income: float, revenue: float) -> CalcResult:
    return CalcResult(
        "Net Profit Margin", _ratio_pct(net_income, revenue, "revenue"), "%",
        "Net Income / Revenue x 100",
        {"net_income": net_income, "revenue": revenue},
    )


# ----------------------------------------------------------------------
# Returns
# ----------------------------------------------------------------------

def roe(net_income: float, shareholders_equity: float) -> CalcResult:
    return CalcResult(
        "ROE", _ratio_pct(net_income, shareholders_equity, "shareholders_equity"), "%",
        "Net Income / Shareholders' Equity x 100",
        {"net_income": net_income, "shareholders_equity": shareholders_equity},
    )


def roa(net_income: float, total_assets: float) -> CalcResult:
    return CalcResult(
        "ROA", _ratio_pct(net_income, total_assets, "total_assets"), "%",
        "Net Income / Total Assets x 100",
        {"net_income": net_income, "total_assets": total_assets},
    )


def roic(nopat: float, invested_capital: float) -> CalcResult:
    return CalcResult(
        "ROIC", _ratio_pct(nopat, invested_capital, "invested_capital"), "%",
        "NOPAT / Invested Capital x 100",
        {"nopat": nopat, "invested_capital": invested_capital},
    )


# ----------------------------------------------------------------------
# Leverage / liquidity
# ----------------------------------------------------------------------

def debt_to_equity(total_debt: float, shareholders_equity: float) -> CalcResult:
    _require_nonzero(shareholders_equity, "shareholders_equity")
    return CalcResult(
        "Debt/Equity", total_debt / shareholders_equity, "x",
        "Total Debt / Shareholders' Equity",
        {"total_debt": total_debt, "shareholders_equity": shareholders_equity},
    )


def current_ratio(current_assets: float, current_liabilities: float) -> CalcResult:
    _require_nonzero(current_liabilities, "current_liabilities")
    return CalcResult(
        "Current Ratio", current_assets / current_liabilities, "x",
        "Current Assets / Current Liabilities",
        {"current_assets": current_assets, "current_liabilities": current_liabilities},
    )


# ----------------------------------------------------------------------
# Cash flow / per-share / valuation
# ----------------------------------------------------------------------

def free_cash_flow(operating_cash_flow: float, capital_expenditure: float) -> CalcResult:
    return CalcResult(
        "Free Cash Flow", operating_cash_flow - capital_expenditure, "",
        "Operating Cash Flow - Capital Expenditure",
        {"operating_cash_flow": operating_cash_flow, "capital_expenditure": capital_expenditure},
    )


def eps(net_income: float, shares_outstanding: float,
        preferred_dividends: float = 0.0) -> CalcResult:
    _require_nonzero(shares_outstanding, "shares_outstanding")
    return CalcResult(
        "EPS", (net_income - preferred_dividends) / shares_outstanding, "",
        "(Net Income - Preferred Dividends) / Shares Outstanding",
        {"net_income": net_income, "shares_outstanding": shares_outstanding,
         "preferred_dividends": preferred_dividends},
    )


def pe_ratio(price_per_share: float, earnings_per_share: float) -> CalcResult:
    _require_nonzero(earnings_per_share, "earnings_per_share")
    return CalcResult(
        "P/E", price_per_share / earnings_per_share, "x",
        "Price per Share / Earnings per Share",
        {"price_per_share": price_per_share, "earnings_per_share": earnings_per_share},
    )


def ev_to_ebitda(enterprise_value: float, ebitda: float) -> CalcResult:
    _require_nonzero(ebitda, "ebitda")
    return CalcResult(
        "EV/EBITDA", enterprise_value / ebitda, "x",
        "Enterprise Value / EBITDA",
        {"enterprise_value": enterprise_value, "ebitda": ebitda},
    )


# ----------------------------------------------------------------------
# Registry used by the query router
# ----------------------------------------------------------------------

CALCULATIONS = {
    "revenue_growth": revenue_growth,
    "cagr": cagr,
    "simple_profit": simple_profit,
    "gross_margin": gross_margin,
    "operating_margin": operating_margin,
    "ebitda_margin": ebitda_margin,
    "net_profit_margin": net_profit_margin,
    "roe": roe,
    "roa": roa,
    "roic": roic,
    "debt_to_equity": debt_to_equity,
    "current_ratio": current_ratio,
    "free_cash_flow": free_cash_flow,
    "eps": eps,
    "pe_ratio": pe_ratio,
    "ev_to_ebitda": ev_to_ebitda,
}


def calculate(name: str, **kwargs) -> CalcResult:
    if name not in CALCULATIONS:
        raise CalculationError(
            f"Unknown calculation '{name}'. Available: {sorted(CALCULATIONS)}"
        )
    return CALCULATIONS[name](**kwargs)
