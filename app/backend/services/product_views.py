"""Comparisons, Insights and Reports - the three product areas that did not exist.

Every figure these return carries where it came from and, where it is derived,
the arithmetic that produced it, so the interface can cite rather than assert.
Nothing here invents a number: the FY2025 report contains only FY2024 revenue,
so a FY2024 line that is not in the document is returned as unavailable with a
reason, never as a blank that reads like a zero or a plausible guess.

  comparison(company)  -> period vs period, with availability per line
  insights(company)    -> evidence-backed observations
  report(company, ...) -> the sections of a financial report, as data
"""

SOURCE = "Aivora_Enterprise_Client_FY2025_Report.txt"
SOURCE_LABEL = "Aivora Enterprise FY2025 Report"

# What the source document actually states for the prior year. Revenue is the
# only FY2024 figure in it; the rest are absent, and absence is reported.
PRIOR_YEAR_AVAILABLE = ("revenue",)
PRIOR_YEAR_REASON = ("not stated for FY2024 in the source document - the FY2025 "
                     "report gives prior-year revenue only")


def _money(value):
    """$10.0M / $500.0K / $0 - the form used throughout the interface."""
    if value is None:
        return None
    sign = "-" if value < 0 else ""
    value = abs(float(value))
    if value >= 1_000_000:
        return f"{sign}${value / 1_000_000:.1f}M"
    if value >= 1_000:
        return f"{sign}${value / 1_000:.0f}K"
    return f"{sign}${value:,.0f}"


def _pct(value, places=1):
    return None if value is None else f"{value:.{places}f}%"


def _change(current, prior):
    """Absolute and percentage movement, or None when the prior is unavailable."""
    if current is None or prior in (None, 0):
        return None, None
    delta = current - prior
    return delta, 100.0 * delta / prior


def comparison(company):
    """FY2024 vs FY2025, line by line, each line honest about availability.

    The brief asks to compare revenue, growth, COGS, gross profit, margins,
    operating expenses, operating profit and net profit. Only revenue has a
    prior-year figure in the source. The other lines are returned with
    available=False and the reason, so the interface can show the current year
    and say plainly that there is nothing to compare it against.
    """
    metrics = company.get("metrics", {})
    revenue = company.get("revenue")
    prior_revenue = company.get("previous_revenue")
    cogs = company.get("cogs")
    gross_profit = company.get("gross_profit")
    opex = company.get("operating_expenses")
    net_profit = company.get("net_profit")
    operating_profit = (None if gross_profit is None or opex is None
                        else gross_profit - opex)

    lines = []

    delta, delta_pct = _change(revenue, prior_revenue)
    lines.append({
        "line": "Revenue", "current": revenue, "current_display": _money(revenue),
        "prior": prior_revenue, "prior_display": _money(prior_revenue),
        "available": True,
        "change_display": _money(delta),
        "change_pct_display": _pct(delta_pct),
        "direction": "up" if (delta or 0) > 0 else "down" if (delta or 0) < 0 else "flat",
        "source": SOURCE_LABEL,
        "basis": "both years stated in the source document",
    })

    for label, current, derived in (
        ("Cost of Goods Sold", cogs, None),
        ("Gross Profit", gross_profit, "Revenue - COGS"),
        ("Operating Expenses", opex, None),
        ("Operating Profit", operating_profit, "Gross Profit - Operating Expenses"),
        ("Net Profit", net_profit, None),
    ):
        lines.append({
            "line": label, "current": current, "current_display": _money(current),
            "prior": None, "prior_display": None,
            "available": False, "unavailable_reason": PRIOR_YEAR_REASON,
            "change_display": None, "change_pct_display": None,
            "direction": "unknown",
            "source": SOURCE_LABEL,
            "basis": derived or "stated in the source document",
        })

    ratios = []
    for label, value, formula in (
        ("Gross Margin", metrics.get("gross_margin"), "Gross Profit / Revenue x 100"),
        ("Operating Margin", metrics.get("operating_margin"),
         "Operating Profit / Revenue x 100"),
        ("Net Margin", metrics.get("net_margin"), "Net Profit / Revenue x 100"),
    ):
        ratios.append({
            "line": label, "current_display": _pct(value),
            "prior_display": None, "available": False,
            "unavailable_reason": ("the FY2024 components needed to compute this "
                                   "margin are not in the source document"),
            "formula": formula, "source": SOURCE_LABEL,
        })

    comparable = sum(1 for line in lines if line["available"])
    return {
        "company": company.get("company"),
        "currency": company.get("currency"),
        "periods": {"prior": "FY2024", "current": company.get("period", "FY2025")},
        "lines": lines,
        "ratios": ratios,
        "comparable_lines": comparable,
        "total_lines": len(lines) + len(ratios),
        "note": (f"{comparable} of {len(lines) + len(ratios)} lines have a "
                 f"prior-year figure in the source document. The rest show "
                 f"FY2025 only - no FY2024 value exists to compare against, and "
                 f"none is estimated."),
        "source_document": SOURCE,
    }


def insights(company):
    """Observations, each tied to a figure, a source and its arithmetic.

    An observation with no supporting figure is not an insight, so every entry
    carries claim, figure, calculation and source. Nothing is phrased as advice
    here - recommendations live under Management Focus in report().
    """
    metrics = company.get("metrics", {})
    revenue = company.get("revenue") or 0
    cogs = company.get("cogs") or 0
    gross_profit = company.get("gross_profit") or 0
    opex = company.get("operating_expenses") or 0
    prior_revenue = company.get("previous_revenue") or 0
    current_assets = company.get("current_assets")
    current_liabilities = company.get("current_liabilities")
    total_debt = company.get("total_debt")
    total_equity = company.get("total_equity")

    items = []

    if prior_revenue:
        items.append({
            "title": "Revenue growth",
            "figure": _pct(metrics.get("revenue_growth")),
            "claim": (f"Revenue increased from {_money(prior_revenue)} to "
                      f"{_money(revenue)}."),
            "calculation": (f"({_money(revenue)} - {_money(prior_revenue)}) / "
                            f"{_money(prior_revenue)} x 100 = "
                            f"{_pct(metrics.get('revenue_growth'))}"),
            "source": SOURCE_LABEL, "tone": "positive",
        })

    if revenue and cogs:
        items.append({
            "title": "Cost structure",
            "figure": _pct(100.0 * cogs / revenue, 0),
            "claim": (f"Direct costs absorb "
                      f"{_pct(100.0 * cogs / revenue, 0)} of revenue, leaving "
                      f"{_money(gross_profit)} of gross profit."),
            "calculation": f"{_money(cogs)} / {_money(revenue)} x 100",
            "source": SOURCE_LABEL, "tone": "watch",
        })

    if gross_profit and opex:
        items.append({
            "title": "Operating expenses",
            "figure": _money(opex),
            "claim": (f"Operating expenses consume "
                      f"{_pct(100.0 * opex / gross_profit, 0)} of gross profit, "
                      f"reducing operating margin to "
                      f"{_pct(metrics.get('operating_margin'))}."),
            "calculation": f"{_money(opex)} / {_money(gross_profit)} x 100",
            "source": SOURCE_LABEL, "tone": "watch",
        })

    if metrics.get("net_margin") is not None:
        items.append({
            "title": "Net margin",
            "figure": _pct(metrics.get("net_margin")),
            "claim": (f"{_money(company.get('net_profit'))} of every "
                      f"{_money(revenue)} of revenue converts to net profit."),
            "calculation": (f"{_money(company.get('net_profit'))} / "
                            f"{_money(revenue)} x 100"),
            "source": SOURCE_LABEL, "tone": "watch",
        })

    if current_assets and current_liabilities:
        ratio = current_assets / current_liabilities
        items.append({
            "title": "Liquidity",
            "figure": f"{ratio:.1f}x",
            "claim": (f"Current assets cover current liabilities "
                      f"{ratio:.1f} times."),
            "calculation": (f"{_money(current_assets)} / "
                            f"{_money(current_liabilities)} = {ratio:.1f}x"),
            "source": SOURCE_LABEL, "tone": "positive",
        })

    if total_debt and total_equity:
        gearing = total_debt / total_equity
        items.append({
            "title": "Gearing",
            "figure": f"{gearing:.2f}x",
            "claim": (f"Debt stands at {gearing:.2f} times shareholders' "
                      f"equity."),
            "calculation": (f"{_money(total_debt)} / {_money(total_equity)} = "
                            f"{gearing:.2f}x"),
            "source": SOURCE_LABEL, "tone": "neutral",
        })

    return {
        "company": company.get("company"),
        "period": company.get("period"),
        "items": items,
        "count": len(items),
        "basis": ("every observation is computed from figures stated in the "
                  "source document; none is estimated or forecast"),
        "source_document": SOURCE,
    }


def _ratio_rows(company):
    metrics = company.get("metrics", {})
    revenue = company.get("revenue")
    rows = [
        ("Gross Margin", _pct(metrics.get("gross_margin")),
         "Gross Profit / Revenue x 100",
         f"{_money(company.get('gross_profit'))} / {_money(revenue)}"),
        ("Operating Margin", _pct(metrics.get("operating_margin")),
         "Operating Profit / Revenue x 100",
         f"{_money((company.get('gross_profit') or 0) - (company.get('operating_expenses') or 0))} / {_money(revenue)}"),
        ("Net Margin", _pct(metrics.get("net_margin")),
         "Net Profit / Revenue x 100",
         f"{_money(company.get('net_profit'))} / {_money(revenue)}"),
        ("Revenue Growth", _pct(metrics.get("revenue_growth")),
         "(Current - Prior) / Prior x 100",
         f"({_money(revenue)} - {_money(company.get('previous_revenue'))}) / {_money(company.get('previous_revenue'))}"),
    ]
    assets = company.get("current_assets")
    liabilities = company.get("current_liabilities")
    if assets and liabilities:
        rows.append(("Current Ratio", f"{assets / liabilities:.1f}x",
                     "Current Assets / Current Liabilities",
                     f"{_money(assets)} / {_money(liabilities)}"))
    debt, equity = company.get("total_debt"), company.get("total_equity")
    if debt and equity:
        rows.append(("Debt-to-Equity", f"{debt / equity:.2f}x",
                     "Total Debt / Total Equity",
                     f"{_money(debt)} / {_money(equity)}"))
    return [{"metric": m, "value": v, "formula": f, "working": w,
             "source": SOURCE_LABEL} for m, v, f, w in rows]


def report(company, analysis=None):
    """A financial report as structured data, for preview and export.

    Returns sections rather than formatted text so the interface can render it
    and the export can serialise it without either re-deriving a figure.
    """
    analysis = analysis or {}
    metrics = company.get("metrics", {})
    revenue = company.get("revenue")
    gross_profit = company.get("gross_profit")
    opex = company.get("operating_expenses")
    operating_profit = (None if gross_profit is None or opex is None
                        else gross_profit - opex)

    statement = [
        {"line": "Revenue", "amount": _money(revenue), "emphasis": False},
        {"line": "Cost of Goods Sold", "amount": _money(company.get("cogs")),
         "emphasis": False},
        {"line": "Gross Profit", "amount": _money(gross_profit), "emphasis": True},
        {"line": "Operating Expenses", "amount": _money(opex), "emphasis": False},
        {"line": "Operating Profit", "amount": _money(operating_profit),
         "emphasis": True},
        {"line": "Net Profit", "amount": _money(company.get("net_profit")),
         "emphasis": True},
    ]

    focus = []
    for item in (analysis.get("recommendations") or []):
        focus.append({"action": item, "source": SOURCE_LABEL})
    if not focus:
        for watch in (company.get("areas_to_watch") or []):
            focus.append({"action": f"Review {watch.get('area', '').lower()}.",
                          "evidence": f"{watch.get('metric')} - {watch.get('note')}",
                          "source": SOURCE_LABEL})

    return {
        "title": f"{company.get('company')} - Financial Report",
        "company": company.get("company"),
        "period": company.get("period"),
        "currency": company.get("currency"),
        "generated_from": SOURCE,
        "sections": {
            "executive_summary": (analysis.get("executive_summary")
                                  or company.get("key_insights")),
            "financial_performance": statement,
            "profitability": [
                {"metric": "Gross Margin", "value": _pct(metrics.get("gross_margin"))},
                {"metric": "Operating Margin",
                 "value": _pct(metrics.get("operating_margin"))},
                {"metric": "Net Margin", "value": _pct(metrics.get("net_margin"))},
            ],
            "financial_ratios": _ratio_rows(company),
            "risk": [
                {"risk": r.get("title"), "severity": r.get("severity"),
                 "evidence": r.get("explanation"),
                 "management_focus": r.get("mitigation")}
                for r in (analysis.get("risks") or [])
            ],
            "risk_level": analysis.get("risk_level") or metrics.get("risk_level"),
            "management_focus": focus,
            "sources": [{"document": SOURCE, "label": SOURCE_LABEL,
                         "note": "all figures in this report are stated in or "
                                 "computed from this document"}],
        },
    }
