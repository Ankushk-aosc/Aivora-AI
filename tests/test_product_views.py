"""Comparisons, Insights, Reports, and the workspace scope behind document Q&A.

The rule these guard: a figure is shown only if the source document states it or
it is computed from figures the source states. The interesting cases are the
absences - a prior-year line that does not exist must be reported as absent, not
rendered as a blank that reads like zero, and never estimated.

Run: deepseek_env/Scripts/python.exe tests/test_product_views.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.backend.services.product_views import (  # noqa: E402
    comparison, insights, report,
)
from app.backend.services import workspace  # noqa: E402

PASSED, FAILED = [], []

COMPANY = {
    "company": "Aivora Enterprise", "currency": "AUD", "period": "FY2025",
    "revenue": 10000000, "previous_revenue": 8000000, "cogs": 7000000,
    "gross_profit": 3000000, "operating_expenses": 2100000, "net_profit": 500000,
    "current_assets": 6000000, "current_liabilities": 3000000,
    "total_debt": 2500000, "total_equity": 5000000,
    "metrics": {"revenue_growth": 25.0, "gross_margin": 30.0,
                "operating_margin": 9.0, "net_margin": 5.0, "risk_level": "Medium"},
    "key_insights": "Revenue increased 25% year-over-year.",
    "areas_to_watch": [
        {"area": "Operating expenses", "metric": "$2.1M",
         "note": "Consumes 70.0% of gross profit."},
    ],
}


def check(name, ok, detail=""):
    (PASSED if ok else FAILED).append(name)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" - {detail}" if detail else ""))


def test_comparison_states_the_figures_it_has():
    data = comparison(COMPANY)
    revenue = next(l for l in data["lines"] if l["line"] == "Revenue")
    check("revenue compares both years", revenue["available"]
          and revenue["prior_display"] == "$8.0M"
          and revenue["current_display"] == "$10.0M", str(revenue))
    check("revenue change is 25.0%", revenue["change_pct_display"] == "25.0%",
          str(revenue["change_pct_display"]))
    check("revenue change direction is up", revenue["direction"] == "up")


def test_comparison_reports_absence_rather_than_inventing():
    """The source gives FY2024 revenue only. Every other prior-year figure must
    come back unavailable, with a reason, and with no number attached."""
    data = comparison(COMPANY)
    absent = [l for l in data["lines"] if not l["available"]]
    check("five lines have no prior-year figure", len(absent) == 5, str(len(absent)))
    for line in absent:
        check(f"{line['line']}: no prior value invented",
              line["prior"] is None and line["prior_display"] is None)
        check(f"{line['line']}: absence is explained",
              bool(line.get("unavailable_reason")))
        check(f"{line['line']}: no change is claimed",
              line["change_display"] is None
              and line["change_pct_display"] is None)
    check("the current-year figure is still shown",
          all(l["current_display"] for l in absent))
    check("margins are not compared against a year that has none",
          all(not r["available"] for r in data["ratios"]))
    check("the count of comparable lines is stated",
          data["comparable_lines"] == 1, str(data["comparable_lines"]))


def test_derived_lines_say_they_are_derived():
    data = comparison(COMPANY)
    gross = next(l for l in data["lines"] if l["line"] == "Gross Profit")
    operating = next(l for l in data["lines"] if l["line"] == "Operating Profit")
    check("gross profit shows its derivation", gross["basis"] == "Revenue - COGS")
    check("operating profit shows its derivation",
          operating["basis"] == "Gross Profit - Operating Expenses")
    check("operating profit is computed, not assumed",
          operating["current"] == 900000, str(operating["current"]))


def test_every_insight_carries_its_evidence():
    data = insights(COMPANY)
    check("insights are produced", data["count"] >= 5, str(data["count"]))
    for item in data["items"]:
        check(f"{item['title']}: has a figure", bool(item["figure"]))
        check(f"{item['title']}: has a claim", bool(item["claim"]))
        check(f"{item['title']}: shows its arithmetic", bool(item["calculation"]))
        check(f"{item['title']}: names its source", bool(item["source"]))
    growth = next(i for i in data["items"] if i["title"] == "Revenue growth")
    check("growth arithmetic is the real one",
          "($10.0M - $8.0M) / $8.0M" in growth["calculation"],
          growth["calculation"])


def test_insights_degrade_rather_than_guess():
    """With liquidity figures absent, the liquidity observation disappears - it
    is not rendered with an empty or assumed value."""
    thin = {k: v for k, v in COMPANY.items()
            if k not in ("current_assets", "current_liabilities",
                         "total_debt", "total_equity")}
    data = insights(thin)
    titles = [i["title"] for i in data["items"]]
    check("liquidity is omitted when its inputs are absent",
          "Liquidity" not in titles, str(titles))
    check("gearing is omitted when its inputs are absent",
          "Gearing" not in titles, str(titles))
    check("the observations that remain are still complete",
          all(i["calculation"] and i["source"] for i in data["items"]))


def test_report_sections_are_complete_and_sourced():
    data = report(COMPANY, {"executive_summary": "Summary.",
                            "risk_level": "Medium",
                            "risks": [{"title": "High Operating Expenses",
                                       "severity": "High",
                                       "explanation": "Consumes 70% of gross profit.",
                                       "mitigation": "Review overheads."}],
                            "recommendations": ["Optimise operating expenses."]})
    sections = data["sections"]
    for name in ("executive_summary", "financial_performance", "profitability",
                 "financial_ratios", "risk", "management_focus", "sources"):
        check(f"report has a {name} section", name in sections and sections[name])
    check("the income statement has every line",
          [r["line"] for r in sections["financial_performance"]] ==
          ["Revenue", "Cost of Goods Sold", "Gross Profit", "Operating Expenses",
           "Operating Profit", "Net Profit"])
    check("subtotals are emphasised, detail lines are not",
          [r["line"] for r in sections["financial_performance"] if r["emphasis"]]
          == ["Gross Profit", "Operating Profit", "Net Profit"])
    check("every ratio shows a formula and its working",
          all(r["formula"] and r["working"] for r in sections["financial_ratios"]))
    check("the ratios are the ones the brief asks for",
          {r["metric"] for r in sections["financial_ratios"]} >=
          {"Gross Margin", "Operating Margin", "Net Margin", "Revenue Growth",
           "Current Ratio", "Debt-to-Equity"})
    current = next(r for r in sections["financial_ratios"]
                   if r["metric"] == "Current Ratio")
    check("current ratio is 2.0x from $6M / $3M",
          current["value"] == "2.0x" and "$6.0M / $3.0M" in current["working"],
          str(current))
    check("the report names its source document",
          sections["sources"][0]["document"].endswith(".txt"))


def test_report_survives_a_missing_risk_engine():
    data = report(COMPANY, None)
    check("a report is still produced without analysis",
          bool(data["sections"]["executive_summary"]))
    focus = data["sections"]["management_focus"]
    check("management focus falls back to the watch list", len(focus) >= 1,
          str(focus))
    check("the fallback carries the evidence behind each item",
          all(item.get("evidence") and item.get("source") for item in focus),
          str(focus))

    # With neither analysis nor a watch list there is nothing to say, and an
    # empty list is the honest result - the interface omits the section rather
    # than printing a heading over nothing.
    bare = {k: v for k, v in COMPANY.items() if k != "areas_to_watch"}
    check("no focus items are invented when there is no evidence",
          report(bare, None)["sections"]["management_focus"] == [])


def test_money_formatting():
    data = comparison(COMPANY)
    shown = {l["line"]: l["current_display"] for l in data["lines"]}
    for line, expected in (("Revenue", "$10.0M"), ("Net Profit", "$500K"),
                           ("Operating Profit", "$900K")):
        check(f"{line} formats as {expected}", shown[line] == expected, shown[line])


def test_workspace_scope():
    """The rule that stopped one company's filing answering for another."""
    names = workspace.documents()
    check("the workspace holds the client filing",
          any("Aivora_Enterprise" in n for n in names), str(names))
    check("a third-party sample is not a workspace document",
          not workspace.owns("Microsoft_FY2024_Q4_Excerpt.txt"))
    check("the client filing is a workspace document",
          workspace.owns("Aivora_Enterprise_Client_FY2025_Report.txt"))
    check("a full path is matched by basename",
          workspace.owns("/anywhere/Aivora_Enterprise_Client_FY2025_Report.txt"))
    check("nothing is owned by an empty citation", not workspace.owns(None))

    class Hit:
        def __init__(self, citation):
            self.citation = citation

    hits = [Hit("Microsoft_FY2024_Q4_Excerpt.txt"),
            Hit("Aivora_Enterprise_Client_FY2025_Report.txt"),
            Hit("Tesla_Q3_2024_Financial_Summary.txt")]
    kept = workspace.scope_hits(hits)
    check("out-of-workspace hits are dropped before they can answer",
          len(kept) == 1
          and kept[0].citation == "Aivora_Enterprise_Client_FY2025_Report.txt",
          str([h.citation for h in kept]))
    check("scoping an empty result is safe", workspace.scope_hits([]) == [])


def main():
    test_comparison_states_the_figures_it_has()
    test_comparison_reports_absence_rather_than_inventing()
    test_derived_lines_say_they_are_derived()
    test_every_insight_carries_its_evidence()
    test_insights_degrade_rather_than_guess()
    test_report_sections_are_complete_and_sourced()
    test_report_survives_a_missing_risk_engine()
    test_money_formatting()
    test_workspace_scope()
    print(f"\n{len(PASSED)}/{len(PASSED) + len(FAILED)} passed")
    sys.exit(1 if FAILED else 0)


if __name__ == "__main__":
    main()
