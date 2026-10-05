"""Monitoring conditions and saved analyses.

The rule under test is the brief's condition for building these at all: a
condition is only offered if the backend can really evaluate it, and a saved
analysis is only called saved if it really persists. So the interesting cases
here are the refusals and the absences - a metric the engine cannot compute must
be rejected at creation, and a metric missing from the current filing must read
as "not evaluable" rather than quietly as "condition not met".

Runs against a temporary database, so the application's own records are not
touched.

Run: deepseek_env/Scripts/python.exe tests/test_alerts_and_analyses.py
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.backend.services import alerts, saved_analyses  # noqa: E402

PASSED, FAILED = [], []

COMPANY = {
    "company": "Aivora Enterprise", "currency": "AUD", "period": "FY2025",
    "revenue": 10000000, "previous_revenue": 8000000, "cogs": 7000000,
    "gross_profit": 3000000, "operating_expenses": 2100000, "net_profit": 500000,
    "current_assets": 6000000, "current_liabilities": 3000000,
    "total_debt": 2500000, "total_equity": 5000000,
    "metrics": {"revenue_growth": 25.0, "gross_margin": 30.0,
                "operating_margin": 9.0, "net_margin": 5.0},
}


def check(name, ok, detail=""):
    (PASSED if ok else FAILED).append(name)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" - {detail}" if detail else ""))


def test_offered_metrics_are_computable():
    """Every metric the API offers must actually produce a number from a
    complete filing. An option that cannot be evaluated must not be offered."""
    offered = alerts.metrics()
    check("metrics are offered", len(offered) >= 10, str(len(offered)))
    for metric in offered:
        _label, _unit, extractor, _formula = alerts.METRICS[metric["key"]]
        value = extractor(COMPANY)
        check(f"{metric['key']} computes from a complete filing",
              isinstance(value, (int, float)), repr(value))
        check(f"{metric['key']} states its formula", bool(metric["formula"]))


def test_the_briefs_example_condition():
    """'Notify me when operating margin falls below 8%' - operating margin is
    9.0%, so the condition is not met, and the engine says why."""
    alerts.create("operating_margin", "below", 8)
    rule = alerts.evaluate_all(COMPANY)["rules"][0]
    check("the example condition evaluates", rule["state"] == "ok", rule["state"])
    check("it reports the actual value", rule["actual_display"] == "9.0%",
          str(rule["actual_display"]))
    check("it explains itself", "9.0%" in rule["explanation"], rule["explanation"])
    check("it shows the formula behind the metric",
          rule["formula"] == "Operating profit / Revenue x 100", rule["formula"])


def test_a_met_condition_is_reported():
    alerts.create("net_margin", "below", 10)
    alerts.create("current_ratio", "above", 1.5)
    result = alerts.evaluate_all(COMPANY)
    by_metric = {r["metric"]: r for r in result["rules"]}
    check("net margin below 10% is met",
          by_metric["net_margin"]["state"] == "triggered")
    check("current ratio above 1.5x is met",
          by_metric["current_ratio"]["state"] == "triggered")
    check("the count of met conditions is right", result["triggered"] == 2,
          str(result["triggered"]))
    check("the evaluation says it is not a background process",
          "no background process" in result["evaluation"])


def test_an_unevaluable_condition_is_refused_at_creation():
    """The whole point: a rule that could never fire is never stored."""
    for metric, operator, threshold, why in (
        ("ebitda_margin", "below", 10, "metric the engine cannot compute"),
        ("", "below", 10, "empty metric"),
        ("revenue", "sideways", 10, "unknown operator"),
        ("revenue", "below", "lots", "non-numeric threshold"),
        ("revenue", "below", None, "missing threshold"),
    ):
        try:
            alerts.create(metric, operator, threshold)
            check(f"refuses: {why}", False, "it was accepted")
        except ValueError as error:
            check(f"refuses: {why}", True, str(error)[:48])


def test_a_metric_absent_from_this_filing_is_not_evaluable():
    """Not the same as 'condition not met'. Reading an absent metric as
    satisfied would be the quiet failure this whole feature must avoid."""
    alerts.create("debt_to_equity", "above", 1)
    thin = {k: v for k, v in COMPANY.items()
            if k not in ("total_debt", "total_equity")}
    rule = next(r for r in alerts.evaluate_all(thin)["rules"]
                if r["metric"] == "debt_to_equity")
    check("an absent metric is not evaluable",
          rule["state"] == "not_evaluable", rule["state"])
    check("it is not reported as satisfied", rule["state"] != "ok")
    check("it explains why it could not be tested", bool(rule.get("reason")))
    check("it claims no value", rule["actual"] is None)


def test_state_change_between_evaluations():
    alerts.create("gross_margin", "below", 35)
    first = next(r for r in alerts.evaluate_all(COMPANY)["rules"]
                 if r["metric"] == "gross_margin")
    check("first evaluation reports no change yet",
          first["changed_since_last_check"] is False)
    improved = dict(COMPANY, metrics=dict(COMPANY["metrics"], gross_margin=40.0))
    second = next(r for r in alerts.evaluate_all(improved)["rules"]
                  if r["metric"] == "gross_margin")
    check("a crossing of the threshold is detected",
          second["changed_since_last_check"] is True
          and second["previous_state"] == "triggered"
          and second["state"] == "ok", str(second))


def test_rules_can_be_removed():
    rule = alerts.create("cogs_ratio", "above", 50)
    before = alerts.evaluate_all(COMPANY)["count"]
    alerts.delete(rule["id"])
    check("a removed rule stops being evaluated",
          alerts.evaluate_all(COMPANY)["count"] == before - 1)


def test_saved_analyses_persist_and_round_trip():
    record = saved_analyses.save(
        "FY2025 Profitability Review", "insights",
        {"items": [{"title": "Revenue growth", "figure": "25.0%"}]},
        company="Aivora Enterprise", period="FY2025",
        source_document="Aivora_Enterprise_Client_FY2025_Report.txt")
    listing = saved_analyses.listing()
    check("the analysis is listed", len(listing) == 1, str(len(listing)))
    check("the listing describes it",
          listing[0]["name"] == "FY2025 Profitability Review"
          and listing[0]["period"] == "FY2025", str(listing[0]))
    check("the listing does not carry the payload",
          "payload" not in listing[0])

    opened = saved_analyses.open_analysis(record["id"])
    check("reopening returns the figures as saved",
          opened["payload"]["items"][0]["figure"] == "25.0%", str(opened["payload"]))
    check("it records the document it came from",
          opened["source_document"].endswith(".txt"))
    check("it says the figures are a snapshot, not a fresh reading",
          "as they stood" in opened["note"])

    saved_analyses.delete(record["id"])
    check("a deleted analysis is gone", saved_analyses.listing() == [])


def test_saved_analyses_refuse_bad_input():
    for args, why in (
        (("", "analysis", {}), "no name"),
        (("   ", "analysis", {}), "blank name"),
        (("x", "nowhere", {}), "unknown view"),
    ):
        try:
            saved_analyses.save(*args)
            check(f"refuses: {why}", False, "it was accepted")
        except ValueError as error:
            check(f"refuses: {why}", True, str(error)[:46])
    try:
        saved_analyses.open_analysis(999999)
        check("refuses: opening an id that does not exist", False)
    except ValueError:
        check("refuses: opening an id that does not exist", True)


def main():
    handle, path = tempfile.mkstemp(suffix="_monitoring.db")
    os.close(handle)
    os.remove(path)
    alerts.DB_PATH = saved_analyses.DB_PATH = path
    try:
        test_offered_metrics_are_computable()
        test_the_briefs_example_condition()
        test_a_met_condition_is_reported()
        test_an_unevaluable_condition_is_refused_at_creation()
        test_a_metric_absent_from_this_filing_is_not_evaluable()
        test_state_change_between_evaluations()
        test_rules_can_be_removed()
        test_saved_analyses_persist_and_round_trip()
        test_saved_analyses_refuse_bad_input()
    finally:
        if os.path.exists(path):
            try:
                os.remove(path)
            except OSError:
                pass
    print(f"\n{len(PASSED)}/{len(PASSED) + len(FAILED)} passed")
    sys.exit(1 if FAILED else 0)


if __name__ == "__main__":
    main()
