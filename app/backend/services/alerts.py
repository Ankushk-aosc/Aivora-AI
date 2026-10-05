"""Monitoring rules: conditions on financial metrics, evaluated against the filing.

The brief's condition for building this at all is that the backend must really
be able to evaluate the rule. So:

* A rule may only name a metric this module can compute from the verified
  figures. An unknown metric is REFUSED at creation rather than stored as a
  rule that could never fire - a saved condition that is silently unevaluable
  is worse than no feature.
* Evaluation is real arithmetic over the current filing, and every result
  carries the value, the threshold and the formula that produced it.

What this is NOT, and the interface says so: there is no scheduler and no
background process, so nothing is "watching" between page loads. A rule is
evaluated when it is asked for. Because each evaluation records its outcome, a
genuine change of state - when a new filing or period moves a metric across a
threshold - is detectable and reported on the next evaluation. That is the
honest extent of "monitoring" here, and it is not described as notification.

    metrics()                       -> the metrics a rule may name
    create(metric, operator, value) -> a stored rule, or ValueError
    evaluate_all()                  -> every rule against the current figures
"""

import os
import sqlite3
import threading
import time
from contextlib import closing

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
DB_PATH = os.path.join(ROOT, "data", "monitoring.db")

# `with sqlite3.connect(...)` commits the transaction but does NOT close the
# handle, so a long-running server leaks one file handle per call - and on
# Windows the open handles keep a lock on the database file. Every use below
# wraps the connection in closing() so it is committed AND closed.

_LOCK = threading.Lock()


def _operating_profit(company):
    gross = company.get("gross_profit")
    opex = company.get("operating_expenses")
    return None if gross is None or opex is None else gross - opex


def _ratio(numerator, denominator):
    if not numerator or not denominator:
        return None
    return numerator / denominator


# key -> (label, unit, extractor, formula). Only what can actually be computed
# from the verified figures appears here; the API offers exactly this list, so
# the interface cannot present a condition the engine cannot test.
METRICS = {
    "revenue": ("Revenue", "currency",
                lambda c: c.get("revenue"), "stated in the filing"),
    "revenue_growth": ("Revenue growth", "percent",
                       lambda c: c.get("metrics", {}).get("revenue_growth"),
                       "(Revenue - Prior revenue) / Prior revenue x 100"),
    "gross_margin": ("Gross margin", "percent",
                     lambda c: c.get("metrics", {}).get("gross_margin"),
                     "Gross profit / Revenue x 100"),
    "operating_margin": ("Operating margin", "percent",
                         lambda c: c.get("metrics", {}).get("operating_margin"),
                         "Operating profit / Revenue x 100"),
    "net_margin": ("Net margin", "percent",
                   lambda c: c.get("metrics", {}).get("net_margin"),
                   "Net profit / Revenue x 100"),
    "gross_profit": ("Gross profit", "currency",
                     lambda c: c.get("gross_profit"), "stated in the filing"),
    "operating_profit": ("Operating profit", "currency", _operating_profit,
                         "Gross profit - Operating expenses"),
    "net_profit": ("Net profit", "currency",
                   lambda c: c.get("net_profit"), "stated in the filing"),
    "operating_expenses": ("Operating expenses", "currency",
                           lambda c: c.get("operating_expenses"),
                           "stated in the filing"),
    "opex_ratio": ("Operating expenses as % of gross profit", "percent",
                   lambda c: _ratio(c.get("operating_expenses"),
                                    c.get("gross_profit")) and
                   100.0 * c["operating_expenses"] / c["gross_profit"],
                   "Operating expenses / Gross profit x 100"),
    "cogs_ratio": ("COGS as % of revenue", "percent",
                   lambda c: _ratio(c.get("cogs"), c.get("revenue")) and
                   100.0 * c["cogs"] / c["revenue"],
                   "COGS / Revenue x 100"),
    "current_ratio": ("Current ratio", "times",
                      lambda c: _ratio(c.get("current_assets"),
                                       c.get("current_liabilities")),
                      "Current assets / Current liabilities"),
    "debt_to_equity": ("Debt to equity", "times",
                       lambda c: _ratio(c.get("total_debt"),
                                        c.get("total_equity")),
                       "Total debt / Total equity"),
}

OPERATORS = {
    "below": (lambda actual, threshold: actual < threshold, "falls below"),
    "above": (lambda actual, threshold: actual > threshold, "rises above"),
}


def _connect():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    connection = sqlite3.connect(DB_PATH, timeout=10)
    connection.row_factory = sqlite3.Row
    connection.execute("""
        CREATE TABLE IF NOT EXISTS rules (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            metric TEXT NOT NULL,
            operator TEXT NOT NULL,
            threshold REAL NOT NULL,
            created_at REAL NOT NULL,
            last_state TEXT,
            last_evaluated REAL
        )""")
    return connection


def metrics():
    """What a rule may be written about, with the formula behind each."""
    return [{"key": key, "label": label, "unit": unit, "formula": formula}
            for key, (label, unit, _fn, formula) in sorted(METRICS.items())]


def _format(value, unit):
    if value is None:
        return None
    if unit == "percent":
        return f"{value:.1f}%"
    if unit == "times":
        return f"{value:.2f}x"
    magnitude = abs(value)
    if magnitude >= 1_000_000:
        return f"${value / 1_000_000:.1f}M"
    if magnitude >= 1_000:
        return f"${value / 1_000:.0f}K"
    return f"${value:,.0f}"


def create(metric, operator, threshold):
    """Store a rule, or refuse it.

    Refusing is the point: a condition naming a metric the engine cannot
    compute would sit in the interface looking active and never fire.
    """
    metric = str(metric or "").strip()
    operator = str(operator or "").strip().lower()
    if metric not in METRICS:
        raise ValueError(
            f"'{metric}' is not a metric this system computes. "
            f"Available: {', '.join(sorted(METRICS))}")
    if operator not in OPERATORS:
        raise ValueError(f"Operator must be one of: {', '.join(OPERATORS)}")
    try:
        threshold = float(threshold)
    except (TypeError, ValueError):
        raise ValueError("Threshold must be a number.")

    with _LOCK, closing(_connect()) as connection, connection:
        cursor = connection.execute(
            "INSERT INTO rules (metric, operator, threshold, created_at) "
            "VALUES (?, ?, ?, ?)",
            (metric, operator, threshold, time.time()))
        return {"id": cursor.lastrowid, "metric": metric, "operator": operator,
                "threshold": threshold}


def delete(rule_id):
    with _LOCK, closing(_connect()) as connection, connection:
        cursor = connection.execute("DELETE FROM rules WHERE id = ?",
                                    (int(rule_id),))
        return {"deleted": cursor.rowcount}


def _rules():
    with closing(_connect()) as connection:
        return [dict(row) for row in
                connection.execute("SELECT * FROM rules ORDER BY id")]


def evaluate_all(company):
    """Every rule against the current figures.

    A rule whose metric cannot be computed from THIS filing comes back
    "not evaluable" with the reason, rather than quietly reading as satisfied.
    """
    results = []
    now = time.time()
    updates = []

    for rule in _rules():
        label, unit, extractor, formula = METRICS[rule["metric"]]
        try:
            actual = extractor(company)
        except Exception:                                   # noqa: BLE001
            actual = None

        test, phrasing = OPERATORS[rule["operator"]]
        condition = f"{label} {phrasing} {_format(rule['threshold'], unit)}"

        if actual is None:
            results.append({
                "id": rule["id"], "metric": rule["metric"], "label": label,
                "condition": condition, "state": "not_evaluable",
                "actual": None, "actual_display": None,
                "threshold_display": _format(rule["threshold"], unit),
                "formula": formula,
                "reason": (f"{label} cannot be computed from the current "
                           f"filing, so this condition is not tested."),
                "changed_since_last_check": False,
            })
            continue

        triggered = bool(test(actual, rule["threshold"]))
        state = "triggered" if triggered else "ok"
        changed = rule["last_state"] is not None and rule["last_state"] != state
        updates.append((state, now, rule["id"]))

        results.append({
            "id": rule["id"], "metric": rule["metric"], "label": label,
            "condition": condition, "state": state,
            "actual": actual, "actual_display": _format(actual, unit),
            "threshold_display": _format(rule["threshold"], unit),
            "formula": formula,
            "explanation": (f"{label} is {_format(actual, unit)}; the condition "
                            f"is {phrasing} {_format(rule['threshold'], unit)}."),
            "changed_since_last_check": changed,
            "previous_state": rule["last_state"],
        })

    if updates:
        with _LOCK, closing(_connect()) as connection, connection:
            connection.executemany(
                "UPDATE rules SET last_state = ?, last_evaluated = ? WHERE id = ?",
                updates)

    triggered = sum(1 for r in results if r["state"] == "triggered")
    return {
        "rules": results,
        "count": len(results),
        "triggered": triggered,
        "evaluated_at": now,
        "evaluation": ("Conditions are tested against the current filing each "
                       "time this view is opened. There is no background "
                       "process, so nothing is checked between visits."),
    }
