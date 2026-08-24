"""Anomaly Detection AI (spec Part 13).

Deterministic statistical methods, not a trained neural network - there
is no labeled anomaly dataset in this project to train one on, so this
does not pretend to be more than it is: z-score / IQR outlier scoring on
numeric fields, plus exact/near-duplicate invoice detection reusing the
same content-hash approach as the dataset deduplicator (data_sources/
cleaning.py), so "verified once, reused twice" rather than reimplemented.
"""

import statistics
from dataclasses import dataclass, field

from data_sources.cleaning import content_hash

DEFAULT_Z_THRESHOLD = 3.0


@dataclass
class AnomalyResult:
    index: int
    value: float
    z_score: float
    is_anomaly: bool
    method: str
    reason: str


def zscore_anomalies(values, threshold: float = DEFAULT_Z_THRESHOLD):
    """Flag values whose z-score exceeds `threshold`. Requires >= 2 values
    with nonzero variance; returns no anomalies (not a crash) otherwise -
    there's no statistical basis to call anything an outlier from too little
    data, and that's reported honestly.

    Known limitation (the "masking effect"): a single dominant outlier
    inflates the standard deviation it is itself measured against, which
    can push its own z-score back under the threshold in small samples.
    detect_transaction_anomalies() defaults to IQR for this reason - IQR
    uses quartiles, which a single extreme value barely shifts.
    """
    if len(values) < 2:
        return []
    mean = statistics.mean(values)
    stdev = statistics.pstdev(values)
    if stdev == 0:
        return []
    results = []
    for i, v in enumerate(values):
        z = (v - mean) / stdev
        is_anom = abs(z) > threshold
        results.append(AnomalyResult(
            index=i, value=v, z_score=round(z, 3), is_anomaly=is_anom,
            method="zscore",
            reason=f"|z|={abs(z):.2f} {'>' if is_anom else '<='} threshold {threshold}",
        ))
    return results


def iqr_anomalies(values, k: float = 1.5):
    """Flag values outside [Q1 - k*IQR, Q3 + k*IQR] - the standard
    boxplot-outlier rule, robust to a few extreme values (unlike z-score,
    which the outliers themselves can distort)."""
    if len(values) < 4:
        return []
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    q1 = sorted_vals[n // 4]
    q3 = sorted_vals[(3 * n) // 4]
    iqr = q3 - q1
    lower, upper = q1 - k * iqr, q3 + k * iqr
    results = []
    for i, v in enumerate(values):
        is_anom = v < lower or v > upper
        results.append(AnomalyResult(
            index=i, value=v, z_score=None, is_anomaly=is_anom, method="iqr",
            reason=f"range=[{lower:.2f}, {upper:.2f}]" if is_anom else "within range",
        ))
    return results


@dataclass
class DuplicateGroup:
    hash: str
    indices: list = field(default_factory=list)
    records: list = field(default_factory=list)


def duplicate_invoices(records, key_fields=("vendor", "amount", "date")):
    """Detect exact-duplicate invoices by hashing the normalized key
    fields together. Two invoices with the same vendor/amount/date are
    flagged for human review, not auto-rejected (Part 30: human-in-the-loop
    for high-impact actions)."""
    seen = {}
    groups = []
    for i, record in enumerate(records):
        key = "|".join(str(record.get(f, "")).strip().lower() for f in key_fields)
        h = content_hash(key)
        if h in seen:
            seen[h].indices.append(i)
            seen[h].records.append(record)
        else:
            group = DuplicateGroup(hash=h, indices=[i], records=[record])
            seen[h] = group
            groups.append(group)
    return [g for g in groups if len(g.indices) > 1]


def detect_transaction_anomalies(transactions, amount_field="amount", method="iqr",
                                  threshold: float = DEFAULT_Z_THRESHOLD):
    """High-level entry point: transactions -> anomaly scores -> flagged records.
    Matches the spec's pipeline: Data -> Anomaly Model -> Anomaly Score ->
    Threshold/Rules -> Human Review -> AI Explanation. This function returns
    everything up to Threshold/Rules; explanation/human-review is a
    downstream concern (the orchestrator/chat layer), not baked in here."""
    amounts = [float(t.get(amount_field, 0)) for t in transactions]
    if method == "iqr":
        scored = iqr_anomalies(amounts)
    else:
        scored = zscore_anomalies(amounts, threshold=threshold)

    flagged = []
    for result, txn in zip(scored, transactions):
        if result.is_anomaly:
            flagged.append({
                "transaction": txn, "z_score": result.z_score,
                "reason": result.reason, "method": result.method,
            })

    return {
        "total_transactions": len(transactions),
        "flagged_count": len(flagged),
        "flagged": flagged,
        "method": method,
        "note": "Statistical outlier detection (unsupervised), not a trained "
        "fraud classifier - see ANOMALY_AI vs FRAUD_AI in the capability registry.",
    }
