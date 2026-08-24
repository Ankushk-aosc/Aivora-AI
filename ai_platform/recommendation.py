"""Recommendation AI (spec Part 16).

Deterministic, rule-based recommendations grounded in the real local
SQLite database (ai_platform/database_ai.py) - not an LLM guessing at
"good advice" with no data backing it. Every recommendation carries the
required fields: recommendation, reason, evidence, confidence, data_sources.

The seeded database is synthetic test data (see database_ai.py's test
seeding) - recommendations are only as meaningful as that data; this is
a demonstration of the reasoning pipeline over whatever real rows exist
in the table, not a claim about analyzing genuine enterprise spend.
"""

from dataclasses import dataclass, field

from ai_platform.anomaly import iqr_anomalies
from ai_platform.database_ai import DB_PATH, run_readonly_query


@dataclass
class Recommendation:
    recommendation: str
    reason: str
    evidence: dict
    confidence: str  # "high" | "medium" | "low" - derived from data volume, not asserted
    data_sources: list = field(default_factory=list)

    def to_dict(self):
        return self.__dict__


def _confidence_from_sample_size(n: int) -> str:
    if n >= 20:
        return "high"
    if n >= 5:
        return "medium"
    return "low"


def vendor_concentration_recommendations(table: str = "transactions") -> list:
    """Flags vendors that account for a disproportionate share of total
    spend - a real, common cost-optimization signal (single-vendor risk,
    negotiating leverage), computed from actual SUM/COUNT, not guessed."""
    result = run_readonly_query(
        f"SELECT vendor, SUM(amount) as total, COUNT(*) as n FROM {table} "
        f"GROUP BY vendor ORDER BY total DESC"
    )
    if not result.rows:
        return []

    grand_total = sum(r["total"] for r in result.rows)
    if grand_total == 0:
        return []

    recs = []
    for row in result.rows:
        share = row["total"] / grand_total
        if share >= 0.35:  # a single vendor holding >35% of spend is a common risk threshold
            recs.append(Recommendation(
                recommendation=f"Review vendor concentration risk with '{row['vendor']}'.",
                reason=f"'{row['vendor']}' accounts for {share*100:.1f}% of total spend "
                f"across {row['n']} transaction(s) - consider diversifying suppliers or "
                f"renegotiating terms given this concentration.",
                evidence={"vendor": row["vendor"], "total_spend": row["total"],
                          "transaction_count": row["n"], "share_of_total": round(share, 4),
                          "grand_total": grand_total},
                confidence=_confidence_from_sample_size(row["n"]),
                data_sources=[f"sqlite:{DB_PATH}#{table}"],
            ))
    return recs


def cost_outlier_recommendations(table: str = "transactions") -> list:
    """Reuses the already-verified IQR anomaly detector (not a second,
    untested implementation) to flag individual outlier transactions
    worth reviewing for cost-optimization purposes."""
    result = run_readonly_query(f"SELECT id, vendor, category, amount FROM {table}")
    if len(result.rows) < 4:
        return []

    amounts = [r["amount"] for r in result.rows]
    anomalies = iqr_anomalies(amounts)

    recs = []
    for anomaly, row in zip(anomalies, result.rows):
        if anomaly.is_anomaly:
            recs.append(Recommendation(
                recommendation=f"Review transaction #{row['id']} ({row['vendor']}, "
                f"{row['category']}) as a spend outlier.",
                reason=f"Amount {row['amount']} falls outside the normal IQR range "
                f"for this dataset ({anomaly.reason}).",
                evidence={"transaction_id": row["id"], "vendor": row["vendor"],
                          "category": row["category"], "amount": row["amount"],
                          "iqr_reason": anomaly.reason},
                confidence=_confidence_from_sample_size(len(result.rows)),
                data_sources=[f"sqlite:{DB_PATH}#{table}", "ai_platform.anomaly (IQR)"],
            ))
    return recs


def get_recommendations(table: str = "transactions") -> list:
    return vendor_concentration_recommendations(table) + cost_outlier_recommendations(table)
