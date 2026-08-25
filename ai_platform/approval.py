"""Human-in-the-loop approval (continuous-engineering priority #22).

    AI -> Recommendation -> Evidence -> Confidence -> Human Review
       -> Approve/Reject -> Action -> Audit

A real, persistent queue (SQLite) for high-impact actions - fraud flags
above a risk threshold, financial recommendations - so they are queued
for a human decision rather than auto-applied. "Action" here means
recording the decision and its audit trail; this project has no real
system (ERP/CRM/payment processor) to actually execute an approved
action against, so approval results in an audit record, not an
external side effect - that boundary is explicit, not glossed over.
"""

import datetime
import json
import os
import sqlite3

DB_PATH = os.path.join("data", "approvals.db")

HIGH_IMPACT_CAPABILITIES = {"FRAUD_AI", "RECOMMENDATION_AI"}


def _get_connection():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute(
        "CREATE TABLE IF NOT EXISTS approval_requests ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, "
        "capability TEXT, action TEXT, evidence TEXT, confidence TEXT, "
        "requested_by TEXT, requested_at TEXT, "
        "status TEXT DEFAULT 'pending', "
        "decided_by TEXT, decided_at TEXT, decision_reason TEXT)"
    )
    return conn


def request_approval(capability: str, action: str, evidence: dict, confidence: str,
                      requested_by: str = "system") -> dict:
    conn = _get_connection()
    try:
        now = datetime.datetime.utcnow().isoformat() + "Z"
        cur = conn.execute(
            "INSERT INTO approval_requests (capability, action, evidence, confidence, "
            "requested_by, requested_at) VALUES (?,?,?,?,?,?)",
            (capability, action, json.dumps(evidence), confidence, requested_by, now),
        )
        conn.commit()
        return get_request(cur.lastrowid)
    finally:
        conn.close()


def get_request(request_id: int) -> dict:
    conn = _get_connection()
    try:
        row = conn.execute(
            "SELECT * FROM approval_requests WHERE id = ?", (request_id,)
        ).fetchone()
        if row is None:
            raise KeyError(f"No approval request #{request_id}")
        d = dict(row)
        d["evidence"] = json.loads(d["evidence"])
        return d
    finally:
        conn.close()


def list_requests(status: str = None) -> list:
    conn = _get_connection()
    try:
        if status:
            rows = conn.execute(
                "SELECT * FROM approval_requests WHERE status = ? ORDER BY id DESC", (status,)
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM approval_requests ORDER BY id DESC"
            ).fetchall()
        out = []
        for row in rows:
            d = dict(row)
            d["evidence"] = json.loads(d["evidence"])
            out.append(d)
        return out
    finally:
        conn.close()


def decide(request_id: int, approved: bool, decided_by: str, reason: str = "") -> dict:
    """Records a human decision. Requires the caller to already have the
    'approve' permission - this module does not check auth itself
    (single-responsibility: ai_platform.auth owns permission checks,
    the API layer wires the two together), but never silently allows a
    decision to be recorded as if it were pre-authorized."""
    conn = _get_connection()
    try:
        existing = conn.execute(
            "SELECT status FROM approval_requests WHERE id = ?", (request_id,)
        ).fetchone()
        if existing is None:
            raise KeyError(f"No approval request #{request_id}")
        if existing["status"] != "pending":
            raise ValueError(
                f"Request #{request_id} is already '{existing['status']}' - "
                "cannot decide it again (immutable audit trail)."
            )

        status = "approved" if approved else "rejected"
        now = datetime.datetime.utcnow().isoformat() + "Z"
        conn.execute(
            "UPDATE approval_requests SET status=?, decided_by=?, decided_at=?, "
            "decision_reason=? WHERE id=?",
            (status, decided_by, now, reason, request_id),
        )
        conn.commit()
        return get_request(request_id)
    finally:
        conn.close()


def requires_approval(capability: str, confidence: str = None, risk_score: float = None) -> bool:
    """Gate used by the orchestrator: does this specific result need a
    human decision before being acted on? Deliberately conservative -
    low-confidence recommendations and high fraud-risk scores both
    require approval; anything not in HIGH_IMPACT_CAPABILITIES does not
    (matches spec Part 30's example list, not a blanket policy)."""
    if capability not in HIGH_IMPACT_CAPABILITIES:
        return False
    if capability == "FRAUD_AI" and risk_score is not None:
        return risk_score >= 0.5
    if capability == "RECOMMENDATION_AI" and confidence is not None:
        return confidence in ("low", "medium")
    return True
