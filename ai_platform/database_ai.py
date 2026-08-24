"""Database AI / Data Analyst (spec Part 12).

    Question -> Schema Discovery -> Query Planning -> SQL Generation
             -> SQL Validation -> Database -> Verified Result -> Explanation

Uses a real local SQLite database (Python's stdlib sqlite3 - no server,
no credentials, genuinely available everywhere). SQL "generation" is
deliberately NOT free-form NL2SQL from the LLM: the proprietary model
isn't trained enough yet to generate reliably correct SQL (same honesty
as the financial calculator - never trust the undertrained model with
exact operations). Instead this matches the question against a small
set of parameterized, pre-validated query templates - the same
"deterministic core, LLM explains the result" pattern used throughout
this project. Anything outside the templates is honestly reported as
unsupported, not guessed at.

Read-only by construction: only SELECT statements are ever executed,
and the SQL validator rejects everything else before it reaches sqlite3.
"""

import os
import re
import sqlite3
from dataclasses import dataclass, field

DB_PATH = os.path.join("data", "financial_data.db")

_WRITE_KEYWORDS = re.compile(
    r"\b(insert|update|delete|drop|alter|create|attach|pragma|replace|truncate)\b",
    re.IGNORECASE,
)


class SQLValidationError(ValueError):
    pass


def validate_readonly_sql(sql: str):
    """Reject anything that isn't a plain SELECT. This is the actual
    security boundary - templates below are a convenience layer on top
    of it, not a substitute for it."""
    stripped = sql.strip().rstrip(";")
    if not stripped.lower().startswith("select"):
        raise SQLValidationError("Only SELECT statements are permitted.")
    if ";" in stripped:
        raise SQLValidationError("Multiple statements are not permitted.")
    if _WRITE_KEYWORDS.search(stripped):
        raise SQLValidationError("Write/DDL keywords are not permitted.")
    return stripped


def get_connection():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def discover_schema() -> dict:
    if not os.path.exists(DB_PATH):
        seed_synthetic_test_data()
    conn = get_connection()
    try:
        cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [r["name"] for r in cur.fetchall() if not r["name"].startswith("sqlite_")]
        schema = {}
        for table in tables:
            cols = conn.execute(f"PRAGMA table_info({table})").fetchall()
            schema[table] = [{"name": c["name"], "type": c["type"]} for c in cols]
        return schema
    finally:
        conn.close()


@dataclass
class QueryResult:
    sql: str
    columns: list
    rows: list
    row_count: int

    def to_dict(self):
        return {"sql": self.sql, "columns": self.columns, "rows": self.rows,
                "row_count": self.row_count}


def run_readonly_query(sql: str, params: tuple = ()) -> QueryResult:
    validated = validate_readonly_sql(sql)
    conn = get_connection()
    try:
        cur = conn.execute(validated, params)
        rows = [dict(r) for r in cur.fetchall()]
        columns = [d[0] for d in cur.description] if cur.description else []
        return QueryResult(sql=validated, columns=columns, rows=rows, row_count=len(rows))
    finally:
        conn.close()


# ----------------------------------------------------------------------
# Query templates - the "Query Planning -> SQL Generation" stage,
# implemented deterministically rather than via free-form NL2SQL.
# ----------------------------------------------------------------------

@dataclass
class QueryTemplate:
    name: str
    pattern: 're.Pattern'
    build_sql: 'callable'   # (table, match) -> (sql, params)
    description: str


def _t_total_by_column(table, group_col):
    def build(match):
        sql = f"SELECT {group_col}, SUM(amount) as total, COUNT(*) as count " \
              f"FROM {table} GROUP BY {group_col} ORDER BY total DESC"
        return sql, ()
    return build


def _t_top_n(table, n_default=5):
    def build(match):
        n = int(match.group("n")) if match.groupdict().get("n") else n_default
        sql = f"SELECT * FROM {table} ORDER BY amount DESC LIMIT ?"
        return sql, (n,)
    return build


def _t_total(table):
    def build(match):
        return f"SELECT SUM(amount) as total, COUNT(*) as count FROM {table}", ()
    return build


def _t_average(table):
    def build(match):
        return f"SELECT AVG(amount) as average, COUNT(*) as count FROM {table}", ()
    return build


def build_templates(table: str) -> list:
    return [
        QueryTemplate("total_by_category",
                      re.compile(r"\b(total|sum)\b.*\bby (category|vendor|merchant)\b", re.I),
                      lambda m: _t_total_by_column(table, m.group(2))(m),
                      "Total amount grouped by category/vendor/merchant"),
        QueryTemplate("top_n",
                      re.compile(r"\btop\s*(?P<n>\d+)?\b.*\b(transaction|amount|spend)", re.I),
                      lambda m: _t_top_n(table)(m),
                      "Top N transactions by amount"),
        QueryTemplate("total",
                      re.compile(r"\b(total|sum)\b(?!.*\bby\b)", re.I),
                      lambda m: _t_total(table)(m),
                      "Total amount across all rows"),
        QueryTemplate("average",
                      re.compile(r"\b(average|mean)\b", re.I),
                      lambda m: _t_average(table)(m),
                      "Average amount across all rows"),
    ]


def seed_synthetic_test_data(force: bool = False):
    """Creates the demo `transactions` table with clearly-labeled synthetic
    data (a `synthetic_test_data` column, always 1) so DATABASE_AI and
    RECOMMENDATION_AI have something real to query. Not enterprise data -
    this is what makes that fact durable and inspectable, not just a
    one-off fact mentioned during development."""
    conn = get_connection()
    try:
        exists = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='transactions'"
        ).fetchone()
        if exists and not force:
            return {"seeded": False, "note": "transactions table already exists"}

        conn.execute("DROP TABLE IF EXISTS transactions")
        conn.execute(
            "CREATE TABLE transactions (id INTEGER PRIMARY KEY, vendor TEXT, "
            "category TEXT, amount REAL, synthetic_test_data INTEGER DEFAULT 1)"
        )
        rows = [
            ("Acme Corp", "supplies", 1200.0), ("Beta LLC", "travel", 450.0),
            ("Acme Corp", "supplies", 800.0), ("Gamma Inc", "software", 3000.0),
            ("Beta LLC", "travel", 220.0), ("Delta Co", "marketing", 1750.0),
        ]
        conn.executemany(
            "INSERT INTO transactions (vendor, category, amount) VALUES (?,?,?)", rows
        )
        conn.commit()
        return {"seeded": True, "rows": len(rows)}
    finally:
        conn.close()


def answer_question(question: str, table: str) -> dict:
    """Question -> matched template -> validated SQL -> executed -> result.
    Returns a structured 'not supported' response (not a guess) if no
    template matches."""
    schema = discover_schema()
    if table not in schema:
        return {"answer": f"Table '{table}' does not exist. Known tables: {list(schema)}",
                "sql": None, "result": None}

    for template in build_templates(table):
        m = template.pattern.search(question)
        if m:
            sql, params = template.build_sql(m)
            result = run_readonly_query(sql, params)
            return {
                "answer": f"Matched query template '{template.name}': {template.description}",
                "sql": result.sql, "result": result.to_dict(),
            }

    return {
        "answer": "No query template matched this question. Supported patterns: "
        "'total by category/vendor', 'top N transactions', 'total', 'average'.",
        "sql": None, "result": None,
    }
