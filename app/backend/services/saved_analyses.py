"""Saved analyses: name a view of the figures, reopen it later.

The brief allows a local-storage fallback if server-side persistence is not
available. It is available - SQLite already backs this application's approvals
and auth - so these are stored server-side and survive a restart, a new browser
and a different machine. Nothing here pretends to be more durable than it is:
the store is a file in data/, and that is what the interface says.

A saved analysis records the figures AS THEY WERE when it was saved. That is the
point of saving one: reopening it must show what the author saw. It therefore
also records the document and period it was taken from, so a reader can tell
whether it still reflects the current filing.

    save(name, view, payload)  -> the stored record
    listing()                  -> saved analyses, newest first
    open(analysis_id)          -> one record, with its payload
    delete(analysis_id)
"""

import json
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

MAX_NAME = 120
MAX_PAYLOAD_BYTES = 256 * 1024
VIEWS = ("analysis", "comparisons", "insights", "reports", "overview")


def _connect():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    connection = sqlite3.connect(DB_PATH, timeout=10)
    connection.row_factory = sqlite3.Row
    connection.execute("""
        CREATE TABLE IF NOT EXISTS analyses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            view TEXT NOT NULL,
            company TEXT,
            period TEXT,
            source_document TEXT,
            payload TEXT NOT NULL,
            created_at REAL NOT NULL
        )""")
    return connection


def save(name, view, payload, company=None, period=None, source_document=None):
    name = str(name or "").strip()
    if not name:
        raise ValueError("Give the analysis a name.")
    if len(name) > MAX_NAME:
        name = name[:MAX_NAME]
    view = str(view or "").strip().lower()
    if view not in VIEWS:
        raise ValueError(f"View must be one of: {', '.join(VIEWS)}")

    serialised = json.dumps(payload if payload is not None else {})
    if len(serialised.encode("utf-8")) > MAX_PAYLOAD_BYTES:
        raise ValueError("That analysis is too large to save "
                         f"(limit {MAX_PAYLOAD_BYTES // 1024} KB).")

    with _LOCK, closing(_connect()) as connection, connection:
        cursor = connection.execute(
            "INSERT INTO analyses (name, view, company, period, "
            "source_document, payload, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (name, view, company, period, source_document, serialised,
             time.time()))
        return {"id": cursor.lastrowid, "name": name, "view": view,
                "created_at": time.time()}


def listing():
    """Saved analyses, newest first. The payload is not included - the list
    only needs to describe them."""
    with closing(_connect()) as connection:
        rows = connection.execute(
            "SELECT id, name, view, company, period, source_document, "
            "created_at FROM analyses ORDER BY created_at DESC")
        return [{
            "id": row["id"], "name": row["name"], "view": row["view"],
            "company": row["company"], "period": row["period"],
            "source_document": row["source_document"],
            "created_at": row["created_at"],
            "created_display": time.strftime("%Y-%m-%d %H:%M",
                                             time.localtime(row["created_at"])),
        } for row in rows]


def open_analysis(analysis_id):
    with closing(_connect()) as connection:
        row = connection.execute("SELECT * FROM analyses WHERE id = ?",
                                 (int(analysis_id),)).fetchone()
    if row is None:
        raise ValueError(f"No saved analysis with id {analysis_id}.")
    try:
        payload = json.loads(row["payload"])
    except ValueError:
        payload = {}
    return {
        "id": row["id"], "name": row["name"], "view": row["view"],
        "company": row["company"], "period": row["period"],
        "source_document": row["source_document"],
        "created_at": row["created_at"],
        "created_display": time.strftime("%Y-%m-%d %H:%M",
                                         time.localtime(row["created_at"])),
        "payload": payload,
        "note": ("These are the figures as they stood when this analysis was "
                 "saved, not a fresh reading of the filing."),
    }


def delete(analysis_id):
    with _LOCK, closing(_connect()) as connection, connection:
        cursor = connection.execute("DELETE FROM analyses WHERE id = ?",
                                    (int(analysis_id),))
        return {"deleted": cursor.rowcount}
