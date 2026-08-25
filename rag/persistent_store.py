"""Persistent chunk storage for RAG (continuous-engineering priority #4).

The existing DocumentStore (rag/retriever.py) is in-memory only - every
uploaded document vanishes when the backend process restarts. This adds
real SQLite-backed persistence (reusing the sqlite3 module already
proven out in ai_platform/database_ai.py).

Deliberately persists chunk TEXT + metadata only, not embedding vectors.
DocumentStore._reindex() re-fits the TF-IDF vocabulary across ALL
currently-loaded chunks on every call - the vector space is a function
of the whole corpus, not fixed per-chunk. Persisting individual vectors
and reloading them later (possibly alongside a differently-fit query
embedding) would silently produce wrong or dimension-mismatched
similarity scores. Caught this before writing the reload path, not
after: the fix is to persist text/metadata, then call _reindex() once
after reloading everything, so the vector space is always internally
consistent.
"""

import datetime
import os
import sqlite3

DB_PATH = os.path.join("data", "vector_store.db")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    path TEXT PRIMARY KEY,
    format TEXT,
    added_at TEXT
);
CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_path TEXT,
    chunk_index INTEGER,
    text TEXT,
    page INTEGER,
    section TEXT,
    source_file TEXT,
    n_tokens INTEGER,
    FOREIGN KEY(doc_path) REFERENCES documents(path)
);
"""


def _get_connection():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)
    return conn


def save_document(doc_path: str, doc_format: str, chunks: list):
    conn = _get_connection()
    try:
        conn.execute(
            "INSERT OR REPLACE INTO documents (path, format, added_at) VALUES (?, ?, ?)",
            (doc_path, doc_format, datetime.datetime.utcnow().isoformat() + "Z"),
        )
        conn.execute("DELETE FROM chunks WHERE doc_path = ?", (doc_path,))
        for chunk in chunks:
            conn.execute(
                "INSERT INTO chunks (doc_path, chunk_index, text, page, section, "
                "source_file, n_tokens) VALUES (?,?,?,?,?,?,?)",
                (doc_path, chunk.index, chunk.text, chunk.page, chunk.section,
                 chunk.source_file, chunk.n_tokens),
            )
        conn.commit()
    finally:
        conn.close()


def load_all_chunks() -> list:
    """Reload every persisted document's chunks, in a stable order, ready
    to be re-embedded by DocumentStore._reindex()."""
    from rag.chunker import Chunk

    conn = _get_connection()
    try:
        rows = conn.execute(
            "SELECT * FROM chunks ORDER BY doc_path, chunk_index"
        ).fetchall()
        return [
            Chunk(text=r["text"], index=r["chunk_index"], page=r["page"],
                  section=r["section"], source_file=r["source_file"],
                  n_tokens=r["n_tokens"])
            for r in rows
        ]
    finally:
        conn.close()


def list_documents() -> list:
    conn = _get_connection()
    try:
        rows = conn.execute(
            "SELECT d.path, d.format, d.added_at, COUNT(c.id) as n_chunks "
            "FROM documents d LEFT JOIN chunks c ON c.doc_path = d.path "
            "GROUP BY d.path"
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def delete_document(doc_path: str):
    conn = _get_connection()
    try:
        conn.execute("DELETE FROM chunks WHERE doc_path = ?", (doc_path,))
        conn.execute("DELETE FROM documents WHERE path = ?", (doc_path,))
        conn.commit()
    finally:
        conn.close()


def clear_all():
    conn = _get_connection()
    try:
        conn.execute("DELETE FROM chunks")
        conn.execute("DELETE FROM documents")
        conn.commit()
    finally:
        conn.close()
