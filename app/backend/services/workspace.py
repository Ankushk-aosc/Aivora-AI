"""Workspace -> company -> documents, and the retrieval scope that follows.

Why this exists. Document Q&A answered "What was revenue?" with Microsoft's
quarterly revenue, labelled DOCUMENT GROUNDED. The retrieval index held three
third-party sample excerpts (Apple, Microsoft, Tesla) and did NOT hold the
client's own filing, while the Documents library displayed only the client's
filing. The interface described one corpus and the engine answered from
another, which is the worst version of this failure: the citation was real, so
nothing looked wrong.

A document is answerable only if it belongs to the workspace being viewed.
That is a product rule, not a ranking tweak - no relevance score should let one
company's filing answer a question about another's. The sample files stay on
disk and can still be indexed; they are simply not this workspace's documents.

    manifest()                      -> the workspace record
    documents()                     -> filenames this workspace may answer from
    owns(path)                      -> is this file in scope
    register(filename)              -> add an uploaded document to the workspace
    ensure_indexed(store)           -> the workspace's documents are searchable
"""

import json
import os
import threading

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
MANIFEST = os.path.join(ROOT, "data", "workspace.json")
DOCUMENT_ROOT = os.path.join(ROOT, "data", "documents")

_LOCK = threading.Lock()

DEFAULT = {
    "workspace": "Aivora Enterprise",
    "company": "Aivora Enterprise",
    "period": "FY2025",
    "currency": "AUD",
    "documents": ["Aivora_Enterprise_Client_FY2025_Report.txt"],
    "note": ("Documents this workspace may be questioned about. Files on disk "
             "that are not listed here - third-party samples, for instance - "
             "are never used to answer a question about this company."),
}


def manifest():
    if os.path.exists(MANIFEST):
        try:
            with open(MANIFEST, encoding="utf-8") as handle:
                data = json.load(handle)
            if isinstance(data.get("documents"), list):
                return data
        except (OSError, ValueError):
            pass
    return dict(DEFAULT)


def _write(data):
    os.makedirs(os.path.dirname(MANIFEST), exist_ok=True)
    with open(MANIFEST, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


def documents():
    """Filenames in scope, restricted to those that actually exist on disk."""
    return [name for name in manifest().get("documents", [])
            if os.path.exists(os.path.join(DOCUMENT_ROOT, os.path.basename(name)))]


def owns(path):
    """Is `path` (or citation) a document of this workspace?"""
    if not path:
        return False
    return os.path.basename(str(path)) in set(documents())


def register(filename):
    """Record an uploaded document as part of this workspace."""
    name = os.path.basename(str(filename))
    with _LOCK:
        data = manifest()
        if name not in data.get("documents", []):
            data.setdefault("documents", []).append(name)
            _write(data)
    return name


def ensure_indexed(store):
    """Index any workspace document the store does not already hold.

    Returns the names newly indexed. Safe to call on every query: it compares
    against what the store reports and does nothing when there is nothing to
    add.
    """
    if store is None:
        return []
    try:
        known = {os.path.basename(d.get("path", ""))
                 for d in (getattr(store, "documents", None) or [])}
    except Exception:                                    # noqa: BLE001
        known = set()

    added = []
    for name in documents():
        if name in known:
            continue
        path = os.path.join(DOCUMENT_ROOT, name)
        try:
            store.add_document(path)
            added.append(name)
        except Exception:                                # noqa: BLE001
            # A document that cannot be parsed must not take the query with it;
            # it simply stays unanswerable, which the caller can see.
            continue
    return added


def scope_hits(hits):
    """Keep only hits from this workspace's documents."""
    return [hit for hit in (hits or [])
            if owns(getattr(hit, "citation", None))]
