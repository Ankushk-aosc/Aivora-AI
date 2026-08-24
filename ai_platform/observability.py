"""AI Observability (spec Part 31).

Structured JSONL request logging: one line per orchestrated request,
recording exactly the fields the spec asks for. Deliberately excludes
raw prompt/response text by default (Part 31: "Do not store sensitive
information unnecessarily") - only metadata and short previews.
"""

import datetime
import json
import os
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field

LOG_PATH = os.path.join("experiments", "ai_observability.jsonl")


@dataclass
class RequestRecord:
    request_id: str
    timestamp: str
    capability: str
    model: str = "Not available"
    tools: list = field(default_factory=list)
    retrieval_sources: int = 0
    latency_ms: float = 0.0
    tokens_generated: int = 0
    device: str = "cpu"
    error: str = None
    output_preview: str = ""

    def to_dict(self):
        return asdict(self)


@contextmanager
def track_request(capability: str, model: str = "Not available", device: str = "cpu"):
    """Usage:
        with track_request("FINANCIAL_LLM", model="checkpoint_100.pt") as rec:
            rec.tools = ["calculator"]
            ... do the work ...
            rec.output_preview = answer[:120]
    Writes one JSONL line on exit, success or failure.
    """
    record = RequestRecord(
        request_id=uuid.uuid4().hex[:12],
        timestamp=datetime.datetime.utcnow().isoformat() + "Z",
        capability=capability, model=model, device=device,
    )
    start = time.time()
    try:
        yield record
    except Exception as e:
        record.error = f"{type(e).__name__}: {e}"
        raise
    finally:
        record.latency_ms = round((time.time() - start) * 1000, 2)
        _append(record)


def _append(record: RequestRecord):
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record.to_dict()) + "\n")


def read_recent(limit: int = 50):
    if not os.path.exists(LOG_PATH):
        return []
    with open(LOG_PATH, encoding="utf-8") as f:
        lines = f.readlines()
    return [json.loads(line) for line in lines[-limit:]]


def stats():
    records = read_recent(limit=10_000)
    if not records:
        return {"total_requests": 0}
    by_capability = {}
    errors = 0
    total_latency = 0.0
    for r in records:
        by_capability[r["capability"]] = by_capability.get(r["capability"], 0) + 1
        if r.get("error"):
            errors += 1
        total_latency += r.get("latency_ms", 0)
    return {
        "total_requests": len(records),
        "by_capability": by_capability,
        "errors": errors,
        "avg_latency_ms": round(total_latency / len(records), 2),
    }
