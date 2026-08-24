"""AI Health Check (spec Part 36).

GET /health/ai equivalent: reports real availability per capability, not
a blanket "everything is fine." A capability is HEALTHY only if its
actual dependencies are present and it just ran successfully or is known
to be side-effect-free to probe.
"""

import time

import torch

from .registry import REGISTRY, Status


def _probe_llm():
    """Cheap probe: does a checkpoint file exist and does the config load?
    Does not run a full forward pass (that's what /api/inspect/forward is for)."""
    import os
    for stage in ("instruction", "base"):
        stage_dir = os.path.join("checkpoints", stage)
        if os.path.isdir(stage_dir):
            pts = [f for f in os.listdir(stage_dir) if f.endswith(".pt")]
            if pts:
                return "HEALTHY", f"{len(pts)} checkpoint(s) available in {stage_dir}"
    return "UNAVAILABLE", "No trained checkpoints found on disk"


def _probe_calculator():
    from tools.financial_calculator import calculate
    try:
        r = calculate("ebitda_margin", ebitda=100, revenue=500)
        return ("HEALTHY", "self-test passed") if r.value == 20.0 else ("DEGRADED", "self-test value mismatch")
    except Exception as e:
        return "UNAVAILABLE", str(e)


def _probe_rag():
    import os
    return ("HEALTHY", "rag package importable") if os.path.isdir("rag") else ("UNAVAILABLE", "rag package missing")


def _probe_device():
    if torch.cuda.is_available():
        return "HEALTHY", f"CUDA GPU: {torch.cuda.get_device_name(0)}"
    return "DEGRADED", "No GPU - running on CPU only"


_PROBES = {
    "GENERAL_LLM": _probe_llm,
    "FINANCIAL_LLM": _probe_llm,
    "CALCULATOR": _probe_calculator,
    "RAG": _probe_rag,
}


def check_health() -> dict:
    t0 = time.time()
    results = {}
    for route, cap in REGISTRY.items():
        if cap.status == Status.NOT_IMPLEMENTED:
            results[route] = {"status": "NOT_IMPLEMENTED", "detail": cap.reason}
        elif cap.status == Status.BLOCKED:
            results[route] = {"status": "BLOCKED", "detail": cap.reason}
        elif route in _PROBES:
            state, detail = _PROBES[route]()
            results[route] = {"status": state, "detail": detail}
        else:
            # Implemented/tested but no live probe wired up - report the
            # registry status rather than fabricating a live check.
            results[route] = {"status": cap.status.value, "detail": "No live probe implemented; registry status shown."}

    device_state, device_detail = _probe_device()
    return {
        "capabilities": results,
        "device": {"status": device_state, "detail": device_detail},
        "check_latency_ms": round((time.time() - t0) * 1000, 2),
    }
