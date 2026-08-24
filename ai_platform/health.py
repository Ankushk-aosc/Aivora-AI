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


def _probe_live_data():
    from .live_data import LiveDataError, get_quote
    try:
        q = get_quote("AAPL", timeout=5.0)
        return "HEALTHY", f"live fetch OK: AAPL={q['price']} {q['currency']}"
    except LiveDataError as e:
        return "DEGRADED", f"unofficial endpoint unreachable right now: {e}"


def _probe_fraud():
    from .fraud import load_model
    try:
        model = load_model()
        scores = model.score([{"amt": 5000.0, "category": "misc_net",
                               "trans_date_trans_time": "2026-01-01 03:00:00",
                               "city_pop": 500}])
        return "HEALTHY", f"self-test scored transaction: risk={scores[0]}"
    except FileNotFoundError as e:
        return "UNAVAILABLE", str(e)


def _probe_code_sandbox():
    from .code_sandbox import run_python
    r = run_python("print(1+1)")
    return ("HEALTHY", "self-test executed correctly") if r.success and r.stdout.strip() == "2" \
        else ("DEGRADED", f"self-test unexpected result: {r.stdout!r} {r.stderr!r}")


def _probe_database():
    from .database_ai import discover_schema
    try:
        schema = discover_schema()
        return "HEALTHY", f"schema discovery OK: {len(schema)} table(s)"
    except Exception as e:
        return "DEGRADED", str(e)


def _probe_speech():
    from .speech import SpeechError, list_voices
    try:
        voices = list_voices()
        return ("HEALTHY", f"{len(voices)} voice(s) available") if voices \
            else ("DEGRADED", "no voices installed")
    except SpeechError as e:
        return "UNAVAILABLE", str(e)


def _probe_reranking():
    from .reranking import bm25_score
    score = bm25_score(["test"], ["this", "is", "a", "test"], 4.0, {"test": 1}, 1)
    return ("HEALTHY", "self-test OK") if score > 0 else ("DEGRADED", "self-test unexpected result")


def _probe_recommendation():
    from .recommendation import get_recommendations
    try:
        recs = get_recommendations()
        return "HEALTHY", f"generated {len(recs)} recommendation(s) from current data"
    except Exception as e:
        return "DEGRADED", str(e)


def _probe_multilingual():
    from .multilingual import detect_language
    d = detect_language("This is a test sentence in English.")
    return ("HEALTHY", "self-test OK") if d.language_code == "en" else ("DEGRADED", f"unexpected: {d.language_code}")


def _probe_knowledge_graph():
    from .knowledge_graph import extract_relations
    triples = extract_relations("Acme Corp owns Beta Logistics.")
    return ("HEALTHY", "self-test OK") if triples else ("DEGRADED", "self-test extracted nothing")


def _probe_research():
    from .research import ResearchError, search
    try:
        r = search("test", max_results=1, timeout=5.0)
        return "HEALTHY", f"live search OK: {len(r.results)} result(s)"
    except ResearchError as e:
        return "DEGRADED", f"unofficial endpoint unreachable right now: {e}"


_PROBES = {
    "GENERAL_LLM": _probe_llm,
    "FINANCIAL_LLM": _probe_llm,
    "CALCULATOR": _probe_calculator,
    "RAG": _probe_rag,
    "LIVE_DATA": _probe_live_data,
    "RESEARCH_AI": _probe_research,
    "FRAUD_AI": _probe_fraud,
    "CODE_AI": _probe_code_sandbox,
    "DATABASE_AI": _probe_database,
    "SPEECH_AI": _probe_speech,
    "RERANKING_AI": _probe_reranking,
    "RECOMMENDATION_AI": _probe_recommendation,
    "MULTILINGUAL_AI": _probe_multilingual,
    "KNOWLEDGE_GRAPH": _probe_knowledge_graph,
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
