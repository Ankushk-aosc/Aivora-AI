"""Backend API (Part 24 / §50).

Thin HTTP layer over the already-verified services - it adds no model,
data, or scoring logic of its own:

  tools.financial_calculator            -> /api/calculate
  app.backend.services.financial_router -> /api/route
  app.backend.services.chat_service     -> /api/chat
  app.backend.services.inspector        -> /api/inspect/*
  rag.DocumentStore                     -> /api/rag/*
  evaluation, experiments, data_sources -> /api/evaluation, /api/experiments, /api/datasets

Implemented on http.server so the project gains no new hard dependency;
every endpoint returns real runtime data or an explicit "Not available".
"""

import json
import os
import sys
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    # Allows `python app/backend/server.py` to import the repo's top-level
    # packages (models, training, data_sources, ...) the same way
    # `python main.py` does.
    sys.path.insert(0, ROOT)

# Lazily-populated process state.
STATE = {
    "checkpoint": None,
    "model": None,
    "config": None,
    "device": "cpu",
    "chat": None,
    "orchestrator": None,
    "document_store": None,
    "knowledge_graph": None,
    "load_error": None,
}
_LOCK = threading.Lock()


def load_checkpoint(checkpoint_path):
    """Load a checkpoint into process state. Returns a status dict."""
    from inference import load_model_for_inference
    from training.trainer import detect_device

    with _LOCK:
        device = detect_device()
        device = "cpu" if device == "mps" else device
        try:
            model, config = load_model_for_inference(checkpoint_path, device=device)
        except Exception as e:
            STATE["load_error"] = f"{type(e).__name__}: {e}"
            STATE["model"] = None
            return {"loaded": False, "error": STATE["load_error"]}

        from app.backend.services.chat_service import FinancialChat
        from ai_platform import AIOrchestrator

        STATE.update({
            "checkpoint": checkpoint_path,
            "model": model,
            "config": config,
            "device": device,
            "load_error": None,
        })
        STATE["chat"] = FinancialChat(
            model=model, device=device,
            document_store=STATE.get("document_store"), max_new_tokens=40,
        )
        STATE["orchestrator"] = AIOrchestrator(
            model=model, device=device, document_store=STATE.get("document_store"),
        )
        return {
            "loaded": True,
            "checkpoint": checkpoint_path,
            "device": device,
            "parameters": sum(p.numel() for p in model.parameters()),
        }


def _require_model():
    if STATE["model"] is None:
        raise ValueError(
            "No checkpoint loaded. POST /api/model/load with "
            '{"checkpoint": "checkpoints/base/checkpoint_100.pt"}'
        )
    return STATE["model"]


# ----------------------------------------------------------------------
# Handlers
# ----------------------------------------------------------------------

def h_health(_payload, _query):
    import torch
    return {
        "status": "ok",
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Not available",
        "checkpoint_loaded": STATE["checkpoint"],
    }


def h_model_status(_payload, _query):
    if STATE["model"] is None:
        return {"loaded": False, "error": STATE["load_error"] or "No checkpoint loaded",
                "available_checkpoints": h_checkpoints(None, None)["checkpoints"]}
    from app.backend.services.inspector import inspect_architecture
    return {"loaded": True, "checkpoint": STATE["checkpoint"],
            "architecture": inspect_architecture(STATE["model"], STATE["device"])}


def h_model_load(payload, _query):
    path = (payload or {}).get("checkpoint")
    if not path:
        raise ValueError('Provide {"checkpoint": "<path to .pt>"}')
    if not os.path.exists(path):
        raise ValueError(f"Checkpoint not found: {path}")
    return load_checkpoint(path)


def h_checkpoints(_payload, _query):
    out = []
    for stage in ("base", "financial", "instruction"):
        stage_dir = os.path.join(ROOT, "checkpoints", stage)
        if not os.path.isdir(stage_dir):
            continue
        for fname in sorted(f for f in os.listdir(stage_dir) if f.endswith(".pt")):
            meta_path = os.path.join(stage_dir, fname.replace(".pt", ".json"))
            meta = {}
            if os.path.exists(meta_path):
                with open(meta_path) as f:
                    meta = json.load(f)
            out.append({
                "stage": stage,
                "path": os.path.relpath(os.path.join(stage_dir, fname), ROOT).replace("\\", "/"),
                "step": meta.get("step"),
                "train_loss": meta.get("train_loss"),
                "val_loss": meta.get("val_loss"),
                "tokens_processed": meta.get("tokens_processed"),
                "timestamp": meta.get("timestamp"),
            })
    return {"checkpoints": out}


def h_calculate(payload, _query):
    from tools.financial_calculator import CALCULATIONS, CalculationError, calculate

    payload = payload or {}
    name = payload.get("calculation")
    if not name:
        return {"available": sorted(CALCULATIONS)}
    try:
        result = calculate(name, **(payload.get("inputs") or {}))
    except CalculationError as e:
        return {"error": str(e), "calculation": name}
    return {
        "calculation": name, "name": result.name, "value": result.value,
        "unit": result.unit, "formatted": result.formatted(),
        "formula": result.formula, "inputs": result.inputs,
    }


def h_route(payload, query):
    from app.backend.services.financial_router import classify, extract_financial_values

    text = (payload or {}).get("query") or (query.get("query", [None])[0])
    if not text:
        raise ValueError('Provide {"query": "..."}')
    store = STATE.get("document_store")
    decision = classify(text, has_document=bool(store and store.chunks))
    return {"query": text, **decision.to_dict(),
            "extracted_values": extract_financial_values(text)}


def h_chat(payload, _query):
    _require_model()
    text = (payload or {}).get("query")
    if not text:
        raise ValueError('Provide {"query": "..."}')
    STATE["chat"].document_store = STATE.get("document_store")
    response = STATE["chat"].ask(text)
    return {"query": text, "answer": response.answer, "route": response.route,
            "source": response.source, "sources": response.sources,
            "detail": response.detail}


def h_inspect_tokens(payload, query):
    from app.backend.services.inspector import inspect_tokens
    text = (payload or {}).get("text") or query.get("text", ["What is EBITDA?"])[0]
    return inspect_tokens(text)


def h_inspect_architecture(_payload, _query):
    from app.backend.services.inspector import inspect_architecture
    return inspect_architecture(_require_model(), STATE["device"])


def h_inspect_moe(payload, query):
    from app.backend.services.inspector import inspect_moe_routing
    text = (payload or {}).get("text") or query.get("text", ["What is EBITDA?"])[0]
    layer = int((payload or {}).get("layer") or query.get("layer", [0])[0])
    return inspect_moe_routing(_require_model(), text, layer=layer, device=STATE["device"])


def h_inspect_forward(payload, query):
    from app.backend.services.inspector import inspect_forward
    text = (payload or {}).get("text") or query.get("text", ["What is EBITDA?"])[0]
    return inspect_forward(_require_model(), text, device=STATE["device"])


def h_rag_upload(payload, _query):
    from rag import DocumentStore
    path = (payload or {}).get("path")
    if not path:
        raise ValueError('Provide {"path": "<document path>"}')
    if not os.path.exists(path):
        raise ValueError(f"File not found: {path}")
    if STATE.get("document_store") is None:
        STATE["document_store"] = DocumentStore()
    info = STATE["document_store"].add_document(path)
    if STATE.get("chat") is not None:
        STATE["chat"].document_store = STATE["document_store"]
    if STATE.get("orchestrator") is not None:
        STATE["orchestrator"].document_store = STATE["document_store"]
    return {"added": info, "store": STATE["document_store"].stats()}


def h_rag_search(payload, query):
    store = STATE.get("document_store")
    if store is None or not store.chunks:
        return {"results": [], "note": "No document loaded"}
    text = (payload or {}).get("query") or query.get("query", [None])[0]
    if not text:
        raise ValueError('Provide {"query": "..."}')
    hits = store.search(text, top_k=int((payload or {}).get("top_k", 4)))
    return {"query": text, "results": [
        {"citation": h.citation, "score": round(h.score, 4), "text": h.chunk.text}
        for h in hits
    ]}


def h_rag_status(_payload, _query):
    store = STATE.get("document_store")
    return store.stats() if store else {"documents": [], "total_chunks": 0,
                                         "note": "No document loaded"}


def h_datasets(_payload, _query):
    from data_sources import list_entries
    from data_sources.manifest import read_manifest
    return {
        "registry": [
            {"name": e.name, "hf_id": e.hf_id, "subset": e.subset, "split": e.split,
             "category": e.category, "license": e.license, "source_url": e.source_url,
             "status": e.verification_status, "fields_used": e.fields_used, "notes": e.notes}
            for e in list_entries()
        ],
        "prepared": read_manifest()["datasets"],
    }


def h_dataset_stats(_payload, _query):
    from data_sources import compute_shard_stats
    shards_root = os.path.join(ROOT, "data", "shards")
    if not os.path.isdir(shards_root):
        return {"datasets": [], "note": "No shards prepared"}
    out = []
    for name in sorted(os.listdir(shards_root)):
        entry = {"name": name}
        for split in ("train", "validation"):
            entry[split] = compute_shard_stats(os.path.join(shards_root, name, split))
        out.append(entry)
    return {"datasets": out}


def h_training_status(_payload, _query):
    """Real training state read from checkpoint metadata on disk. This
    server does not run training, so it reports the last recorded run."""
    checkpoints = h_checkpoints(None, None)["checkpoints"]
    if not checkpoints:
        return {"status": "Not available", "note": "No checkpoints found"}
    latest = max(checkpoints, key=lambda c: (c["timestamp"] or ""))
    meta_path = os.path.join(ROOT, latest["path"].replace(".pt", ".json"))
    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
    return {
        "latest_checkpoint": latest["path"],
        "stage": meta.get("stage", "base"),
        "step": meta.get("step"),
        "train_loss": meta.get("train_loss"),
        "val_loss": meta.get("val_loss"),
        "best_val_loss": meta.get("best_val_loss"),
        "tokens_processed": meta.get("tokens_processed"),
        "dataset_config": meta.get("dataset_config", "Not available"),
        "runtime": meta.get("runtime", "Not available"),
        "timestamp": meta.get("timestamp"),
        "history": sorted(checkpoints, key=lambda c: (c["step"] or 0)),
    }


def h_evaluation(_payload, _query):
    """Return evaluation results previously measured and written to disk.
    Never synthesises scores."""
    results = {}
    for label, fname in (("base", "eval_base.json"), ("comparison", "compare.json")):
        path = os.path.join(ROOT, fname)
        if os.path.exists(path):
            with open(path) as f:
                results[label] = json.load(f)
    if not results:
        return {"status": "Not available",
                "note": "Run `python main.py evaluate --checkpoint <ckpt> --output eval.json`"}
    return results


def h_experiments(_payload, _query):
    from experiments import list_experiments
    return {"experiments": list_experiments()}


def h_colab_status(_payload, _query):
    import torch
    return {
        "colab": "Not available",
        "note": "No Google Colab/MCP runtime is connected to this process.",
        "local_device": "cuda" if torch.cuda.is_available() else "cpu",
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Not available",
    }


# ----------------------------------------------------------------------
# AI Capability Platform (registry, orchestrator, health, and the
# statistical capabilities that don't need a model or GPU)
# ----------------------------------------------------------------------

def h_ai_capabilities(_payload, _query):
    from ai_platform import REGISTRY, summary
    return {"capabilities": [c.to_dict() for c in REGISTRY.values()], "summary": summary()}


def h_ai_health(_payload, _query):
    from ai_platform.health import check_health
    return check_health()


def h_ai_orchestrate(payload, query):
    text = (payload or {}).get("query") or query.get("query", [None])[0]
    if not text:
        raise ValueError('Provide {"query": "..."}')
    if STATE.get("orchestrator") is None:
        from ai_platform import AIOrchestrator
        STATE["orchestrator"] = AIOrchestrator(
            model=STATE.get("model"), device=STATE.get("device", "cpu"),
            document_store=STATE.get("document_store"),
        )
    response = STATE["orchestrator"].handle(text)
    return response.to_dict()


def h_ai_forecast(payload, _query):
    from ai_platform.forecasting import forecast
    payload = payload or {}
    history = payload.get("history")
    if not history or len(history) < 2:
        raise ValueError('Provide {"history": [numbers...], "periods_ahead": N, '
                          '"method": "linear_trend"|"exponential_smoothing"}')
    result = forecast(
        [float(v) for v in history],
        periods_ahead=int(payload.get("periods_ahead", 3)),
        method=payload.get("method", "linear_trend"),
    )
    return result.to_dict()


def h_ai_anomaly(payload, _query):
    from ai_platform.anomaly import detect_transaction_anomalies, duplicate_invoices
    payload = payload or {}
    transactions = payload.get("transactions")
    if not transactions:
        raise ValueError('Provide {"transactions": [{"amount": N, ...}, ...]}')

    result = detect_transaction_anomalies(
        transactions, amount_field=payload.get("amount_field", "amount"),
        method=payload.get("method", "iqr"),
    )
    if payload.get("check_duplicates"):
        dups = duplicate_invoices(transactions, key_fields=payload.get(
            "duplicate_key_fields", ("vendor", "amount", "date")))
        result["duplicate_groups"] = [
            {"hash": g.hash, "indices": g.indices, "records": g.records} for g in dups
        ]
    return result


def h_ai_observability(_payload, query):
    from ai_platform.observability import read_recent, stats
    limit = int(query.get("limit", [50])[0])
    return {"stats": stats(), "recent": read_recent(limit=limit)}


def h_ai_security_check(payload, _query):
    from ai_platform.security import check_input
    text = (payload or {}).get("text")
    if not text:
        raise ValueError('Provide {"text": "..."}')
    return check_input(text).to_dict()


def h_ai_research(payload, query):
    from ai_platform.research import ResearchError, search
    text = (payload or {}).get("query") or query.get("query", [None])[0]
    if not text:
        raise ValueError('Provide {"query": "..."}')
    try:
        report = search(text, max_results=int((payload or {}).get("max_results", 5)))
    except ResearchError as e:
        return {"error": str(e)}
    return report.to_dict()


def h_ai_fraud(payload, _query):
    from ai_platform.fraud import load_model
    payload = payload or {}
    transactions = payload.get("transactions")
    if not transactions:
        raise ValueError('Provide {"transactions": [{"amt": N, "category": "...", '
                          '"trans_date_trans_time": "YYYY-MM-DD HH:MM:SS", "city_pop": N}, ...]}')
    model = load_model()
    scores = model.score(transactions)
    return {
        "scores": scores,
        "model_meta": {k: model.meta[k] for k in ("dataset_id", "dataset_license",
                                                    "dataset_note", "algorithm", "metrics")},
    }


def h_ai_code_execute(payload, _query):
    from ai_platform.code_sandbox import run_python
    payload = payload or {}
    code = payload.get("code")
    if not code:
        raise ValueError('Provide {"code": "..."}')
    result = run_python(code, timeout=float(payload.get("timeout", 5.0)))
    return {"stdout": result.stdout, "stderr": result.stderr,
            "returncode": result.returncode, "timed_out": result.timed_out,
            "rejected_reason": result.rejected_reason, "success": result.success}


def h_ai_database_query(payload, _query):
    from ai_platform.database_ai import answer_question
    payload = payload or {}
    question = payload.get("question")
    table = payload.get("table", "transactions")
    if not question:
        raise ValueError('Provide {"question": "...", "table": "..."}')
    return answer_question(question, table)


def h_ai_database_schema(_payload, _query):
    from ai_platform.database_ai import discover_schema
    return {"schema": discover_schema()}


def h_ai_kg_add(payload, _query):
    payload = payload or {}
    text = payload.get("text")
    if not text:
        raise ValueError('Provide {"text": "..."}')
    kg = _get_kg()
    triples = kg.add_text(text, source=payload.get("source"))
    return {"triples_added": [{"subject": s, "relation": r, "object": o} for s, r, o in triples],
            "stats": kg.stats()}


def h_ai_kg_relationships(payload, query):
    entity = (payload or {}).get("entity") or query.get("entity", [None])[0]
    if not entity:
        raise ValueError('Provide {"entity": "..."}')
    kg = _get_kg()
    return {"entity": entity, "relationships": kg.relationships(entity)}


def h_ai_kg_path(payload, query):
    payload = payload or {}
    a = payload.get("a") or query.get("a", [None])[0]
    b = payload.get("b") or query.get("b", [None])[0]
    if not a or not b:
        raise ValueError('Provide {"a": "...", "b": "..."}')
    kg = _get_kg()
    return {"a": a, "b": b, "path": kg.path_between(a, b)}


def h_ai_kg_all(_payload, _query):
    return _get_kg().to_dict()


def h_ai_language_detect(payload, query):
    from ai_platform.multilingual import detect_language
    text = (payload or {}).get("text") or query.get("text", [None])[0]
    if not text:
        raise ValueError('Provide {"text": "..."}')
    d = detect_language(text)
    return {"text": d.text, "language_code": d.language_code,
            "language_name": d.language_name, "confidence": d.confidence,
            "all_candidates": d.all_candidates}


def h_ai_speech_synthesize(payload, _query):
    from ai_platform.speech import SpeechError, synthesize
    payload = payload or {}
    text = payload.get("text")
    if not text:
        raise ValueError('Provide {"text": "..."}')
    try:
        path = synthesize(text, voice=payload.get("voice"))
    except SpeechError as e:
        return {"error": str(e)}
    return {"wav_path": path, "size_bytes": os.path.getsize(path)}


def h_ai_recommend(_payload, _query):
    from ai_platform.recommendation import get_recommendations
    recs = get_recommendations()
    return {"recommendations": [r.to_dict() for r in recs]}


def _get_kg():
    if STATE.get("knowledge_graph") is None:
        from ai_platform.knowledge_graph import KnowledgeGraph
        STATE["knowledge_graph"] = KnowledgeGraph()
    return STATE["knowledge_graph"]


ROUTES = {
    ("GET", "/api/health"): h_health,
    ("GET", "/api/model/status"): h_model_status,
    ("POST", "/api/model/load"): h_model_load,
    ("GET", "/api/checkpoints"): h_checkpoints,
    ("GET", "/api/calculate"): h_calculate,
    ("POST", "/api/calculate"): h_calculate,
    ("GET", "/api/route"): h_route,
    ("POST", "/api/route"): h_route,
    ("POST", "/api/chat"): h_chat,
    ("GET", "/api/inspect/tokens"): h_inspect_tokens,
    ("POST", "/api/inspect/tokens"): h_inspect_tokens,
    ("GET", "/api/inspect/architecture"): h_inspect_architecture,
    ("GET", "/api/inspect/moe"): h_inspect_moe,
    ("POST", "/api/inspect/moe"): h_inspect_moe,
    ("GET", "/api/inspect/forward"): h_inspect_forward,
    ("POST", "/api/inspect/forward"): h_inspect_forward,
    ("POST", "/api/rag/upload"): h_rag_upload,
    ("GET", "/api/rag/search"): h_rag_search,
    ("POST", "/api/rag/search"): h_rag_search,
    ("GET", "/api/rag/status"): h_rag_status,
    ("GET", "/api/datasets"): h_datasets,
    ("GET", "/api/datasets/stats"): h_dataset_stats,
    ("GET", "/api/training/status"): h_training_status,
    ("GET", "/api/evaluation"): h_evaluation,
    ("GET", "/api/experiments"): h_experiments,
    ("GET", "/api/colab/status"): h_colab_status,
    ("GET", "/api/ai/capabilities"): h_ai_capabilities,
    ("GET", "/api/ai/health"): h_ai_health,
    ("POST", "/api/ai/orchestrate"): h_ai_orchestrate,
    ("GET", "/api/ai/orchestrate"): h_ai_orchestrate,
    ("POST", "/api/ai/forecast"): h_ai_forecast,
    ("POST", "/api/ai/anomaly"): h_ai_anomaly,
    ("GET", "/api/ai/observability"): h_ai_observability,
    ("POST", "/api/ai/security-check"): h_ai_security_check,
    ("GET", "/api/ai/research"): h_ai_research,
    ("POST", "/api/ai/research"): h_ai_research,
    ("POST", "/api/ai/fraud"): h_ai_fraud,
    ("POST", "/api/ai/code/execute"): h_ai_code_execute,
    ("POST", "/api/ai/database"): h_ai_database_query,
    ("GET", "/api/ai/database/schema"): h_ai_database_schema,
    ("POST", "/api/ai/knowledge-graph/add"): h_ai_kg_add,
    ("GET", "/api/ai/knowledge-graph/relationships"): h_ai_kg_relationships,
    ("POST", "/api/ai/knowledge-graph/relationships"): h_ai_kg_relationships,
    ("GET", "/api/ai/knowledge-graph/path"): h_ai_kg_path,
    ("POST", "/api/ai/knowledge-graph/path"): h_ai_kg_path,
    ("GET", "/api/ai/knowledge-graph"): h_ai_kg_all,
    ("GET", "/api/ai/language/detect"): h_ai_language_detect,
    ("POST", "/api/ai/language/detect"): h_ai_language_detect,
    ("POST", "/api/ai/speech/synthesize"): h_ai_speech_synthesize,
    ("GET", "/api/ai/recommend"): h_ai_recommend,
}


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        pass  # keep the console clean; errors still surface in responses

    def _send(self, status, body):
        raw = json.dumps(body, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()
        self.wfile.write(raw)

    def do_OPTIONS(self):
        self._send(204, {})

    def _serve_frontend(self, path):
        rel = "index.html" if path in ("/", "/index.html") else path.lstrip("/")
        file_path = os.path.join(ROOT, "app", "frontend", rel)
        if not os.path.isfile(file_path):
            return False
        with open(file_path, "rb") as f:
            raw = f.read()
        ctype = ("text/html" if rel.endswith(".html")
                 else "text/css" if rel.endswith(".css")
                 else "application/javascript" if rel.endswith(".js")
                 else "application/octet-stream")
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)
        return True

    def _handle(self, method):
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"
        query = parse_qs(parsed.query)

        if method == "GET" and not path.startswith("/api"):
            if self._serve_frontend(path):
                return
            self._send(404, {"error": f"Not found: {path}"})
            return

        handler = ROUTES.get((method, path))
        if handler is None:
            self._send(404, {"error": f"No route for {method} {path}",
                              "routes": sorted(f"{m} {p}" for m, p in ROUTES)})
            return

        payload = None
        if method == "POST":
            length = int(self.headers.get("Content-Length") or 0)
            if length:
                try:
                    payload = json.loads(self.rfile.read(length).decode("utf-8"))
                except json.JSONDecodeError as e:
                    self._send(400, {"error": f"Invalid JSON body: {e}"})
                    return

        try:
            self._send(200, handler(payload, query))
        except ValueError as e:
            self._send(400, {"error": str(e)})
        except Exception as e:
            self._send(500, {"error": f"{type(e).__name__}: {e}",
                              "traceback": traceback.format_exc().splitlines()[-4:]})

    def do_GET(self):
        self._handle("GET")

    def do_POST(self):
        self._handle("POST")


def serve(host="127.0.0.1", port=8000, checkpoint=None):
    if checkpoint:
        print(f"Loading checkpoint {checkpoint} ...")
        print(f"  {load_checkpoint(checkpoint)}")
    server = ThreadingHTTPServer((host, port), Handler)
    print(f"Backend listening on http://{host}:{port}")
    print(f"  API:      http://{host}:{port}/api/health")
    print(f"  Frontend: http://{host}:{port}/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()
    serve(args.host, args.port, args.checkpoint)
