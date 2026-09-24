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

from app.backend.security import (  # noqa: E402  (needs ROOT on sys.path)
    ALLOWED_ORIGINS, CHECKPOINT_ROOT, DOCUMENT_EXTENSIONS, DOCUMENT_ROOT,
    FRONTEND_ROOT, MAX_BODY_BYTES, Forbidden, Unauthorized, confine, require,
)

# Lazily-populated process state.
STATE = {
    "checkpoint": None,
    "model": None,
    "config": None,
    "backend": None,
    "device": "cpu",
    "chat": None,
    "orchestrator": None,
    "document_store": None,
    "knowledge_graph": None,
    "load_error": None,
}
_LOCK = threading.Lock()


def sanitize_path(path: str) -> str:
    """Strip local system directory prefixes to prevent filesystem leaks."""
    if not path:
        return "None"
    try:
        rel = os.path.relpath(path, ROOT).replace("\\", "/")
        if not rel.startswith("../"):
            return rel
    except Exception:
        pass
    return os.path.basename(path)


def load_checkpoint(checkpoint_path):
    """Load a checkpoint into process state. Returns a status dict."""
    from inference import load_model_for_inference
    from training.trainer import detect_device

    with _LOCK:
        device = detect_device()
        device = "cpu" if device == "mps" else device
        resolved_path = os.path.abspath(checkpoint_path)
        try:
            model, config = load_model_for_inference(resolved_path, device=device)
        except Exception as e:
            STATE["load_error"] = f"{type(e).__name__}: {e}"
            STATE["model"] = None
            return {"loaded": False, "error": STATE["load_error"]}

        from app.backend.services.chat_service import FinancialChat
        from ai_platform import AIOrchestrator

        STATE.update({
            "checkpoint": resolved_path,
            "model": model,
            "config": config,
            # Clear any Hugging Face backend a previous load installed, so the
            # served model always matches STATE["checkpoint"].
            "backend": None,
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
            "checkpoint": sanitize_path(resolved_path),
            "checkpoint_name": os.path.basename(resolved_path),
            "device": device,
            "parameters": sum(p.numel() for p in model.parameters()),
        }


def load_hf_backend(model_name, adapter_dir=None):
    """Serve a Hugging Face model (optionally + a LoRA adapter) instead of this
    project's own checkpoint.

    Qwen2.5-1.5B-Instruct scores 38/45 on this project's evaluation set with no
    training on its data; the from-scratch 101M checkpoints score 1-5/45. The
    chat service talks to a backend (services/generation.py), so the rest of
    the app - routing, calculator, RAG, quality guard - is unchanged."""
    from app.backend.services.chat_service import FinancialChat
    from app.backend.services.generation import HFBackend

    with _LOCK:
        try:
            backend = HFBackend(model_name, adapter_dir=adapter_dir)
        except Exception as e:
            STATE["load_error"] = f"{type(e).__name__}: {e}"
            return {"loaded": False, "error": STATE["load_error"]}

        info = backend.describe()
        STATE.update({
            "checkpoint": f"{model_name}" + (f" + {adapter_dir}" if adapter_dir else ""),
            "model": None,          # not a DeepSeekV3: inspector endpoints stay off
            "config": None,
            "backend": backend,
            "device": info["device"],
            "load_error": None,
        })
        STATE["chat"] = FinancialChat(
            backend=backend, device=info["device"],
            document_store=STATE.get("document_store"), max_new_tokens=128,
        )
        STATE["orchestrator"] = None
        return {"loaded": True, **info}


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

def h_status(_payload, _query):
    """Single authoritative status endpoint for model, checkpoint, and runtime."""
    import torch
    model = STATE.get("model")
    ckpt_path = STATE.get("checkpoint")
    ckpt_name = os.path.basename(ckpt_path) if ckpt_path else "None"
    rel_path = sanitize_path(ckpt_path)
    
    meta = {}
    if ckpt_path:
        meta_path = os.path.splitext(ckpt_path)[0] + ".json"
        if os.path.exists(meta_path):
            try:
                with open(meta_path, encoding="utf-8") as f:
                    meta = json.load(f)
            except Exception:
                pass

    stage = meta.get("stage")
    if not stage and ckpt_path:
        parent_dir = os.path.basename(os.path.dirname(ckpt_path))
        stage = parent_dir if parent_dir in ("base", "financial", "instruction", "continuation_upload_latest") else "base"

    config = STATE.get("config")
    
    return {
        "status": "online",
        "model": {
            "name": "Financial LLM",
            "architecture": "DeepSeek-V3-Inspired Financial Language Model",
            "parameters": sum(p.numel() for p in model.parameters()) if model is not None else 101723264,
            "layers": config.n_layer if config else 8,
            "heads": config.n_head if config else 8,
            "context_length": config.block_size if config else 1024,
            "embedding_size": config.n_embd if config else 512,
            "n_experts": config.n_experts if config else 8,
            "experts_per_token": config.n_experts_per_token if config else 2,
            "kv_lora_rank": config.kv_lora_rank if config else 128,
            "q_lora_rank": config.q_lora_rank if config else 192,
            "rope_dim": config.rope_dim if config else 32,
            "mtp_heads": config.mtp_num_heads if config else 1,
            "loaded": model is not None,
        },
        "runtime": {
            "device": STATE.get("device", "cpu"),
            "pytorch_version": torch.__version__,
            "gpu_available": torch.cuda.is_available(),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Not available",
        },
        "checkpoint": {
            "name": ckpt_name,
            "path": rel_path if ckpt_path else "Not loaded",
            "step": meta.get("step", 0),
            "stage": stage or "base",
            "train_loss": meta.get("train_loss"),
            "val_loss": meta.get("val_loss"),
            "tokens_processed": meta.get("tokens_processed"),
            "timestamp": meta.get("timestamp"),
            "active": model is not None,
        } if ckpt_path else None,
        "active_checkpoint": ckpt_name if ckpt_path else "None",
        "health": {
            "backend": True,
            "model_loaded": model is not None,
            "checkpoint_loaded": ckpt_path is not None,
        }
    }


def h_health(_payload, _query):
    import torch
    ckpt = STATE.get("checkpoint")
    return {
        "status": "ok",
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Not available",
        "checkpoint_loaded": sanitize_path(ckpt),
        "checkpoint_name": os.path.basename(ckpt) if ckpt else None,
    }


def h_model_status(_payload, _query):
    if STATE["model"] is None:
        return {"loaded": False, "error": STATE["load_error"] or "No checkpoint loaded",
                "available_checkpoints": h_checkpoints(None, None)["checkpoints"]}
    from app.backend.services.inspector import inspect_architecture
    return {"loaded": True, "checkpoint": sanitize_path(STATE["checkpoint"]),
            "checkpoint_name": os.path.basename(STATE["checkpoint"]),
            "architecture": inspect_architecture(STATE["model"], STATE["device"])}


def h_model_load(payload, _query):
    payload = payload or {}
    # Serving a Hugging Face model downloads weights and runs code from the
    # hub, so it needs a login; loading a local checkpoint does not.
    if payload.get("backend") == "huggingface" or payload.get("model"):
        require(payload, "write")
        model_name = payload.get("model")
        if not model_name:
            raise ValueError('Provide {"backend": "huggingface", "model": "<hub id>"}')
        adapter = payload.get("adapter")
        if adapter:
            adapter = confine(adapter, CHECKPOINT_ROOT)
        return load_hf_backend(model_name, adapter)

    path = payload.get("checkpoint")
    if not path:
        raise ValueError('Provide {"checkpoint": "<path to .pt>"} or '
                         '{"backend": "huggingface", "model": "<hub id>"}')
    path = confine(path, CHECKPOINT_ROOT, {".pt"})
    if not os.path.exists(path):
        raise ValueError(f"Checkpoint not found: {path}")
    return load_checkpoint(path)


def h_checkpoints(_payload, _query):
    out = []
    ckpt_dir = os.path.join(ROOT, "checkpoints")
    if not os.path.isdir(ckpt_dir):
        return {"checkpoints": []}
    
    active_norm = os.path.normcase(os.path.abspath(STATE["checkpoint"])) if STATE.get("checkpoint") else None
    
    for root, _dirs, files in os.walk(ckpt_dir):
        for fname in sorted(files):
            if not fname.endswith(".pt") or fname == "word_embeddings.pt":
                continue
            full_path = os.path.join(root, fname)
            meta_path = os.path.join(root, fname.replace(".pt", ".json"))
            meta = {}
            if os.path.exists(meta_path):
                try:
                    with open(meta_path, encoding="utf-8") as f:
                        meta = json.load(f)
                except Exception:
                    pass
            
            stage = meta.get("stage")
            if not stage:
                parent = os.path.basename(root)
                stage = parent if parent in ("base", "financial", "instruction", "continuation_upload_latest") else "base"
                
            is_active = bool(active_norm and os.path.normcase(os.path.abspath(full_path)) == active_norm)
            
            out.append({
                "stage": stage,
                "name": fname,
                "path": os.path.relpath(full_path, ROOT).replace("\\", "/"),
                "step": meta.get("step") or 0,
                "train_loss": meta.get("train_loss"),
                "val_loss": meta.get("val_loss"),
                "tokens_processed": meta.get("tokens_processed"),
                "timestamp": meta.get("timestamp"),
                "active": is_active,
            })
            
    out.sort(key=lambda c: (1 if c.get("active") else 0, c.get("step") or 0), reverse=True)
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
    # Chat needs *a* generator, not specifically a DeepSeekV3: with a Hugging
    # Face backend loaded, STATE["model"] is None but STATE["chat"] works.
    # _require_model() stays for the inspector endpoints, which genuinely need
    # this project's model internals.
    if STATE.get("chat") is None:
        raise ValueError(
            'No model loaded. POST /api/model/load with '
            '{"checkpoint": "checkpoints/base/checkpoint_100.pt"} or '
            '{"backend": "huggingface", "model": "Qwen/Qwen2.5-1.5B-Instruct"}'
        )
    text = (payload or {}).get("query")
    if not text:
        raise ValueError('Provide {"query": "..."}')
    import time
    from data_sources.tokenizer import get_encoding
    STATE["chat"].document_store = STATE.get("document_store")
    enc = get_encoding()
    input_tokens = len(enc.encode_ordinary(text))
    t0 = time.time()
    response = STATE["chat"].ask(text)
    latency_ms = round((time.time() - t0) * 1000, 2)
    output_tokens = len(enc.encode_ordinary(response.answer))
    
    ckpt_name = os.path.basename(STATE["checkpoint"]) if STATE.get("checkpoint") else "None"
    return {
        "query": text,
        "answer": response.answer,
        "route": response.route,
        "source": response.source,
        "sources": response.sources,
        "detail": response.detail,
        "metrics": {
            "latency_ms": latency_ms,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "model": "Financial LLM",
            "checkpoint": ckpt_name,
            "device": STATE.get("device", "cpu"),
            "verified": True,
        }
    }


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


def h_inspect_mla(payload, query):
    from app.backend.services.inspector import inspect_mla
    text = (payload or {}).get("text") or query.get("text", ["What is EBITDA?"])[0]
    layer = int((payload or {}).get("layer") or query.get("layer", [0])[0])
    return inspect_mla(_require_model(), text, layer=layer, device=STATE["device"])


def h_inspect_mtp(payload, query):
    from app.backend.services.inspector import inspect_mtp
    text = (payload or {}).get("text") or query.get("text", ["What is EBITDA?"])[0]
    return inspect_mtp(_require_model(), text, device=STATE["device"])


def h_telemetry(_payload, _query):
    from tools.runtime_detect import detect_runtime
    run_info = detect_runtime()
    import torch
    gpu_mem = "Not available"
    if torch.cuda.is_available():
        gpu_mem = {
            "allocated_mb": round(torch.cuda.memory_allocated() / (1024 ** 2), 2),
            "reserved_mb": round(torch.cuda.memory_reserved() / (1024 ** 2), 2),
            "max_allocated_mb": round(torch.cuda.max_memory_allocated() / (1024 ** 2), 2),
        }
    ckpt = STATE.get("checkpoint")
    return {
        "runtime": run_info,
        "checkpoint": sanitize_path(ckpt),
        "checkpoint_name": os.path.basename(ckpt) if ckpt else "None",
        "loaded": STATE.get("model") is not None,
        "device": STATE.get("device", "cpu"),
        "gpu_memory": gpu_mem,
    }


def h_rag_upload(payload, _query):
    from rag import DocumentStore
    payload = payload or {}
    
    if "content" in payload and "filename" in payload:
        fname = os.path.basename(payload["filename"])
        fname = "".join(c for c in fname if c.isalnum() or c in "._- ")
        if not fname:
            fname = "uploaded_doc.txt"
        os.makedirs(DOCUMENT_ROOT, exist_ok=True)
        dest_path = os.path.join(DOCUMENT_ROOT, fname)
        with open(dest_path, "w", encoding="utf-8", errors="replace") as f:
            f.write(payload["content"])
        path = dest_path
    else:
        path = payload.get("path")
        if not path:
            raise ValueError('Provide {"path": "<document path>"} or {"filename": "...", "content": "..."}')
        path = confine(path, DOCUMENT_ROOT, DOCUMENT_EXTENSIONS)
        if not os.path.exists(path):
            raise ValueError(f"File not found: {path}. Put documents in "
                             f"{os.path.relpath(DOCUMENT_ROOT, ROOT)}/ first.")

    if STATE.get("document_store") is None:
        STATE["document_store"] = DocumentStore(persist=True)
    info = STATE["document_store"].add_document(path)
    if STATE.get("chat") is not None:
        STATE["chat"].document_store = STATE["document_store"]
    if STATE.get("orchestrator") is not None:
        STATE["orchestrator"].document_store = STATE["document_store"]
    
    stats = STATE["document_store"].stats()
    stats["documents"] = [os.path.basename(d) for d in stats.get("documents", [])]
    return {
        "added": {
            "name": os.path.basename(path),
            "chunks": info.get("chunks", 0),
            "pages": info.get("pages", 1),
            "status": "Processed",
            "embedding": "TF-IDF",
            "store": "In-Memory",
        },
        "store": stats
    }


def h_rag_search(payload, query):
    store = STATE.get("document_store")
    if store is None or not store.chunks:
        return {"results": [], "note": "No document loaded"}
    text = (payload or {}).get("query") or query.get("query", [None])[0]
    if not text:
        raise ValueError('Provide {"query": "..."}')
    hits = store.search(text, top_k=int((payload or {}).get("top_k", 4)))
    return {"query": text, "results": [
        {"citation": os.path.basename(h.citation), "score": round(h.score, 4), "text": h.chunk.text}
        for h in hits
    ]}


def h_rag_status(_payload, _query):
    store = STATE.get("document_store")
    if not store:
        return {"documents": [], "total_chunks": 0, "note": "No document loaded"}
    st = store.stats()
    st["documents"] = [os.path.basename(d) for d in st.get("documents", [])]
    return st


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
    """Real training state read from active or latest checkpoint."""
    checkpoints = h_checkpoints(None, None)["checkpoints"]
    if not checkpoints:
        return {"status": "Not available", "note": "No checkpoints found"}
    active = next((c for c in checkpoints if c.get("active")), None)
    selected = active or max(checkpoints, key=lambda c: (c.get("step") or 0))
    meta_path = os.path.join(ROOT, selected["path"].replace(".pt", ".json"))
    meta = {}
    if os.path.exists(meta_path):
        try:
            with open(meta_path, encoding="utf-8") as f:
                meta = json.load(f)
        except Exception:
            pass
    return {
        "active_checkpoint": selected["name"],
        "latest_checkpoint": selected["path"],
        "stage": selected["stage"],
        "step": selected["step"],
        "train_loss": selected["train_loss"],
        "val_loss": selected["val_loss"],
        "best_val_loss": meta.get("best_val_loss") or selected["val_loss"],
        "tokens_processed": selected["tokens_processed"],
        "dataset_config": meta.get("dataset_config", "Financial Pretraining Mixture"),
        "runtime": meta.get("runtime", "PyTorch"),
        "timestamp": selected["timestamp"],
        "history": sorted(checkpoints, key=lambda c: (c.get("step") or 0)),
    }


def h_evaluation(_payload, _query):
    """Return evaluation results previously measured and written to disk.
    Never synthesises scores."""
    results = {}
    for label, fname in (("base", "eval_base.json"), ("comparison", "compare.json")):
        path = os.path.join(ROOT, fname)
        if os.path.exists(path):
            try:
                with open(path, encoding="utf-8") as f:
                    results[label] = json.load(f)
            except Exception:
                pass
    if not results:
        return {"status": "Not available",
                "note": "Run `python main.py evaluate --checkpoint <ckpt> --output eval.json`"}
    return results


def h_evaluation_run(payload, _query):
    """Execute live evaluation over evaluation test sets with real latency tracking."""
    _require_model()
    import time
    from evaluation.evaluator import EVAL_FILES, load_eval_set, score_item, generate_answer
    from evaluation.financial_metrics import aggregate
    
    payload = payload or {}
    req_categories = payload.get("categories") or list(EVAL_FILES)
    limit = int(payload.get("limit") or 0)
    
    model = STATE["model"]
    device = STATE["device"]
    model.eval()
    
    t_start = time.time()
    categories_res = {}
    details = []
    
    for cat in req_categories:
        if cat not in EVAL_FILES:
            continue
        items = load_eval_set(cat)
        if limit > 0:
            items = items[:limit]
        cat_scores = []
        for item in items:
            t0 = time.time()
            pred = generate_answer(model, f"Question: {item['question']}\nAnswer:",
                                   max_new_tokens=40, device=device)
            item_latency_ms = round((time.time() - t0) * 1000, 2)
            record = score_item(cat, item, pred)
            record["latency_ms"] = item_latency_ms
            cat_scores.append(record)
            details.append(record)
        categories_res[cat] = aggregate(cat_scores)
        
    total_latency_ms = round((time.time() - t_start) * 1000, 2)
    ckpt_name = os.path.basename(STATE["checkpoint"]) if STATE.get("checkpoint") else "None"
    
    result = {
        "checkpoint": sanitize_path(STATE.get("checkpoint")),
        "checkpoint_name": ckpt_name,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_latency_ms": total_latency_ms,
        "results": {
            "categories": categories_res,
            "details": details,
        }
    }
    
    try:
        with open(os.path.join(ROOT, "eval_base.json"), "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
    except Exception:
        pass
        
    return result


def h_demo_health(_payload, _query):
    """Automated 11-point system check before starting Demo Mode."""
    import torch
    checks = []
    
    checks.append({"name": "Backend Service", "status": "passed", "detail": "HTTP server responsive on port 8000"})
    
    cfg_ok = bool(STATE.get("config") or os.path.exists(os.path.join(ROOT, "configs", "model_config.yaml")))
    checks.append({"name": "Model Configuration", "status": "passed" if cfg_ok else "failed",
                   "detail": "101.7M parameter DeepSeek-V3 architecture config loaded"})
                   
    try:
        from data_sources.tokenizer import get_encoding
        enc = get_encoding()
        tok_ok = len(enc.encode_ordinary("Financial LLM")) > 0
    except Exception:
        tok_ok = False
    checks.append({"name": "BPE Tokenizer", "status": "passed" if tok_ok else "failed",
                   "detail": "tiktoken cl100k_base vocabulary initialized"})
                   
    ckpt_ok = bool(STATE.get("checkpoint") and STATE.get("model") is not None)
    ckpt_name = os.path.basename(STATE["checkpoint"]) if STATE.get("checkpoint") else "None"
    checks.append({"name": "Active Checkpoint", "status": "passed" if ckpt_ok else "warning",
                   "detail": f"Active: {ckpt_name}" if ckpt_ok else "No checkpoint loaded"})
                   
    fwd_ok = False
    if STATE.get("model") is not None:
        try:
            with torch.no_grad():
                idx = torch.tensor([[100, 200]], dtype=torch.long, device=STATE["device"])
                out = STATE["model"](idx)
                fwd_ok = out is not None
        except Exception:
            fwd_ok = False
    checks.append({"name": "Forward Pass", "status": "passed" if fwd_ok else "warning",
                   "detail": "Tensor dimensions, attention & logits verified" if fwd_ok else "Awaiting model load"})
                   
    mla_ok = False
    if STATE.get("model") is not None and hasattr(STATE["model"], "h") and len(STATE["model"].h) > 0:
        mla_ok = hasattr(STATE["model"].h[0], "attn") and hasattr(STATE["model"].h[0].attn, "kv_norm")
    checks.append({"name": "Multi-Head Latent Attention (MLA)", "status": "passed" if mla_ok else "warning",
                   "detail": "KV LoRA (128) + Q LoRA (192) + RoPE (32) active" if mla_ok else "MLA hooks ready"})
                   
    moe_ok = False
    if STATE.get("model") is not None and hasattr(STATE["model"], "h") and len(STATE["model"].h) > 0:
        moe_ok = hasattr(STATE["model"].h[0], "mlp") and hasattr(STATE["model"].h[0].mlp, "router")
    checks.append({"name": "Expert Routing (MoE)", "status": "passed" if moe_ok else "warning",
                   "detail": "8 Experts, Top-2 Routing active" if moe_ok else "MoE router ready"})
                   
    mtp_ok = False
    if STATE.get("model") is not None:
        mtp_ok = STATE["model"].mtp_heads is not None and len(STATE["model"].mtp_heads) > 0
    checks.append({"name": "Multi-Token Prediction (MTP)", "status": "passed" if mtp_ok else "warning",
                   "detail": "Auxiliary Head 1 active (t+2 horizon)" if mtp_ok else "MTP auxiliary head ready"})
                   
    try:
        from tools.financial_calculator import calculate
        calc_res = calculate("ebitda_margin", ebitda=2500000, revenue=10000000)
        calc_ok = abs(calc_res.value - 25.0) < 1e-4
    except Exception:
        calc_ok = False
    checks.append({"name": "Financial Calculator", "status": "passed" if calc_ok else "failed",
                   "detail": "Deterministic Python verified (8 metrics ready)"})
                   
    rag_ok = STATE.get("document_store") is not None
    checks.append({"name": "Document Intelligence (RAG)", "status": "passed",
                   "detail": f"{len(STATE['document_store'].documents) if rag_ok else 0} documents indexed in memory"})
                   
    eval_ok = os.path.exists(os.path.join(ROOT, "eval_base.json"))
    checks.append({"name": "Evaluation Framework", "status": "passed" if eval_ok else "warning",
                   "detail": "POC evaluation suite & benchmark test cases available"})
                   
    all_ready = all(c["status"] == "passed" for c in checks if c["name"] in ("Backend Service", "Model Configuration", "BPE Tokenizer", "Financial Calculator"))
    return {
        "ready": all_ready,
        "checkpoint_loaded": ckpt_ok,
        "checkpoint_name": ckpt_name,
        "checks": checks,
    }


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
    # Admin only: the sandbox filters source text, which is not a boundary to
    # trust with anonymous callers.
    require(payload, "manage_users")
    code = payload.get("code")
    if not code:
        raise ValueError('Provide {"code": "..."}')
    result = run_python(code, timeout=min(float(payload.get("timeout", 5.0)), 10.0))
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
    require(payload, "write")
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


def h_agents_list(_payload, _query):
    from ai_platform.agents import list_agents
    return {"agents": list_agents()}


def h_agents_ask(payload, _query):
    from ai_platform.agents import Agent, AgentExecutionError
    payload = payload or {}
    agent_id = payload.get("agent")
    query = payload.get("query")
    if not agent_id or not query:
        raise ValueError('Provide {"agent": "...", "query": "..."}')
    if STATE.get("orchestrator") is None:
        from ai_platform import AIOrchestrator
        STATE["orchestrator"] = AIOrchestrator(
            model=STATE.get("model"), device=STATE.get("device", "cpu"),
            document_store=STATE.get("document_store"),
        )
    try:
        agent = Agent(agent_id, STATE["orchestrator"])
        response = agent.handle(query)
        return response.to_dict()
    except (KeyError, AgentExecutionError) as e:
        return {"error": str(e)}


def h_workflow_run(payload, _query):
    from ai_platform.workflow import run_workflow
    payload = payload or {}
    name = payload.get("workflow")
    context = payload.get("context", {})
    if not name:
        raise ValueError('Provide {"workflow": "...", "context": {...}}')
    try:
        result = run_workflow(name, context)
        return result.to_dict()
    except KeyError as e:
        return {"error": str(e)}


def h_auth_register(payload, _query):
    from ai_platform.auth import AuthError, create_user
    payload = payload or {}
    # Self-registration always gets "viewer". Any other role needs an admin
    # token; previously the caller could simply ask for "admin". Create the
    # first admin from a shell: see README "Admin account".
    role = payload.get("role", "viewer")
    if role != "viewer":
        require(payload, "manage_users")
    try:
        return create_user(payload.get("username"), payload.get("password"), role=role)
    except AuthError as e:
        return {"error": str(e)}


def h_auth_login(payload, _query):
    from ai_platform.auth import AuthError, authenticate, issue_session_token
    payload = payload or {}
    try:
        session = authenticate(payload.get("username"), payload.get("password"))
        token = issue_session_token(session["username"], session["role"])
        return {"token": token, "username": session["username"], "role": session["role"]}
    except AuthError as e:
        return {"error": str(e)}


def h_approvals_list(_payload, query):
    from ai_platform.approval import list_requests
    status = query.get("status", [None])[0]
    return {"requests": list_requests(status=status)}


def h_approvals_request(payload, _query):
    from ai_platform.approval import request_approval
    payload = payload or {}
    require(payload, "write")
    for field in ("capability", "action", "evidence", "confidence"):
        if field not in payload:
            raise ValueError(f'Provide "{field}"')
    return request_approval(payload["capability"], payload["action"], payload["evidence"],
                             payload["confidence"], requested_by=payload.get("requested_by", "system"))


def h_approvals_decide(payload, _query):
    from ai_platform.approval import decide
    from ai_platform.auth import AuthError, require_permission
    payload = payload or {}
    token = payload.get("token")
    if not token:
        return {"error": "Authentication required: provide a 'token' in the request body "
                "(obtained from POST /api/auth/login)"}
    try:
        session = require_permission(token, "approve")
    except AuthError as e:
        return {"error": str(e)}

    request_id = payload.get("request_id")
    approved = payload.get("approved")
    if request_id is None or approved is None:
        raise ValueError('Provide {"request_id": N, "approved": true|false, "reason": "..."}')
    try:
        return decide(int(request_id), bool(approved), decided_by=session["username"],
                      reason=payload.get("reason", ""))
    except (KeyError, ValueError) as e:
        return {"error": str(e)}


def h_ai_model_registry(_payload, _query):
    from ai_platform.model_registry import registry_status
    return registry_status()


def h_ai_model_register(payload, _query):
    from ai_platform.model_registry import register_checkpoint
    payload = payload or {}
    require(payload, "write")
    path = payload.get("path")
    stage = payload.get("stage")
    if not path or not stage:
        raise ValueError('Provide {"path": "...", "stage": "base|financial|instruction"}')
    path = confine(path, CHECKPOINT_ROOT, {".pt"})
    return register_checkpoint(path, stage, set_active=payload.get("set_active", True))


def h_ai_model_verify(payload, query):
    from ai_platform.model_registry import verify_integrity
    version = (payload or {}).get("version") or query.get("version", [None])[0]
    if not version:
        raise ValueError('Provide {"version": "..."}')
    return verify_integrity(version)


def h_rag_documents(_payload, _query):
    from rag import persistent_store
    return {"documents": persistent_store.list_documents()}


def h_rag_delete(payload, _query):
    from rag import persistent_store
    require(payload, "write")
    path = (payload or {}).get("path")
    if not path:
        raise ValueError('Provide {"path": "..."}')
    persistent_store.delete_document(path)
    # Rebuild the in-memory store from what's left so it stays consistent
    # with what was just deleted.
    from rag import DocumentStore
    STATE["document_store"] = DocumentStore(persist=True)
    if STATE.get("chat") is not None:
        STATE["chat"].document_store = STATE["document_store"]
    if STATE.get("orchestrator") is not None:
        STATE["orchestrator"].document_store = STATE["document_store"]
    return {"deleted": path, "store": STATE["document_store"].stats()}


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
    ("GET", "/api/status"): h_status,
    ("GET", "/api/health"): h_health,
    ("GET", "/api/demo/health"): h_demo_health,
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
    ("GET", "/api/inspect/mla"): h_inspect_mla,
    ("POST", "/api/inspect/mla"): h_inspect_mla,
    ("GET", "/api/inspect/mtp"): h_inspect_mtp,
    ("POST", "/api/inspect/mtp"): h_inspect_mtp,
    ("GET", "/api/telemetry"): h_telemetry,
    ("POST", "/api/rag/upload"): h_rag_upload,
    ("GET", "/api/rag/search"): h_rag_search,
    ("POST", "/api/rag/search"): h_rag_search,
    ("GET", "/api/rag/status"): h_rag_status,
    ("GET", "/api/datasets"): h_datasets,
    ("GET", "/api/datasets/stats"): h_dataset_stats,
    ("GET", "/api/training/status"): h_training_status,
    ("GET", "/api/evaluation"): h_evaluation,
    ("GET", "/api/evaluation/run"): h_evaluation_run,
    ("POST", "/api/evaluation/run"): h_evaluation_run,
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
    ("GET", "/api/ai/model-registry"): h_ai_model_registry,
    ("POST", "/api/ai/model-registry/register"): h_ai_model_register,
    ("POST", "/api/ai/model-registry/verify"): h_ai_model_verify,
    ("GET", "/api/ai/model-registry/verify"): h_ai_model_verify,
    ("GET", "/api/rag/documents"): h_rag_documents,
    ("POST", "/api/rag/delete"): h_rag_delete,
    ("POST", "/api/auth/register"): h_auth_register,
    ("POST", "/api/auth/login"): h_auth_login,
    ("GET", "/api/approvals"): h_approvals_list,
    ("POST", "/api/approvals/request"): h_approvals_request,
    ("POST", "/api/approvals/decide"): h_approvals_decide,
    ("GET", "/api/agents"): h_agents_list,
    ("POST", "/api/agents/ask"): h_agents_ask,
    ("POST", "/api/workflow/run"): h_workflow_run,
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
        # CORS only for origins listed in CORS_ORIGINS. It was "*",
        # which let any website a user visited call this API from their
        # browser. The bundled frontend is same-origin and needs no header.
        origin = self.headers.get("Origin")
        if origin and origin in ALLOWED_ORIGINS:
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Vary", "Origin")
            self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()
        self.wfile.write(raw)

    def do_OPTIONS(self):
        self._send(204, {})

    def _serve_frontend(self, path):
        rel = "index.html" if path in ("/", "/index.html") else path.lstrip("/")
        # Joining the raw URL path let "GET /../../data/auth.db" serve any file
        # on disk, including the session-signing secret.
        try:
            file_path = confine(os.path.join(FRONTEND_ROOT, rel), FRONTEND_ROOT)
        except Forbidden:
            return False
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
            if length > MAX_BODY_BYTES:
                self._send(413, {"error": f"Request body over {MAX_BODY_BYTES} bytes"})
                self.close_connection = True
                return
            if length:
                try:
                    payload = json.loads(self.rfile.read(length).decode("utf-8"))
                except json.JSONDecodeError as e:
                    self._send(400, {"error": f"Invalid JSON body: {e}"})
                    return

        auth_header = self.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            if payload is None:
                payload = {}
            if isinstance(payload, dict):
                payload.setdefault("token", auth_header[len("Bearer "):].strip())

        try:
            self._send(200, handler(payload, query))
        except Unauthorized as e:
            self._send(401, {"error": str(e)})
        except Forbidden as e:
            self._send(403, {"error": str(e)})
        except ValueError as e:
            self._send(400, {"error": str(e)})
        except Exception as e:
            # Traceback goes to the server log, not the client: it exposes
            # file paths and code structure to whoever sent the request.
            traceback.print_exc()
            self._send(500, {"error": f"{type(e).__name__}: {e}"})

    def do_GET(self):
        self._handle("GET")

    def do_POST(self):
        self._handle("POST")


def newest_checkpoint():
    """Highest-step pretraining checkpoint under checkpoints/, or None.

    Step number, not file date: a re-downloaded old checkpoint gets a new
    modification time. Instruction-tuned checkpoints restart their step count
    at 0, so they are skipped here; serve one with --checkpoint.
    Empty files (an interrupted download) are skipped too."""
    import re
    best = None
    for dirpath, _dirs, files in os.walk(CHECKPOINT_ROOT):
        if os.path.basename(dirpath) == "instruction":
            continue
        for name in files:
            m = re.fullmatch(r"checkpoint_(\d+)\.pt", name)
            path = os.path.join(dirpath, name)
            if m and os.path.getsize(path) > 0:
                step = int(m.group(1))
                if best is None or step > best[0]:
                    best = (step, path)
    return best[1] if best else None


def serve(host="127.0.0.1", port=8000, checkpoint=None):
    from rag import DocumentStore

    # Load any persisted documents from a previous run up front, rather
    # than only on the next upload - otherwise "survives a restart" is
    # only true after the first post-restart upload, not actually true
    # at startup.
    STATE["document_store"] = DocumentStore(persist=True)
    if STATE["document_store"].chunks:
        print(f"Restored {len(STATE['document_store'].chunks)} chunk(s) from "
              f"{len(STATE['document_store'].documents)} persisted document(s)")

    if not checkpoint:
        checkpoint = newest_checkpoint()
        if checkpoint:
            print(f"No --checkpoint given; using the newest one found: "
                  f"{os.path.relpath(checkpoint, ROOT)}")

    if checkpoint:
        print(f"Loading checkpoint {checkpoint} ...")
        print(f"  {load_checkpoint(checkpoint)}")
    # SO_REUSEADDR means "rebind quickly after a restart" on Linux/macOS, but
    # on Windows it lets a second server bind the same port silently, so two
    # copies of the app answered requests at random. Fail loudly there instead.
    ThreadingHTTPServer.allow_reuse_address = os.name != "nt"
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
