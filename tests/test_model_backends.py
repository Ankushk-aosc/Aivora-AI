"""Check /api/model/load's Hugging Face branch (run: python tests/test_model_backends.py): auth, path confinement, and
that a loaded backend actually answers chat requests.

HFBackend itself is stubbed - a real 1.5B model needs more RAM than this
machine has free - but everything around it is the real server code.
"""
import http.client
import json
import os
import sys
import tempfile
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TMP = tempfile.mkdtemp(prefix="aivora_backend_")
import ai_platform.auth as auth  # noqa: E402

auth.DB_PATH = os.path.join(TMP, "auth.db")
auth.SESSION_SECRET_PATH = os.path.join(TMP, ".session_secret")

from http.server import ThreadingHTTPServer  # noqa: E402

import app.backend.server as server  # noqa: E402
import app.backend.services.generation as generation  # noqa: E402

PASSED, FAILED = [], []


def check(name, ok, detail=""):
    (PASSED if ok else FAILED).append(name)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" - {detail}" if detail else ""))


class StubHF:
    """Stands in for HFBackend: same interface, no weights."""

    kind = "huggingface"

    def __init__(self, model_name, adapter_dir=None, device=None, dtype=None):
        self.model_name, self.adapter_dir = model_name, adapter_dir
        self.device = device or "cpu"
        self.prompts = []

    def generate(self, prompt, **kwargs):
        self.prompts.append(prompt)
        return "EBITDA is earnings before interest, taxes, depreciation and amortization."

    def describe(self):
        return {"kind": self.kind, "model": self.model_name, "adapter": self.adapter_dir,
                "device": self.device, "parameters": 1_500_000_000, "context_length": 32768}


generation.HFBackend = StubHF


def request(method, path, body=None, headers=None):
    conn = http.client.HTTPConnection("127.0.0.1", PORT, timeout=30)
    conn.request(method, path, body=json.dumps(body).encode() if body is not None else None,
                 headers={"Content-Type": "application/json", **(headers or {})})
    r = conn.getresponse()
    raw = r.read()
    try:
        return r.status, json.loads(raw)
    except ValueError:
        return r.status, raw


srv = ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
PORT = srv.server_address[1]
threading.Thread(target=srv.serve_forever, daemon=True).start()

auth.create_user("root", "admin-pass-123", role="admin")
admin = auth.issue_session_token("root", "admin")
auth.create_user("vic", "viewer-pass-1", role="viewer")
viewer = auth.issue_session_token("vic", "viewer")

HF = {"backend": "huggingface", "model": "Qwen/Qwen2.5-1.5B-Instruct"}

status, body = request("POST", "/api/model/load", HF)
check("loading a hub model without a login -> 401", status == 401, f"HTTP {status}")

status, body = request("POST", "/api/model/load", {**HF, "token": viewer})
check("viewer cannot load a hub model -> 403", status == 403, f"HTTP {status}")

status, body = request("POST", "/api/model/load", {**HF, "token": admin})
check("admin loads the hub model", status == 200 and body.get("loaded") is True, f"{status} {body}")
check("status reports the hub model", body.get("model") == HF["model"], str(body))

status, body = request("POST", "/api/model/load",
                       {**HF, "adapter": "../../etc", "token": admin})
check("adapter path outside checkpoints/ -> 403", status == 403, f"HTTP {status}")

# The whole point: chat must answer through the backend. Use a question the
# glossary does NOT cover, and a sentinel string only the stub can produce -
# "What is EBITDA?" would pass from the glossary even with a dead backend.
SENTINEL = "SENTINEL-FROM-BACKEND"
StubHF.generate = lambda self, prompt, **kw: f"{SENTINEL}: companies raise capital."
status, body = request("POST", "/api/chat", {"query": "Why do companies issue bonds?"})
check("chat answers via the loaded backend (sentinel present)",
      status == 200 and SENTINEL in str(body.get("answer", "")), str(body)[:200])

status, body = request("POST", "/api/chat",
                       {"query": "Calculate EBITDA margin for revenue 500 and EBITDA 100."})
check("calculator still handles numbers (not the model)",
      "20.00%" in str(body.get("answer", "")), str(body)[:160])

status, body = request("GET", "/api/model/status")
check("model status endpoint still works", status == 200, f"HTTP {status}")

srv.shutdown()
import shutil  # noqa: E402

shutil.rmtree(TMP, ignore_errors=True)
print(f"\n{len(PASSED)}/{len(PASSED) + len(FAILED)} passed")
sys.exit(1 if FAILED else 0)
