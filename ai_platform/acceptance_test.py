"""Final end-to-end acceptance test (continuous-engineering priority #30).

Runs the full chain against a REAL live backend over real HTTP - not
in-process function calls, so this exercises the actual serving layer,
routing, and JSON (de)serialization a real client would go through:

    USER -> AUTH -> AI ORCHESTRATOR -> AGENT -> DOCUMENT -> RAG
         -> LLM -> TOOL -> WORKFLOW -> HUMAN APPROVAL -> AUDIT LOG -> RESPONSE

TENANT is intentionally not a stage here: multi-tenancy is registered
NOT_IMPLEMENTED in this project (single-process, single-tenant design),
and faking a tenant stage would misrepresent what was actually tested.
Each stage reports PASS/FAIL with the real evidence, not an assumption;
one failed stage does not abort the rest, so the report shows exactly
how much of the chain genuinely works, not just the first failure.
"""

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field


@dataclass
class StageResult:
    stage: str
    passed: bool
    detail: str
    evidence: dict = field(default_factory=dict)


def _request(base_url, method, path, payload=None, timeout=60):
    url = f"{base_url}{path}"
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, method=method,
                                  headers={"Content-Type": "application/json"} if data else {})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def run_acceptance_test(base_url: str = "http://127.0.0.1:8123", document_path: str = None) -> list:
    stages = []
    import secrets
    suffix = secrets.token_hex(4)
    admin_user = f"e2e_admin_{suffix}"
    viewer_user = f"e2e_viewer_{suffix}"

    # ---------------- AUTH ----------------
    try:
        _request(base_url, "POST", "/api/auth/register",
                 {"username": admin_user, "password": "e2e-test-password-1", "role": "admin"})
        _request(base_url, "POST", "/api/auth/register",
                 {"username": viewer_user, "password": "e2e-test-password-2", "role": "viewer"})
        admin_login = _request(base_url, "POST", "/api/auth/login",
                                {"username": admin_user, "password": "e2e-test-password-1"})
        viewer_login = _request(base_url, "POST", "/api/auth/login",
                                 {"username": viewer_user, "password": "e2e-test-password-2"})
        admin_token = admin_login["token"]
        viewer_token = viewer_login["token"]
        stages.append(StageResult("AUTH", True,
                                   "Registered + logged in admin and viewer users, got real session tokens",
                                   {"admin_role": admin_login["role"], "viewer_role": viewer_login["role"]}))
    except Exception as e:
        stages.append(StageResult("AUTH", False, f"{type(e).__name__}: {e}"))
        return stages  # nothing downstream can be meaningfully attributed without auth

    # ---------------- AI ORCHESTRATOR + AGENT ----------------
    try:
        agent_resp = _request(base_url, "POST", "/api/agents/ask",
                               {"agent": "financial_analyst",
                                "query": "Revenue is 500 and EBITDA is 100. Calculate EBITDA margin."})
        ok = agent_resp.get("capability") == "CALCULATOR" and "20.00%" in agent_resp.get("answer", "")
        stages.append(StageResult("AI_ORCHESTRATOR_AGENT", ok,
                                   f"financial_analyst routed to {agent_resp.get('capability')}",
                                   {"answer_excerpt": agent_resp.get("answer", "")[:80]}))
    except Exception as e:
        stages.append(StageResult("AI_ORCHESTRATOR_AGENT", False, f"{type(e).__name__}: {e}"))

    # ---------------- Scope enforcement (agent correctly refuses out-of-scope) ----------------
    try:
        refused = _request(base_url, "POST", "/api/agents/ask",
                            {"agent": "financial_analyst", "query": "Detect fraud in these transactions"})
        ok = "error" in refused and "not permitted" in refused["error"]
        stages.append(StageResult("AGENT_SCOPE_ENFORCEMENT", ok,
                                   "financial_analyst correctly refused a FRAUD_AI-routed query",
                                   {"error": refused.get("error", "")[:100]}))
    except Exception as e:
        stages.append(StageResult("AGENT_SCOPE_ENFORCEMENT", False, f"{type(e).__name__}: {e}"))

    # ---------------- DOCUMENT + RAG ----------------
    if document_path:
        try:
            upload = _request(base_url, "POST", "/api/rag/upload", {"path": document_path})
            search = _request(base_url, "POST", "/api/rag/search",
                               {"query": "What was EBITDA?", "top_k": 2})
            ok = upload.get("added", {}).get("chunks_added", 0) > 0 and len(search.get("results", [])) > 0
            stages.append(StageResult("DOCUMENT_RAG", ok,
                                       f"Uploaded and retrieved from {document_path}",
                                       {"chunks_added": upload.get("added", {}).get("chunks_added"),
                                        "results_found": len(search.get("results", []))}))
        except Exception as e:
            stages.append(StageResult("DOCUMENT_RAG", False, f"{type(e).__name__}: {e}"))
    else:
        stages.append(StageResult("DOCUMENT_RAG", False, "Skipped - no document_path provided"))

    # ---------------- WORKFLOW (Agent -> Tool -> Verification chain) ----------------
    workflow_ok = False
    request_id = None
    if document_path:
        try:
            wf = _request(base_url, "POST", "/api/workflow/run", {
                "workflow": "invoice_review",
                "context": {"document_path": document_path, "invoice_category": "grocery_pos",
                            "invoice_timestamp": "2026-01-01 03:15:00", "invoice_city_pop": 500},
            })
            workflow_ok = wf.get("status") in ("completed", "awaiting_approval")
            request_id = wf.get("context", {}).get("approval_request_id")
            stages.append(StageResult("WORKFLOW", workflow_ok,
                                       f"invoice_review workflow status: {wf.get('status')}",
                                       {"steps_completed": wf.get("steps_completed"),
                                        "approval_request_id": request_id}))
        except Exception as e:
            stages.append(StageResult("WORKFLOW", False, f"{type(e).__name__}: {e}"))
    else:
        stages.append(StageResult("WORKFLOW", False, "Skipped - no document_path provided"))

    # ---------------- HUMAN APPROVAL (RBAC-enforced) ----------------
    if request_id:
        try:
            denied = _request(base_url, "POST", "/api/approvals/decide",
                               {"token": viewer_token, "request_id": request_id, "approved": True})
            denied_ok = "error" in denied and "lacks permission" in denied["error"]

            approved = _request(base_url, "POST", "/api/approvals/decide",
                                 {"token": admin_token, "request_id": request_id,
                                  "approved": True, "reason": "E2E acceptance test"})
            approved_ok = approved.get("status") == "approved"

            ok = denied_ok and approved_ok
            stages.append(StageResult("HUMAN_APPROVAL", ok,
                                       "Viewer correctly denied; admin correctly approved",
                                       {"viewer_denied": denied_ok, "admin_approved": approved_ok}))
        except Exception as e:
            stages.append(StageResult("HUMAN_APPROVAL", False, f"{type(e).__name__}: {e}"))
    else:
        stages.append(StageResult("HUMAN_APPROVAL", False,
                                   "Skipped - no approval request was generated by the workflow "
                                   "(the transaction did not cross the risk threshold this run)"))

    # ---------------- AUDIT LOG (immutability check) ----------------
    if request_id:
        try:
            reapprove = _request(base_url, "POST", "/api/approvals/decide",
                                  {"token": admin_token, "request_id": request_id, "approved": False})
            ok = "error" in reapprove and "already" in reapprove["error"]
            stages.append(StageResult("AUDIT_LOG_IMMUTABILITY", ok,
                                       "Re-deciding an already-decided request was correctly rejected",
                                       {"error": reapprove.get("error", "")}))
        except Exception as e:
            stages.append(StageResult("AUDIT_LOG_IMMUTABILITY", False, f"{type(e).__name__}: {e}"))
    else:
        stages.append(StageResult("AUDIT_LOG_IMMUTABILITY", False, "Skipped - no approval request"))

    # ---------------- MODEL_REGISTRY integrity ----------------
    try:
        reg = _request(base_url, "GET", "/api/ai/model-registry")
        active = reg.get("active", {})
        verified_all = True
        details = {}
        for stage_name, version in active.items():
            v = _request(base_url, "POST", "/api/ai/model-registry/verify", {"version": version})
            details[version] = v.get("valid")
            verified_all = verified_all and v.get("valid", False)
        stages.append(StageResult("MODEL_REGISTRY_INTEGRITY", verified_all and bool(active),
                                   f"Verified {len(active)} active checkpoint(s)", details))
    except Exception as e:
        stages.append(StageResult("MODEL_REGISTRY_INTEGRITY", False, f"{type(e).__name__}: {e}"))

    # ---------------- OBSERVABILITY ----------------
    try:
        obs = _request(base_url, "GET", "/api/ai/observability?limit=5")
        ok = obs.get("stats", {}).get("total_requests", 0) > 0
        stages.append(StageResult("OBSERVABILITY", ok,
                                   f"{obs.get('stats', {}).get('total_requests')} total requests logged",
                                   obs.get("stats", {})))
    except Exception as e:
        stages.append(StageResult("OBSERVABILITY", False, f"{type(e).__name__}: {e}"))

    return stages


def print_report(stages: list):
    print("=" * 70)
    print("FINAL END-TO-END ACCEPTANCE TEST")
    print("=" * 70)
    for s in stages:
        mark = "PASS" if s.passed else "FAIL"
        print(f"[{mark}] {s.stage}: {s.detail}")
        if s.evidence:
            print(f"       evidence: {s.evidence}")
    print("-" * 70)
    passed = sum(1 for s in stages if s.passed)
    print(f"{passed}/{len(stages)} stages passed")
    print("=" * 70)
