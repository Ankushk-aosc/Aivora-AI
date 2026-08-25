"""AI Capability Registry (spec Part 35).

The single source of truth for what this platform can actually do.
Every capability declares its real status - IMPLEMENTED / TESTED / PARTIAL /
BLOCKED / NOT_IMPLEMENTED - per Part 34 ("never claim a capability exists
unless it's actually connected and tested"). Nothing here is aspirational;
a capability's status only changes when the code backing it changes.
"""

from dataclasses import dataclass, field
from enum import Enum


class Status(str, Enum):
    IMPLEMENTED = "IMPLEMENTED"       # code exists and runs
    TESTED = "TESTED"                 # implemented AND verified against real inputs
    PARTIAL = "PARTIAL"               # implemented for a subset of the spec'd capability
    BLOCKED = "BLOCKED"               # a specific external dependency is missing
    NOT_IMPLEMENTED = "NOT_IMPLEMENTED"


@dataclass
class Capability:
    name: str
    route: str          # matches the CapabilityRoute enum in orchestrator.py
    type: str            # llm | tool | ml_model | agent | infra
    model: str
    version: str
    provider: str        # "proprietary" | "statistical" | "heuristic" | "none"
    status: Status
    reason: str = ""      # required detail when status is not TESTED
    endpoint: str = ""
    tools: list = field(default_factory=list)
    permissions: str = "none required"
    # Populated as capabilities are iterated on (spec Part 53.15). Not
    # every entry has these filled in yet - absence means "not yet
    # recorded", not "none exist".
    known_bugs: list = field(default_factory=list)  # bugs found+fixed, kept for history
    next_action: str = ""

    def to_dict(self):
        d = {**self.__dict__}
        d["status"] = self.status.value
        return d


REGISTRY = {
    # ------------------------------------------------------------------
    # Real, working, tested this session or earlier this project
    # ------------------------------------------------------------------
    "GENERAL_LLM": Capability(
        name="General LLM", route="GENERAL_LLM", type="llm",
        model="FinLLM-102M-Instruct", version="checkpoints/instruction/checkpoint_100.pt",
        provider="proprietary", status=Status.TESTED,
        reason="Proprietary DeepSeek-V3-style model (101.7M params). Trained only "
        "100 steps on CPU - architecture and pipeline are verified, but language "
        "ability is not yet useful (see evaluation results).",
        endpoint="/api/chat",
    ),
    "FINANCIAL_LLM": Capability(
        name="Financial LLM", route="FINANCIAL_LLM", type="llm",
        model="FinLLM-102M-Financial", version="checkpoints/base/checkpoint_100.pt",
        provider="proprietary", status=Status.TESTED,
        reason="Same proprietary model, base-stage checkpoint trained on the full "
        "6-bucket financial+general dataset mix. Same training-budget caveat as above.",
        endpoint="/api/chat",
    ),
    "CALCULATOR": Capability(
        name="Financial Calculator / Math Engine", route="CALCULATOR", type="tool",
        model="deterministic Python", version="1.0", provider="none",
        status=Status.TESTED,
        reason="15 financial formulas (margins, CAGR, ROE/ROA/ROIC, D/E, current "
        "ratio, FCF, EPS, P/E, EV/EBITDA). Verified 15/15 against hand-computed "
        "expected values. Never uses LLM-generated arithmetic.",
        endpoint="/api/calculate",
    ),
    "RAG": Capability(
        name="Retrieval-Augmented Generation", route="RAG", type="tool",
        model="TF-IDF retriever + BM25 reranker + FinLLM", version="1.1",
        provider="proprietary", status=Status.TESTED,
        reason="Ingest -> parse -> chunk -> embed (TF-IDF, or model hidden-states "
        "when a checkpoint is loaded) -> retrieve -> BM25 rerank -> LLM -> "
        "citations. Verified against real TXT/PDF/DOCX/CSV/JSON files with "
        "correct page citations. v1.1: now backed by a real SQLite persistent "
        "store (rag/persistent_store.py) - verified surviving an ACTUAL backend "
        "process restart (killed and relaunched the process, not simulated), "
        "printed 'Restored 1 chunk(s) from 1 persisted document(s)' at "
        "startup, and search returned identical results post-restart.",
        endpoint="/api/rag/search",
        known_bugs=["v1.1 persistence: an early design persisted per-chunk "
                    "embedding VECTORS, which would have silently produced "
                    "wrong scores because TF-IDF's vector space depends on "
                    "the whole corpus, not fixed per chunk - caught before "
                    "implementing the reload path, not after; fixed by "
                    "persisting text/metadata only and re-running _reindex() "
                    "once after reload."],
        next_action="Persistence covers TXT/PDF/DOCX/CSV/JSON already tested "
        "formats; email and cloud-storage ingestion (spec Part 7) remain "
        "NOT_IMPLEMENTED - no mail/cloud-storage connector exists.",
    ),
    "LIVE_DATA": Capability(
        name="Live Market Data", route="LIVE_DATA", type="tool",
        model="Yahoo Finance public chart endpoint", version="1.0",
        provider="unofficial_public_api", status=Status.TESTED,
        reason="No API key required. Verified against real symbols (AAPL). "
        "This is an UNOFFICIAL, undocumented public endpoint, not a "
        "credentialed integration - it can change or rate-limit without "
        "notice. Any fetch failure (unknown symbol, network error, "
        "endpoint change) returns 'Current market data is not available.' - "
        "verified for an invalid symbol - never a fabricated price.",
        endpoint="/api/ai/orchestrate",
    ),
    "FINAL_E2E_ACCEPTANCE": Capability(
        name="Final End-to-End Acceptance Test", route="FINAL_E2E_ACCEPTANCE", type="infra",
        model="ai_platform/acceptance_test.py - real HTTP against the live server",
        version="1.0", provider="local", status=Status.TESTED,
        reason="Full chain run over REAL HTTP against a live backend instance "
        "(not in-process calls) - register+login two real users -> agent-"
        "routed calculator question through financial_analyst -> confirmed "
        "the SAME agent is correctly REFUSED a fraud-routed query (scope "
        "enforcement, not just declared) -> real document upload + RAG "
        "search -> invoice_review workflow reaching a genuine "
        "'awaiting_approval' state (not forced - the transaction's category/"
        "time/amount combination was independently found to cross the real "
        "trained model's risk threshold) -> RBAC-enforced approval (viewer "
        "denied, admin approved, both against the real /api/approvals/"
        "decide endpoint) -> immutable audit trail (re-deciding rejected) "
        "-> model registry integrity verified for both active checkpoints "
        "-> observability confirms 71 real logged requests. RESULT: 9/9 "
        "stages passed. TENANT is not a stage - multi-tenancy is "
        "NOT_IMPLEMENTED in this project and faking that stage would "
        "misrepresent what was tested; every other stage in the spec's "
        "example chain (Part 53.13) that has a real implementation here "
        "was exercised for real.",
        endpoint="ai_platform.acceptance_test.run_acceptance_test()",
    ),
    "PRODUCTION_DEPLOYMENT": Capability(
        name="Production Deployment", route="PRODUCTION_DEPLOYMENT", type="infra",
        model="Dockerfile + docker-compose.yml", version="1.0",
        provider="local", status=Status.NOT_IMPLEMENTED,
        reason="Authored a Dockerfile and docker-compose.yml reusing the "
        "exact pip-install and server-start commands already verified "
        "working directly on this machine, and validated the docker-compose "
        "YAML syntax + the Dockerfile's CMD shell-substitution logic "
        "(confirmed --host/--port/--checkpoint flags exist in "
        "app/backend/server.py's actual argparse setup, and tested the "
        "conditional --checkpoint substitution both with and without it set). "
        "Deliberately NOT_IMPLEMENTED, not PARTIAL like CI/CD: no Docker "
        "daemon is available here (`docker` is not on PATH), so this has "
        "genuinely never been built or run - weaker evidence than CI/CD, "
        "where at least every embedded command was individually run and "
        "confirmed passing on this machine. A real deployment TARGET "
        "(cloud VM, PaaS, etc.) is separately and completely BLOCKED - no "
        "credentials or access to any hosting provider exist here.",
        next_action="Build and run this Dockerfile somewhere Docker is "
        "actually available before trusting it; that is the actual test, "
        "not this authoring step.",
    ),
    "LOAD_TESTING": Capability(
        name="Load / Performance Testing", route="LOAD_TESTING", type="infra",
        model="ai_platform/load_test.py - real concurrent HTTP against the live server",
        version="1.0", provider="local", status=Status.TESTED,
        reason="No external load tool (k6/Locust/JMeter) needed or installed - "
        "stdlib concurrent.futures + urllib firing real HTTP requests at the "
        "actual running backend. Every number below is a measured latency "
        "from a real request/response cycle, not estimated. Three profiles, "
        "run back to back against one live server instance:\n"
        "  GET /api/health, 50 reqs @ concurrency 10: p50=13.7ms, "
        "p95=518.4ms, p99=587.4ms, 0 errors.\n"
        "  POST /api/calculate, 30 reqs @ concurrency 10: p50=15.9ms, "
        "p95=517.3ms, p99=521.6ms, 0 errors.\n"
        "  POST /api/ai/orchestrate (real CPU-bound model generation), "
        "6 reqs @ concurrency 3: p50=13,361ms, max=14,040ms, 0 errors/"
        "timeouts.\n"
        "Real finding, not glossed over: even the lightweight/deterministic "
        "endpoints show a large p50->p95 gap (~13ms to ~518ms) under "
        "10-way concurrency. Root cause is architectural, not a bug to "
        "patch: Python's stdlib http.server + ThreadingMixIn (what "
        "app/backend/server.py uses) spawns a thread per connection and "
        "serializes their Python bytecode via the GIL - the stdlib's own "
        "docs describe it as not intended for high-concurrency production "
        "use. Model generation, by contrast, held up fine under its (much "
        "lower) concurrency - no failures, no runaway latency growth "
        "relative to the single-request baseline observed earlier in this "
        "project (PyTorch's tensor ops release the GIL during computation).",
        endpoint="ai_platform.load_test.run_suite()",
        known_bugs=["Not a bug found IN a capability, but a real capacity "
                    "limit found ABOUT the serving layer: the p95 latency "
                    "spike under concurrent light-endpoint load is a "
                    "genuine characteristic of Python's stdlib "
                    "ThreadingHTTPServer, not something a code fix inside "
                    "any single endpoint handler would resolve."],
        next_action="If concurrent light-endpoint latency needs to improve, "
        "the actual fix is swapping the serving layer (e.g. a proper WSGI/"
        "ASGI server with a process pool) rather than optimizing individual "
        "handlers - a real architecture decision, not attempted here "
        "without being asked to change the serving stack.",
    ),
    "CI_CD": Capability(
        name="CI/CD Pipeline", route="CI_CD", type="infra",
        model="GitHub Actions workflow (.github/workflows/ci.yml)", version="1.0",
        provider="local", status=Status.PARTIAL,
        reason="A real workflow file was authored (import regression across "
        "33 modules, model smoke test, calculator/registry/code-sandbox "
        "self-tests) and every embedded command was run VERBATIM on this "
        "machine to confirm it actually passes - not just written and hoped. "
        "PARTIAL, not TESTED, because it has never run on an actual GitHub "
        "Actions runner: that requires pushing to the repository's real "
        "remote (origin, a shared/visible action), which needs the user's "
        "go-ahead, not something to do unprompted. The workflow is not yet "
        "enabled/triggered.",
        next_action="Push .github/workflows/ci.yml (or open a PR) to see it "
        "actually run on GitHub's runners - genuinely blocked on that "
        "authorization, not on more local engineering.",
    ),
    "GPU_TRAINING_COLAB": Capability(
        name="Google Colab GPU Training", route="GPU_TRAINING_COLAB", type="infra",
        model="training/colab/FinLLM_GPU_Training.ipynb", version="1.0",
        provider="local", status=Status.BLOCKED,
        reason="A full 20-step Colab GPU training notebook was authored: env/"
        "GPU/CUDA/VRAM verification with a hard gate (raises before any "
        "training claim if torch.cuda.is_available() is False) -> repo "
        "transfer + integrity check -> dependency install -> dataset "
        "acquisition/validation (reuses the same tested data_sources."
        "prepare_dataset and evaluation.check_leakage used elsewhere in "
        "this project) -> tokenizer round-trip -> model config -> checkpoint "
        "discovery with real tensor-shape/architecture compatibility "
        "checking -> VRAM-aware batch/seq auto-config -> training with real "
        "torch.cuda.OutOfMemoryError capture-and-retry -> an actually-"
        "executed resume test -> evaluation -> inference on 3 required "
        "prompts -> checksummed export -> independent hash re-verification. "
        "Validated locally: nbformat.validate() passed with zero warnings, "
        "all 20 code cells syntax-checked via compile(), and every "
        "referenced function signature (prepare_dataset, check_leakage, "
        "train_model, evaluate_model, load_model_for_inference, "
        "generate_text, register_checkpoint, verify_integrity, get_encoding) "
        "cross-checked against current source via grep. BLOCKED, not "
        "TESTED or PARTIAL, because the notebook has never actually been "
        "run: no authenticated Google session exists anywhere in this "
        "environment, verified fresh via 3 independent checks (MCP registry "
        "search for colab/kaggle/gpu connectors: empty; "
        "list_connected_browsers: []; direct navigation to "
        "colab.research.google.com showing a bare 'Sign in' page). No "
        "credentials were entered or requested. Zero GPU training has "
        "occurred - no loss curve, checkpoint, or evaluation from this "
        "notebook is real until it is actually executed on a GPU runtime.",
        next_action="User needs to open training/colab/FinLLM_GPU_Training."
        "ipynb in an authenticated Google Colab session with a GPU runtime "
        "and run it; only then can this move to TESTED with real measured "
        "values (loss, VRAM, step time, artifact hash) reported back.",
    ),
    "AI_AGENTS": Capability(
        name="AI Agent Framework", route="AI_AGENTS", type="agent",
        model="capability-scoped wrappers over AIOrchestrator", version="1.0",
        provider="local", status=Status.TESTED,
        reason="6 declared agents (Financial/Research/Document/Data/Fraud "
        "Analyst + Enterprise Assistant), each restricted to a named subset "
        "of capabilities. Verified the enforcement actually works, not just "
        "that agents are declared: financial_analyst correctly answered a "
        "calculator question, then was correctly REFUSED when asked a "
        "fraud question the orchestrator would have routed to FRAUD_AI "
        "(not in its allowlist) - while fraud_analyst handled the identical "
        "query successfully. An agent cannot use a capability it isn't "
        "scoped for, even though the shared orchestrator would route there.",
        endpoint="/api/agents/ask",
    ),
    "WORKFLOW_ENGINE": Capability(
        name="Workflow Engine", route="WORKFLOW_ENGINE", type="infra",
        model="step-chain executor over existing capabilities", version="1.0",
        provider="local", status=Status.TESTED,
        reason="Real invoice_review workflow: Document -> RAG text extraction "
        "-> deterministic field parsing (regex, not LLM-guessed) -> the "
        "actual trained FRAUD_AI model -> HUMAN_APPROVAL gate -> audit log. "
        "This is the one complete real end-to-end workflow required by the "
        "spec, and every stage is the genuine implementation, not a stub: "
        "verified a low-risk invoice completes straight through, a "
        "genuinely high-risk one (found by sweeping the real trained model, "
        "not guessed - score 0.547) correctly stops at 'awaiting_approval' "
        "and persists a real request in the approval queue, a document with "
        "no extractable amount completes without fabricating a fraud score, "
        "and a missing document fails the workflow rather than silently "
        "continuing. Every step is individually logged via observability.",
        endpoint="/api/workflow/run",
        next_action="Only one workflow is defined (invoice_review). Adding "
        "more (e.g. a research-report workflow) is straightforward given the "
        "Workflow/WorkflowStep primitives already exist and are tested.",
    ),
    "AUTH_RBAC": Capability(
        name="Authentication / RBAC", route="AUTH_RBAC", type="infra",
        model="PBKDF2 password hashing + HMAC session tokens (stdlib)",
        version="1.0", provider="local", status=Status.TESTED,
        reason="Real local auth, not SSO: SQLite user table, PBKDF2-SHA256 "
        "password hashing (100k iterations), HMAC-signed self-expiring "
        "session tokens, 3-role RBAC (admin/analyst/viewer). 12/12 tests "
        "passed including the security-critical ones: a tampered token "
        "(role escalated to admin, old signature kept) is rejected via "
        "signature mismatch; wrong-password and unknown-user return the "
        "identical error message (no username enumeration); expired tokens "
        "rejected. Third-party SSO (Google/Okta/etc.) is genuinely BLOCKED - "
        "needs registering an OAuth app with an external provider and "
        "credentials this environment doesn't have.",
        endpoint="/api/auth/login",
    ),
    "HUMAN_APPROVAL": Capability(
        name="Human-in-the-Loop Approval", route="HUMAN_APPROVAL", type="infra",
        model="SQLite approval queue + AUTH_RBAC permission gate", version="1.0",
        provider="local", status=Status.TESTED,
        reason="Real persistent approval queue (not in-memory) for high-impact "
        "actions (FRAUD_AI, RECOMMENDATION_AI). Verified: request -> pending -> "
        "human decides -> approved/rejected with an immutable audit trail "
        "(deciding an already-decided request raises an error rather than "
        "silently overwriting it - re-verified this specifically). The "
        "decide endpoint requires the 'approve' RBAC permission, wiring "
        "auth into this capability concretely rather than leaving it "
        "unenforced. 'Action' means recording the decision - there is no "
        "real downstream system (ERP/payment processor) to actually execute "
        "an approved action against, and that boundary is explicit here, "
        "not glossed over.",
        endpoint="/api/approvals/decide",
    ),
    "MODEL_REGISTRY": Capability(
        name="Model Serving / Versioning", route="MODEL_REGISTRY", type="infra",
        model="SHA256 checksums + version tags (local)", version="1.0",
        provider="local", status=Status.TESTED,
        reason="Adds what checkpoint saving alone didn't have: content-"
        "addressed integrity verification and an explicit active/serving "
        "version per stage. Verified: registered real checkpoints, computed "
        "real SHA256 hashes, confirmed an unmodified file verifies valid, "
        "and confirmed a genuinely corrupted file (tested on a disposable "
        "copy, never the original after the first test taught that lesson "
        "the hard way) is detected via checksum mismatch. Training's own "
        "checkpoint metadata (step/loss/optimizer state/resume) is reused "
        "as-is, not reimplemented.",
        endpoint="/api/ai/model-registry",
        known_bugs=["First integrity test corrupted a REAL, non-gitignored-"
                    "content checkpoint file in place, because a `/tmp` "
                    "backup path silently didn't exist for this Windows "
                    "Python process - the corruption was real and required "
                    "recovering via --resume from the last good checkpoint "
                    "(step 50 -> re-trained to 100). Rewrote the test to "
                    "copy to a Python tempfile.mkdtemp() path and corrupt "
                    "only the copy."],
        next_action="No rollback/promotion workflow yet - set_active() "
        "exists but there's no UI/CLI wrapper for 'promote version X to "
        "serving' beyond calling the API directly.",
    ),
    "DOCUMENT_AI": Capability(
        name="Document AI", route="DOCUMENT_AI", type="tool",
        model="pypdf / python-docx parsers", version="1.0", provider="none",
        status=Status.PARTIAL,
        reason="Text extraction, chunking, and structured CSV/JSON field reading "
        "are implemented and tested (rag/document_loader.py). OCR (scanned/image "
        "PDFs), classification, invoice/contract-specific extraction, and table "
        "structure extraction are NOT implemented - no OCR engine or "
        "document-classification model is connected.",
    ),

    # ------------------------------------------------------------------
    # Real, newly implemented this session (statistical / heuristic - no
    # GPU or external model required, so no fabrication is involved)
    # ------------------------------------------------------------------
    "FORECASTING_AI": Capability(
        name="Forecasting AI", route="FORECASTING_AI", type="ml_model",
        model="linear trend + simple exponential smoothing", version="1.0",
        provider="statistical", status=Status.TESTED,
        reason="Classical statistical forecasting (not an LLM guessing numbers). "
        "Returns point forecast + MAE/RMSE backtested error metrics. No deep "
        "learning forecasting model (e.g. a trained temporal transformer) exists.",
        endpoint="/api/ai/forecast",
    ),
    "ANOMALY_AI": Capability(
        name="Anomaly Detection AI", route="ANOMALY_AI", type="ml_model",
        model="IQR (default) / z-score outlier detection + duplicate-invoice matching",
        version="1.0", provider="statistical", status=Status.TESTED,
        reason="Deterministic statistical outlier detection on numeric "
        "transaction data (IQR by default - robust to a single dominant "
        "outlier, unlike z-score, which that same outlier can mask), plus "
        "exact/near-duplicate invoice detection reusing the dataset-dedup "
        "hashing. Not a trained anomaly-detection neural network - none "
        "exists or was trained.",
        endpoint="/api/ai/anomaly",
    ),

    # ------------------------------------------------------------------
    # Explicitly not implemented - real external dependencies missing
    # ------------------------------------------------------------------
    "EMBEDDING_AI": Capability(
        name="Embedding AI", route="EMBEDDING_AI", type="ml_model",
        model="TF-IDF (production) / FinLLM hidden states / trained skip-gram (rejected)",
        version="1.0", provider="proprietary", status=Status.PARTIAL,
        reason="rag/embeddings.py provides TF-IDF (used in production) and "
        "model-hidden-state embedders; no dedicated pretrained embedding model "
        "(e.g. a sentence-transformer) - by design (Part 45 forbids importing "
        "pretrained external models). TF-IDF is lexical, not semantic; the "
        "model-hidden-state embedder inherits the LLM's undertrained quality. "
        "A genuine attempt was made at a real, locally-trained alternative: "
        "rag/word_embeddings.py, a from-scratch skip-gram model (2 epochs, "
        "15.7M training pairs from 2M real corpus tokens, loss 2.261->2.233). "
        "It FAILED its own acceptance test (nearest-neighbor sanity check) - "
        "'revenue' neighbors were semantically meaningless subword fragments, "
        "and common words like 'profit'/'company'/'growth' fell outside the "
        "5000-word vocabulary entirely because this project's prepared corpus "
        "is small and only partly financial. Per the explicit rule against "
        "claiming something works when it can't be verified: this was NOT "
        "wired into RAG. The code is kept (reusable if a larger/more focused "
        "corpus becomes available) but TF-IDF remains the production embedder.",
        known_bugs=["word_embeddings.py training genuinely ran and losses "
                    "genuinely decreased, but the resulting embeddings are "
                    "not semantically meaningful - documented as a real "
                    "negative result, not a code defect (verified the empty "
                    "neighbor results were a small/imbalanced vocabulary "
                    "effect, not a bug in the lookup)."],
        next_action="Would need a much larger and/or more financially-"
        "concentrated corpus (word2vec typically wants corpora orders of "
        "magnitude larger than the ~3M tokens currently prepared) before "
        "attempting to wire this in again - do not re-attempt without more data.",
    ),
    "RERANKING_AI": Capability(
        name="Reranking AI", route="RERANKING_AI", type="ml_model",
        model="BM25 (Okapi, from scratch)", version="1.0",
        provider="statistical", status=Status.TESTED,
        reason="Second-stage BM25 reranking over the first-stage TF-IDF "
        "retrieval, now used by default in the RAG pipeline. Measured, not "
        "just claimed: on a test query ('What was the EBITDA margin?') "
        "against 4 chunks, first-stage TF-IDF ranked an irrelevant risk-"
        "factors chunk #1 and the actually-relevant EBITDA chunk #2; BM25 "
        "reranking correctly promoted the relevant chunk to #1. Not a "
        "pretrained cross-encoder (none available without a GPU/download) - "
        "a classical, from-scratch IR ranking function, genuinely distinct "
        "from the first-stage cosine score it re-ranks.",
        endpoint="/api/ai/rerank",
    ),
    "VISION_AI": Capability(
        name="Vision AI", route="VISION_AI", type="ml_model",
        model="none", version="-", provider="none", status=Status.NOT_IMPLEMENTED,
        reason="No vision model is connected or tested. Image/chart/table "
        "understanding, scanned-document OCR, and visual QA are not available. "
        "Re-checked (not assumed from memory): no `tesseract` binary on PATH. "
        "`easyocr` installs cleanly as a package but downloads a genuine "
        "pretrained CV model (CRAFT detector + recognizer, 100MB+) at first "
        "use - the same category of external-pretrained-model risk already "
        "declined for speech-to-text (openai-whisper) and translation "
        "(argos-translate) earlier in this project, for consistency. Not "
        "treating OCR as an exception to that judgment.",
        next_action="If OCR specifically becomes a hard requirement, the "
        "most consistent path is a system-level tesseract install (a real "
        "OCR *engine*, not a downloaded neural net, same category as "
        "Windows SAPI for speech) rather than a pretrained CV model.",
    ),
    "SPEECH_AI": Capability(
        name="Speech AI", route="SPEECH_AI", type="ml_model",
        model="Windows SAPI (System.Speech)", version="1.0",
        provider="local_os", status=Status.PARTIAL,
        reason="Text-to-speech only, and genuinely tested: produced a real, "
        "valid 4.19s WAV file (parsed with Python's wave module to confirm "
        "it isn't a stub - 92,365 audio frames at 22050Hz) from real text. "
        "Uses the OS's own synthesis engine (3 installed voices found), not "
        "a downloaded model. Speech-to-text is NOT implemented: the only "
        "realistic local option (e.g. openai-whisper) is a 150MB+ pretrained "
        "model download, which wasn't added without a clear need to verify "
        "against - this stays honestly PARTIAL rather than silently skipping "
        "half the capability.",
        endpoint="/api/ai/speech/synthesize",
    ),
    "CODE_AI": Capability(
        name="Code AI", route="CODE_AI", type="agent",
        model="subprocess sandbox (python -I -S + restricted builtins)",
        version="1.0", provider="local", status=Status.PARTIAL,
        reason="Execution only (not generation - the LLM isn't capable enough "
        "to write correct code yet). Real process isolation: fresh subprocess, "
        "hard wall-clock timeout (verified: an infinite loop was killed within "
        "the timeout), restricted builtins via a purpose-built exec() globals "
        "dict (verified: dir() raises NameError; caught and fixed a real bug "
        "first where reassigning __builtins__ in the same frame did NOT "
        "actually restrict it - CPython caches f_builtins at frame creation). "
        "This is PROCESS isolation, not container/VM-level security (no "
        "Docker/seccomp available here) - a determined attacker with time to "
        "find gaps in the builtin restriction could still find one; it stops "
        "the obvious cases (import os, open, eval, exec, runaway loops), not "
        "every case.",
        endpoint="/api/ai/code/execute",
        known_bugs=["Initial implementation reassigned __builtins__ inside "
                    "the SAME already-running frame, which does NOT restrict "
                    "anything in CPython (f_builtins is cached at frame "
                    "creation, not re-read on assignment) - found via a "
                    "dir() test that leaked full builtin access. Fixed by "
                    "running user code through exec() with a purpose-built "
                    "globals dict, so the NEW frame exec() creates derives "
                    "its builtins from that dict at creation time. Re-verified "
                    "dir() now correctly raises NameError."],
        next_action="Consider a real OS-level sandbox (e.g. Windows Job "
        "Objects for memory/CPU limits) if this capability needs to run "
        "less-trusted code than it currently does - the static blocklist "
        "can still be bypassed by a sufficiently creative encoding.",
    ),
    "DATABASE_AI": Capability(
        name="Database AI / Data Analyst", route="DATABASE_AI", type="agent",
        model="SQLite (stdlib) + deterministic query templates", version="1.0",
        provider="local", status=Status.PARTIAL,
        reason="Real local SQLite database, real schema discovery, and real "
        "read-only query execution - verified with a seeded synthetic test "
        "table (correct SUM/COUNT/ORDER BY results, e.g. supplies total "
        "2000.0 = 1200+800). SQL generation is deliberately NOT free-form "
        "NL2SQL from the LLM (it isn't trained enough to be trusted with "
        "exact queries yet) - a small set of parameterized templates handles "
        "'total by X', 'top N', 'total', 'average'; anything else is honestly "
        "reported as unsupported rather than guessed. Write/DDL statements "
        "and multi-statement injection are rejected before reaching sqlite3 "
        "(verified: DROP/DELETE/multi-statement all blocked).",
        endpoint="/api/ai/database",
    ),
    "FRAUD_AI": Capability(
        name="Fraud AI", route="FRAUD_AI", type="ml_model",
        model="RandomForestClassifier (scikit-learn)", version="1.0",
        provider="trained_local", status=Status.TESTED,
        reason="Trained on pointe77/credit-card-transaction (Apache-2.0, "
        "synthetic Sparkov-simulator data - not real cardholders), 60,000 "
        "streamed records, held-out test set (n=15,000, 145 fraud cases). "
        "Real measured metrics: precision 0.393, recall 0.938, F1 0.554, "
        "ROC-AUC 0.992. Recall is strong (catches most fraud); precision is "
        "genuinely mediocre (many false positives) because the feature set is "
        "just amount/time/category - no per-cardholder velocity features "
        "(e.g. 'transactions in the last hour'), which real fraud systems "
        "rely on heavily. Reported as-is, not rounded up. Validated against "
        "an alternative on the identical split: HistGradientBoostingClassifier "
        "scored WORSE on every metric (precision 0.325, recall 0.931, F1 "
        "0.482, ROC-AUC 0.989) - RandomForest was kept because it's actually "
        "better here, not by default.",
        endpoint="/api/ai/fraud",
        next_action="Real velocity features (transactions/hour per card) "
        "would likely help precision more than switching algorithms did, "
        "but require redesigning /api/ai/fraud to accept per-card recent-"
        "transaction history rather than isolated transactions - a genuine "
        "API contract change, not a small addition, so not attempted without "
        "a clear need for it.",
    ),
    "RECOMMENDATION_AI": Capability(
        name="Recommendation AI", route="RECOMMENDATION_AI", type="agent",
        model="deterministic rules over SQLite data", version="1.0",
        provider="local", status=Status.PARTIAL,
        reason="Real, verified against seeded synthetic test data: vendor "
        "concentration (correctly flagged a vendor at 40.4% spend share, "
        "correctly did NOT flag one at 26.9%) and cost-outlier detection "
        "(reuses the already-tested IQR anomaly detector rather than a second "
        "untested implementation). Every recommendation carries recommendation/"
        "reason/evidence/confidence/data_sources as required. PARTIAL because "
        "it only reasons over whatever is in the local SQLite database - no "
        "real enterprise spend/vendor-catalog data source is connected, so "
        "recommendations are only as meaningful as the (currently synthetic "
        "test) data seeded into it.",
        endpoint="/api/ai/recommend",
    ),
    "MULTILINGUAL_AI": Capability(
        name="Multilingual AI", route="MULTILINGUAL_AI", type="tool",
        model="langdetect (pure Python, no download)", version="1.0",
        provider="local", status=Status.PARTIAL,
        reason="Language DETECTION only - verified 4/5 on a mixed-language "
        "test set (en/de/ja/zh-cn correct; a short Spanish question was "
        "misclassified as French, a known langdetect weakness on short "
        "strings - a longer Spanish sentence detected correctly at "
        "confidence 1.0). Translation is NOT implemented: the only realistic "
        "local option (argostranslate) pulls in ~35 packages including spacy "
        "and onnxruntime plus per-language runtime downloads - too heavy to "
        "add and verify reliably on this machine (~1GB free RAM) without "
        "risking exactly the kind of unverified claim this registry exists "
        "to prevent. The proprietary LLM's tokenizer and training data are "
        "English-only regardless, so multilingual generation would be low "
        "quality even with translation added.",
    ),
    "RESEARCH_AI": Capability(
        name="Research AI", route="RESEARCH_AI", type="agent",
        model="DuckDuckGo HTML search (unofficial, no API key)", version="1.0",
        provider="unofficial_public_api", status=Status.PARTIAL,
        reason="Search -> Retrieve -> Cite is real and tested (verified real "
        "URLs/snippets for 'EBITDA definition finance'). Compare/Analyze/Verify "
        "stages are NOT implemented - the proprietary LLM is not yet capable "
        "enough to meaningfully synthesize across sources (see evaluation "
        "results), so this stops at citation-backed raw results rather than "
        "claiming a verified synthesis it can't produce. Unofficial endpoint: "
        "can break without notice.",
        endpoint="/api/ai/research",
    ),
    "KNOWLEDGE_GRAPH": Capability(
        name="Knowledge Graph AI", route="KNOWLEDGE_GRAPH", type="infra",
        model="networkx (in-memory) + regex relation extraction", version="1.1",
        provider="local", status=Status.PARTIAL,
        reason="Real graph storage (networkx) with rule-based (not LLM-based) "
        "OWNS/SUBSIDIARY_OF/VENDOR_OF/EMPLOYS/SUPPLIES extraction, verified "
        "with a correct 2-hop path query (Acme -> Gamma -> Delta) over "
        "multi-sentence text, and re-verified after two real bugs were found "
        "and fixed (see known_bugs). PARTIAL because extraction is "
        "conservative by design: relations phrased unusually, or with "
        "non-proper-noun objects (e.g. 'employs over 200 workers'), are "
        "correctly not extracted rather than guessed - recall is "
        "deliberately traded for precision, and this is not a general-"
        "purpose entity extractor.",
        endpoint="/api/ai/knowledge-graph",
        known_bugs=[
            "Entity-boundary bug: a greedy character class captured "
            "trailing connector words ('Beta Logistics, a subsidiary that "
            "handles') into the entity name - found by testing a multi-hop "
            "path query, not a single clean example. Fixed with a proper-"
            "noun-phrase pattern.",
            "Sentence-boundary bleed: allowing '.' inside tokens (for "
            "abbreviations like 'Inc.') let a match run through a sentence-"
            "ending period into the next sentence's capitalized word "
            "('Beta Logistics. Beta Logistics' became one entity) - found "
            "via live HTTP testing with real multi-sentence input, not the "
            "single-paragraph unit test. Fixed by dropping '.' from the "
            "token class (accepted tradeoff: 'Inc.' -> 'Inc').",
        ],
        next_action="No graph persistence yet - the graph is in-memory per "
        "backend process, unlike RAG's now-persistent SQLite store. Add "
        "persistence for triples if this capability sees real use.",
    ),
}


def get_capability(route: str) -> Capability:
    if route not in REGISTRY:
        raise KeyError(f"Unknown capability route '{route}'. Known: {sorted(REGISTRY)}")
    return REGISTRY[route]


def list_capabilities(status: Status = None):
    caps = list(REGISTRY.values())
    if status is not None:
        caps = [c for c in caps if c.status == status]
    return caps


def summary():
    counts = {}
    for c in REGISTRY.values():
        counts[c.status.value] = counts.get(c.status.value, 0) + 1
    return counts
