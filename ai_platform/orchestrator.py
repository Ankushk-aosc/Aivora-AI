"""AI Orchestrator + Capability Router (spec Parts 24-27).

    Input -> Security Check -> Capability Router -> Capability -> Verification
          -> Output Security Check -> Observability -> Response

Reuses, rather than duplicates, everything already built and verified:
app.backend.services.financial_router for the core GENERAL/FINANCIAL_
KNOWLEDGE/NUMERICAL/DOCUMENT/LIVE_DATA classification, tools.financial_
calculator for arithmetic, rag for retrieval, ai_platform.{security,
verification,observability,registry} for the platform-level concerns.

Every route not backed by real code returns a NOT_IMPLEMENTED response
that names the missing dependency - it never falls through to the LLM
pretending to be a vision/speech/database/etc capability.
"""

import inspect
import re
from dataclasses import dataclass, field

from app.backend.services.financial_router import (
    Route as CoreRoute, classify as core_classify, extract_financial_values,
)
from ai_platform.registry import REGISTRY, Status
from ai_platform.security import check_input, check_output
from ai_platform.verification import sufficient_evidence, verify_response, INSUFFICIENT_EVIDENCE_MESSAGE
from ai_platform.observability import track_request
from tools.financial_calculator import CALCULATIONS, CalculationError, calculate

# Keyword triggers for capabilities the core financial_router doesn't know
# about. Checked BEFORE the core router so an explicit "analyze this image"
# request isn't silently swallowed into GENERAL_LLM.
_EXTRA_INTENTS = [
    ("DATABASE_AI", re.compile(r"\bsql\b|\bquery the database\b", re.I)),
    ("VISION_AI", re.compile(r"\b(image|photo|picture|screenshot|chart image|scanned)\b", re.I)),
    ("SPEECH_AI", re.compile(r"\b(transcribe|speech to text|voice (memo|note|command)|audio (file|recording))\b", re.I)),
    ("CODE_AI", re.compile(r"\b(write|generate|run|execute)\b.*\b(code|script|program)\b", re.I)),
    ("FRAUD_AI", re.compile(r"\bfraud (detection|score|risk)\b|\bdetect(ion)?\b.*\bfraud\b|"
                            r"\bfraudulent\b|\bfraud\b.*\btransactions?\b", re.I)),
    ("ANOMALY_AI", re.compile(r"\b(anomal(y|ies)|outliers?|unusual (transactions?|spending|activity)|duplicate invoices?)\b", re.I)),
    ("FORECASTING_AI", re.compile(r"\bforecast\b|\bproject(ed|ion)?\b.*\b(revenue|expense|cash.?flow|demand)\b|\bpredict\b.*\b(revenue|sales|demand)\b", re.I)),
    ("RECOMMENDATION_AI", re.compile(r"\brecommend(ation)?\b.*\b(vendor|cost|budget|allocation)\b", re.I)),
    ("MULTILINGUAL_AI", re.compile(r"\btranslate\b|\bin (spanish|french|german|hindi|chinese|japanese)\b", re.I)),
    ("RESEARCH_AI", re.compile(r"\bresearch\b.*\b(report|question|topic|news|articles?)\b|"
                               r"\b(find|search for)\b.*\b(recent|latest)\b.*\b(news|articles?)\b", re.I)),
    ("KNOWLEDGE_GRAPH", re.compile(r"\brelationship(s)? between\b|\bknowledge graph\b|\bwho owns\b", re.I)),
]

_CORE_TO_CAPABILITY = {
    CoreRoute.GENERAL: "GENERAL_LLM",
    CoreRoute.FINANCIAL_KNOWLEDGE: "FINANCIAL_LLM",
    CoreRoute.NUMERICAL: "CALCULATOR",
    CoreRoute.DOCUMENT: "RAG",
    CoreRoute.LIVE_DATA: "LIVE_DATA",
    CoreRoute.UNKNOWN: "GENERAL_LLM",
}


@dataclass
class OrchestratorResponse:
    capability: str
    status: str            # registry Status value at time of routing
    answer: str
    reason: str
    model: str = "Not available"
    tools_used: list = field(default_factory=list)
    knowledge_sources: list = field(default_factory=list)
    verification: dict = None
    security: dict = None
    request_id: str = None

    def to_dict(self):
        return {
            "capability": self.capability, "status": self.status, "answer": self.answer,
            "reason": self.reason, "model": self.model, "tools_used": self.tools_used,
            "knowledge_sources": self.knowledge_sources, "verification": self.verification,
            "security": self.security, "request_id": self.request_id,
        }


def classify_capability(query: str, has_document: bool = False) -> tuple:
    """Returns (capability_route, reason). Checks the extra-capability
    keyword triggers first, then falls back to the core financial router."""
    for route, pattern in _EXTRA_INTENTS:
        m = pattern.search(query or "")
        if m:
            return route, f"matched pattern for {route}: {m.group(0)!r}"

    decision = core_classify(query, has_document=has_document)
    capability = _CORE_TO_CAPABILITY[decision.route]
    return capability, "; ".join(decision.reasons)


class AIOrchestrator:
    def __init__(self, model=None, device="cpu", document_store=None):
        self.model = model
        self.device = device
        self.document_store = document_store

    def _generate(self, prompt: str, max_new_tokens: int = 40) -> str:
        if self.model is None:
            return "[no model checkpoint loaded]"
        import torch
        from data_sources.tokenizer import get_encoding
        enc = get_encoding()
        ids = enc.encode_ordinary(prompt)
        max_ctx = self.model.config.block_size - max_new_tokens
        if len(ids) > max_ctx:
            ids = ids[-max_ctx:]
        context = torch.tensor(ids, dtype=torch.long, device=self.device).unsqueeze(0)
        with torch.no_grad():
            out = self.model.generate(context, max_new_tokens, 0.8, 40)
        return enc.decode(out[0, len(ids):].tolist()).strip()

    def _not_implemented(self, capability: str) -> OrchestratorResponse:
        cap = REGISTRY[capability]
        return OrchestratorResponse(
            capability=capability, status=cap.status.value,
            answer=f"{cap.name} is not available: {cap.reason}",
            reason="Capability registry reports this as "
            f"{cap.status.value}; see ai_platform/registry.py for details.",
        )

    def handle(self, query: str) -> OrchestratorResponse:
        # 1. Input security check
        input_security = check_input(query)
        if input_security.blocked:
            with track_request("AI_SECURITY", model="none", device=self.device) as record:
                record.output_preview = "blocked: prompt injection detected"
                response = OrchestratorResponse(
                    capability="AI_SECURITY", status=Status.TESTED.value,
                    answer="This request was blocked by the AI security layer.",
                    reason="Prompt-injection pattern detected in the input.",
                    security=input_security.to_dict(),
                )
                response.request_id = record.request_id
            return response

        # 2. Capability routing
        has_document = bool(self.document_store and self.document_store.chunks)
        capability, reason = classify_capability(query, has_document=has_document)
        cap_entry = REGISTRY[capability]

        with track_request(capability, model=str(self.model.__class__.__name__)
                            if self.model else "none", device=self.device) as record:
            if cap_entry.status in (Status.NOT_IMPLEMENTED, Status.BLOCKED):
                response = self._not_implemented(capability)
                response.reason = reason
                record.output_preview = response.answer[:120]
                response.request_id = record.request_id
                return response

            response = self._dispatch(capability, query, reason)
            record.tools_used = response.tools_used
            record.retrieval_sources = len(response.knowledge_sources)
            record.output_preview = (response.answer or "")[:120]
            response.request_id = record.request_id

        # 3. Output security check
        output_security = check_output(response.answer)
        response.security = {"input": input_security.to_dict(),
                              "output": output_security.to_dict()}
        return response

    def _dispatch(self, capability: str, query: str, reason: str) -> OrchestratorResponse:
        cap_entry = REGISTRY[capability]

        if capability == "CALCULATOR":
            return self._handle_numerical(query, reason)
        if capability == "RAG":
            return self._handle_rag(query, reason)
        if capability == "LIVE_DATA":
            return self._handle_live_data(query, reason)
        if capability == "RESEARCH_AI":
            return self._handle_research(query, reason)
        if capability == "ANOMALY_AI":
            return OrchestratorResponse(
                capability=capability, status=cap_entry.status.value,
                answer="Anomaly detection is available via POST /api/ai/anomaly "
                "with a list of transactions - it needs structured data, not a "
                "chat question.",
                reason=reason, tools_used=["ai_platform.anomaly"],
            )
        if capability == "FRAUD_AI":
            return OrchestratorResponse(
                capability=capability, status=cap_entry.status.value,
                answer="Fraud scoring is available via POST /api/ai/fraud with "
                "a list of transactions ({amt, category, trans_date_trans_time, "
                "city_pop}) - it needs structured data, not a chat question. "
                f"Model: {cap_entry.model}, recall 0.938 / precision 0.393 on "
                "held-out synthetic test data.",
                reason=reason, tools_used=["ai_platform.fraud"],
            )
        if capability == "CODE_AI":
            return OrchestratorResponse(
                capability=capability, status=cap_entry.status.value,
                answer="Code execution is available via POST /api/ai/code/execute "
                "with {\"code\": \"...\"} - runs in a sandboxed subprocess with a "
                "timeout and restricted builtins (process isolation, not "
                "container-level - see the registry entry for exact scope).",
                reason=reason, tools_used=["ai_platform.code_sandbox"],
            )
        if capability == "SPEECH_AI":
            return OrchestratorResponse(
                capability=capability, status=cap_entry.status.value,
                answer="Text-to-speech is available via POST "
                "/api/ai/speech/synthesize with {\"text\": \"...\"} (returns a "
                "WAV file). Speech-to-text is not implemented.",
                reason=reason, tools_used=["ai_platform.speech"],
            )
        if capability == "KNOWLEDGE_GRAPH":
            return OrchestratorResponse(
                capability=capability, status=cap_entry.status.value,
                answer="Knowledge graph queries are available via POST "
                "/api/ai/knowledge-graph/{add,relationships,path} - it needs "
                "structured text ingestion and entity names, not a single "
                "chat question.",
                reason=reason, tools_used=["ai_platform.knowledge_graph"],
            )
        if capability == "RECOMMENDATION_AI":
            from ai_platform.recommendation import get_recommendations
            recs = get_recommendations()
            if not recs:
                answer = "No recommendations triggered by the current data."
            else:
                answer = "\n\n".join(f"- {r.recommendation}\n  {r.reason}" for r in recs)
            return OrchestratorResponse(
                capability=capability, status=cap_entry.status.value, answer=answer,
                reason=reason, tools_used=["ai_platform.recommendation"],
                knowledge_sources=[{"citation": s, "score": None}
                                    for r in recs for s in r.data_sources],
            )
        if capability == "MULTILINGUAL_AI":
            from ai_platform.multilingual import detect_language
            try:
                d = detect_language(query)
                answer = (f"Detected language: {d.language_name} ({d.language_code}), "
                          f"confidence {d.confidence}. Translation is not implemented "
                          "(see registry for why) - only detection is available.")
            except ValueError as e:
                answer = f"Language detection failed: {e}"
            return OrchestratorResponse(
                capability=capability, status=cap_entry.status.value, answer=answer,
                reason=reason, tools_used=["ai_platform.multilingual"],
            )
        if capability == "DATABASE_AI":
            return OrchestratorResponse(
                capability=capability, status=cap_entry.status.value,
                answer="Database queries are available via POST /api/ai/database "
                "with {\"question\": \"...\", \"table\": \"...\"} - handles "
                "'total by X', 'top N', 'total', 'average' against a local "
                "SQLite database (read-only, template-matched, not free-form "
                "NL2SQL). See GET /api/ai/database/schema for available tables.",
                reason=reason, tools_used=["ai_platform.database_ai"],
            )
        if capability == "FORECASTING_AI":
            return OrchestratorResponse(
                capability=capability, status=cap_entry.status.value,
                answer="Forecasting is available via POST /api/ai/forecast with "
                "a historical numeric series - it needs structured data, not a "
                "chat question.",
                reason=reason, tools_used=["ai_platform.forecasting"],
            )

        # GENERAL_LLM / FINANCIAL_LLM
        answer = self._generate(f"Question: {query}\nAnswer:")
        verification = verify_response(answer)
        return OrchestratorResponse(
            capability=capability, status=cap_entry.status.value, answer=answer,
            reason=reason, model=cap_entry.model, verification=verification.to_dict(),
        )

    @staticmethod
    def _resolve_arg(param_name: str, values: dict):
        for candidate in (param_name, param_name.replace("shareholders_", ""),
                          param_name.replace("total_", "")):
            if candidate in values:
                return values[candidate]
        return None

    def _handle_numerical(self, query: str, reason: str) -> OrchestratorResponse:
        values = extract_financial_values(query)
        chosen = None
        for name, fn in CALCULATIONS.items():
            sig = inspect.signature(fn)
            required = [p.name for p in sig.parameters.values() if p.default is inspect.Parameter.empty]
            if required and all(self._resolve_arg(r, values) is not None for r in required):
                chosen = name
                break

        if chosen is None:
            return OrchestratorResponse(
                capability="CALCULATOR", status=Status.TESTED.value,
                answer=f"Could not identify a complete calculation. Parsed values: "
                f"{values or 'none'}. Available: {sorted(CALCULATIONS)}",
                reason=reason,
            )

        try:
            fn = CALCULATIONS[chosen]
            sig = inspect.signature(fn)
            kwargs = {}
            for p in sig.parameters.values():
                v = self._resolve_arg(p.name, values)
                if v is not None:
                    kwargs[p.name] = v
            result = calculate(chosen, **kwargs)
        except CalculationError as e:
            return OrchestratorResponse(
                capability="CALCULATOR", status=Status.TESTED.value,
                answer=f"Calculation error: {e}", reason=reason,
            )

        explanation = self._generate(f"Question: {query}\nAnswer: {result.name} is {result.formatted()}.")
        verification = verify_response(explanation, verified_value=result.value)
        answer = f"{result.name} = {result.formatted()}\n\nFormula: {result.formula}"
        if explanation and self.model is not None:
            answer += f"\n\nModel explanation: {explanation}"

        return OrchestratorResponse(
            capability="CALCULATOR", status=Status.TESTED.value, answer=answer,
            reason=reason, tools_used=[chosen], verification=verification.to_dict(),
        )

    def _handle_live_data(self, query: str, reason: str) -> OrchestratorResponse:
        from ai_platform.live_data import LiveDataError, extract_symbol, get_quote

        symbol = extract_symbol(query)
        if symbol is None:
            return OrchestratorResponse(
                capability="LIVE_DATA", status=Status.TESTED.value,
                answer="Current market data is not available: could not "
                "identify a ticker symbol or company name in the request.",
                reason=reason,
            )
        try:
            quote = get_quote(symbol)
        except LiveDataError as e:
            return OrchestratorResponse(
                capability="LIVE_DATA", status=Status.TESTED.value,
                answer="Current market data is not available.",
                reason=f"{reason}; fetch failed: {e}",
            )

        answer = (f"{quote['symbol']}: {quote['price']} {quote['currency']} "
                  f"({quote['exchange']}). Previous close: {quote['previous_close']}.")
        return OrchestratorResponse(
            capability="LIVE_DATA", status=Status.TESTED.value, answer=answer,
            reason=reason, tools_used=["ai_platform.live_data"],
            knowledge_sources=[{"citation": quote["source"], "score": 1.0}],
        )

    def _handle_research(self, query: str, reason: str) -> OrchestratorResponse:
        from ai_platform.research import ResearchError, format_report, search

        try:
            report = search(query, max_results=5)
        except ResearchError as e:
            return OrchestratorResponse(
                capability="RESEARCH_AI", status=Status.PARTIAL.value,
                answer=f"Search failed: {e}", reason=reason,
            )

        sources = [{"citation": r.citation(), "score": None} for r in report.results]
        return OrchestratorResponse(
            capability="RESEARCH_AI", status=Status.PARTIAL.value,
            answer=format_report(report), reason=reason,
            tools_used=["ai_platform.research (DuckDuckGo HTML)"],
            knowledge_sources=sources,
        )

    def _handle_rag(self, query: str, reason: str) -> OrchestratorResponse:
        if self.document_store is None or not self.document_store.chunks:
            return OrchestratorResponse(
                capability="RAG", status=Status.TESTED.value,
                answer="No document is currently loaded. Upload one first.",
                reason=reason,
            )
        from rag.retriever import build_context
        from ai_platform.reranking import rerank

        # Over-fetch from the first-stage retriever, then let BM25 re-rank
        # down to the final top-3 (Part 6's pipeline). BM25 determines
        # ORDER only - the sufficient_evidence() threshold was calibrated
        # against the first-stage TF-IDF cosine scale (~0-1), and BM25
        # scores are unbounded (~0-5+ in testing); substituting one for the
        # other would silently break the threshold, so `retrieved` keeps
        # each candidate's original TF-IDF score for the gate and citations.
        candidates = self.document_store.search(query, top_k=8)
        retrieved = [r.chunk for r in rerank(query, candidates, top_k=3)] if candidates else []
        # Re-attach original scores by chunk identity for the gate/citations.
        score_by_chunk = {id(c.chunk): c.score for c in candidates}
        retrieved = [
            type(candidates[0])(chunk=chunk, score=score_by_chunk.get(id(chunk), 0.0))
            for chunk in retrieved
        ] if candidates else []
        top_score = retrieved[0].score if retrieved else None

        if not sufficient_evidence(top_score):
            return OrchestratorResponse(
                capability="RAG", status=Status.TESTED.value,
                answer=INSUFFICIENT_EVIDENCE_MESSAGE,
                reason=f"{reason}; top retrieval score {top_score} below "
                "the grounding threshold - refusing to answer ungrounded.",
                knowledge_sources=[],
            )

        context, sources = build_context(retrieved)
        generated = self._generate(f"Context: {context}\n\nQuestion: {query}\nAnswer:")
        verification = verify_response(generated, retrieved_chunks=[r.chunk.text for r in retrieved])

        return OrchestratorResponse(
            capability="RAG", status=Status.TESTED.value, answer=generated,
            reason=reason, tools_used=["rag.DocumentStore"],
            knowledge_sources=sources, verification=verification.to_dict(),
        )
