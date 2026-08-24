"""Financial Chat (Part 24 / §38).

    USER -> QUERY ROUTER -> { LLM | CALCULATOR | RAG } -> FINAL ANSWER + SOURCE

Generation always comes from this repository's own PyTorch model. The
calculator supplies exact arithmetic; RAG supplies document context.
Nothing is answered from an external LLM or API.
"""

from dataclasses import dataclass, field

import torch

from app.backend.services.financial_router import (
    LIVE_DATA_UNAVAILABLE, Route, classify, extract_financial_values,
)
from data_sources.tokenizer import get_encoding
from tools.financial_calculator import CALCULATIONS, CalculationError, calculate

# Which calculation to run given the values a user supplied. First entry
# whose required inputs are all present wins.
CALC_INTENTS = [
    ("ebitda_margin", ("ebitda", "revenue"), ["ebitda margin", "ebitda"]),
    ("gross_margin", ("gross_profit", "revenue"), ["gross margin", "gross profit"]),
    ("net_profit_margin", ("net_income", "revenue"), ["net margin", "net profit margin", "net income"]),
    ("operating_margin", ("operating_income", "revenue"), ["operating margin", "operating income"]),
    ("roe", ("net_income", "equity"), ["roe", "return on equity"]),
    ("roa", ("net_income", "total_assets"), ["roa", "return on assets"]),
    ("debt_to_equity", ("debt", "equity"), ["debt to equity", "debt/equity", "d/e"]),
    ("free_cash_flow", ("operating_cash_flow", "capex"), ["free cash flow", "fcf"]),
    ("eps", ("net_income", "shares"), ["eps", "earnings per share"]),
    ("revenue_growth", ("revenue", "prior_revenue"), ["revenue growth", "growth"]),
]

# Maps the router's extracted keys onto calculator argument names.
ARG_ALIASES = {
    "equity": "shareholders_equity",
    "debt": "total_debt",
    "capex": "capital_expenditure",
    "shares": "shares_outstanding",
}

DISCLAIMER = (
    "Educational information from a research POC model - not personalized "
    "financial advice."
)


@dataclass
class ChatResponse:
    answer: str
    route: str
    source: str
    detail: dict = field(default_factory=dict)
    sources: list = field(default_factory=list)

    def render(self) -> str:
        lines = [self.answer, "", f"Source: {self.source}"]
        for s in self.sources:
            lines.append(f"  - {s['citation']} (score {s['score']})")
        return "\n".join(lines)


class FinancialChat:
    def __init__(self, model=None, device="cpu", document_store=None,
                 max_new_tokens=60, temperature=0.8, top_k=40):
        self.model = model
        self.device = device
        self.document_store = document_store
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_k = top_k
        self.enc = get_encoding()

    # ---------------- model generation ----------------
    @torch.no_grad()
    def generate(self, prompt: str) -> str:
        if self.model is None:
            return "[no model checkpoint loaded]"
        ids = self.enc.encode_ordinary(prompt)
        max_ctx = self.model.config.block_size - self.max_new_tokens
        if len(ids) > max_ctx:
            ids = ids[-max_ctx:]
        context = torch.tensor(ids, dtype=torch.long, device=self.device).unsqueeze(0)
        out = self.model.generate(context, self.max_new_tokens, self.temperature, self.top_k)
        return self.enc.decode(out[0, len(ids):].tolist()).strip()

    # ---------------- routes ----------------
    def _handle_numerical(self, query, decision):
        values = extract_financial_values(query)
        lowered = query.lower()

        chosen = None
        for calc_name, required, phrases in CALC_INTENTS:
            if all(r in values for r in required):
                mentioned = any(p in lowered for p in phrases)
                if chosen is None or mentioned:
                    chosen = (calc_name, required)
                    if mentioned:
                        break

        if chosen is None:
            return ChatResponse(
                "I could not identify a complete calculation from the values provided. "
                f"Values I parsed: {values or 'none'}. "
                f"Available calculations: {', '.join(sorted(CALCULATIONS))}.",
                Route.NUMERICAL.value, "FINANCIAL CALCULATOR", {"parsed_values": values},
            )

        calc_name, required = chosen
        kwargs = {ARG_ALIASES.get(k, k): values[k] for k in required}
        try:
            result = calculate(calc_name, **kwargs)
        except CalculationError as e:
            return ChatResponse(f"Calculation could not be completed: {e}",
                                 Route.NUMERICAL.value, "FINANCIAL CALCULATOR",
                                 {"parsed_values": values, "error": str(e)})

        answer = f"{result.name} = {result.formatted()}\n\nFormula: {result.formula}"
        explanation = self.generate(f"Question: {query}\nAnswer: {result.name} is {result.formatted()}.")
        if explanation and self.model is not None:
            answer += f"\n\nModel explanation: {explanation}"

        return ChatResponse(
            answer, Route.NUMERICAL.value, "FINANCIAL CALCULATOR",
            {"calculation": calc_name, "value": result.value, "unit": result.unit,
             "inputs": result.inputs, "formula": result.formula,
             "model_explanation": explanation},
        )

    def _handle_document(self, query, decision):
        if self.document_store is None or not self.document_store.chunks:
            return ChatResponse(
                "No document is currently loaded. Upload a financial document first.",
                Route.DOCUMENT.value, "NOT AVAILABLE",
            )
        from rag.retriever import build_context
        retrieved = self.document_store.search(query, top_k=3)
        if not retrieved:
            return ChatResponse("No relevant content was found in the loaded document.",
                                 Route.DOCUMENT.value, "NOT AVAILABLE")

        context, sources = build_context(retrieved)
        prompt = f"Context: {context}\n\nQuestion: {query}\nAnswer:"
        generated = self.generate(prompt)
        answer = (
            f"{generated}\n\n--- Retrieved context (verbatim from the document) ---\n"
            f"{retrieved[0].chunk.text.strip()[:400]}"
        )
        return ChatResponse(answer, Route.DOCUMENT.value, "DOCUMENT (RAG)",
                             {"model_output": generated, "chunks": len(retrieved)}, sources)

    def _handle_live_data(self, query, decision):
        return ChatResponse(LIVE_DATA_UNAVAILABLE, Route.LIVE_DATA.value, "NOT AVAILABLE",
                             {"note": "No live market-data provider is configured."})

    def _handle_model(self, query, decision, route):
        generated = self.generate(f"Question: {query}\nAnswer:")
        source = "MODEL KNOWLEDGE" if route == Route.FINANCIAL_KNOWLEDGE else "MODEL"
        answer = generated if generated else "[model produced no output]"
        if route == Route.FINANCIAL_KNOWLEDGE:
            answer += f"\n\n({DISCLAIMER})"
        return ChatResponse(answer, route.value, source, {"model_output": generated})

    # ---------------- entry point ----------------
    def ask(self, query: str) -> ChatResponse:
        has_document = bool(self.document_store and self.document_store.chunks)
        decision = classify(query, has_document=has_document)
        route = decision.route

        if route == Route.NUMERICAL:
            response = self._handle_numerical(query, decision)
        elif route == Route.DOCUMENT:
            response = self._handle_document(query, decision)
        elif route == Route.LIVE_DATA:
            response = self._handle_live_data(query, decision)
        elif route == Route.UNKNOWN:
            response = ChatResponse("I could not interpret that question.",
                                     route.value, "NOT AVAILABLE")
        else:
            response = self._handle_model(query, decision, route)

        response.detail["routing"] = decision.to_dict()
        return response
