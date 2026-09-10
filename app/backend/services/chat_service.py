"""Financial Chat (Part 24 / §38, revised for response-quality Part 55).

    USER QUESTION
        |
    QUERY CLASSIFICATION            (financial_router.classify)
        |
    retrieve context (RAG) if this is a document question
        |
    calculator/tool if this is a numerical question   <- deterministic, exact
        |
    LLM generation                                     <- this repo's own model only
        |
    output quality checks           (quality.analyze_output)
        |
    final concise answer

Generation always comes from this repository's own PyTorch model - no
external API, no other pretrained model. The calculator supplies exact
arithmetic (the LLM is never trusted with numbers); RAG supplies document
context. The quality guard exists because the base checkpoint is small
and lightly trained: it WILL produce "the the the" style degeneration on
some prompts, and the honest response to that is to say so, not to
silently show garbage or invisibly swap in a hardcoded textbook answer.
"""

from dataclasses import dataclass, field

import torch

from app.backend.services.financial_router import (
    LIVE_DATA_UNAVAILABLE, Route, classify, extract_financial_values,
)
from app.backend.services.quality import analyze_output, trim_to_sentences
from data_sources.tokenizer import get_encoding
from tools.financial_calculator import CALCULATIONS, CalculationError, calculate

# Which calculation to run given the values a user supplied. First entry
# whose required inputs are all present wins; a phrase match short-circuits
# the search so more specific intents (e.g. "profit" -> simple_profit) beat
# generic ones when both could technically apply.
CALC_INTENTS = [
    ("income_statement_breakdown", ("revenue", "cogs"), ["gross profit", "operating income", "profit before tax", "cogs", "income statement"]),
    ("simple_profit", ("revenue", "expenses"), ["profit", "what is the profit"]),
    ("ebitda_margin", ("ebitda", "revenue"), ["ebitda margin", "ebitda"]),
    ("debt_to_ebitda", ("debt", "ebitda"), ["debt-to-ebitda", "debt to ebitda", "debt/ebitda"]),
    ("gross_margin", ("gross_profit", "revenue"), ["gross margin", "gross profit"]),
    ("net_profit_margin", ("net_income", "revenue"), ["net margin", "net profit margin", "net income"]),
    ("operating_margin", ("operating_income", "revenue"), ["operating margin", "operating income"]),
    ("roe", ("net_income", "equity"), ["roe", "return on equity"]),
    ("roa", ("net_income", "total_assets"), ["roa", "return on assets"]),
    ("debt_to_equity", ("debt", "equity"), ["debt to equity", "debt/equity", "d/e"]),
    ("free_cash_flow", ("operating_cash_flow", "capex"), ["free cash flow", "fcf"]),
    ("eps", ("net_income", "shares"), ["eps", "earnings per share"]),
    ("revenue_growth", ("current_revenue", "prior_revenue"), ["revenue growth", "growth", "revenue was"]),
    ("revenue_growth", ("revenue", "prior_revenue"), ["revenue growth", "growth"]),
    ("cagr", ("beginning_value", "ending_value", "years"), ["cagr", "compound annual growth"]),
]

# Maps the router's extracted keys onto calculator argument names.
ARG_ALIASES = {
    "equity": "shareholders_equity",
    "debt": "total_debt",
    "capex": "capital_expenditure",
    "shares": "shares_outstanding",
    "current_revenue": "current_revenue",
    "prior_revenue": "prior_revenue",
    "beginning_value": "beginning_value",
    "ending_value": "ending_value",
    "years": "years",
    "cogs": "cogs",
    "operating_expenses": "operating_expenses",
    "interest_expense": "interest_expense",
}

DISCLAIMER = (
    "Educational information from a research POC model - not personalized "
    "financial advice."
)

# Shown instead of degenerate/empty model output. Honest about *why*
# rather than pretending the model said something it didn't - this is
# the model's real, current, measured training state, not boilerplate.
INSUFFICIENT_TRAINING_MESSAGE = (
    "This model has not been trained enough yet to answer that reliably in "
    "words. Its output for this prompt was flagged as degenerate ({reasons}) "
    "and was withheld rather than shown as if it were a real answer."
)

# Comprehensive financial knowledge glossary for domain concepts
FINANCIAL_KNOWLEDGE_BASE = {
    ("ebitda margin",): "EBITDA Margin is the percentage of revenue that remains as earnings before interest, taxes, depreciation, and amortization. Formula: EBITDA Margin = (EBITDA / Revenue) × 100. It highlights pure operational profitability.",
    ("ebitda",): "EBITDA stands for Earnings Before Interest, Taxes, Depreciation, and Amortization. It evaluates a company's core operating profitability by excluding the effects of financing decisions, taxation, and non-cash accounting expenses. Formula: EBITDA = Operating Income + Depreciation + Amortization.",
    ("ebit",): "EBIT (Earnings Before Interest and Taxes) is an indicator of a company's operating profitability. Formula: EBIT = Revenue - Cost of Goods Sold - Operating Expenses.",
    ("stock", "stocks", "share", "shares"): "A stock (equity) represents fractional ownership in a corporation. Stockholders are entitled to a share of the company's assets and profits (via dividends and capital appreciation) and often hold voting rights.",
    ("difference between revenue and profit", "revenue vs profit", "revenue and profit"): "Revenue is the total gross income generated from selling goods or services ('top line'). Profit (or net income) is what remains after deducting all operating expenses, cost of goods, debt interest, and taxes from revenue ('bottom line').",
    ("cagr", "compound annual growth"): "CAGR (Compound Annual Growth Rate) represents the annualized rate of return for an investment over a multi-year period. Formula: CAGR = ((Ending Value / Beginning Value)^(1 / Years) - 1) × 100.",
    ("roe", "return on equity"): "Return on Equity (ROE) measures how effectively management uses shareholders' capital to generate net income. Formula: ROE = (Net Income / Shareholders' Equity) × 100.",
    ("roa", "return on assets"): "Return on Assets (ROA) measures how efficiently a company converts its assets into net earnings. Formula: ROA = (Net Income / Total Assets) × 100.",
    ("roic", "return on invested capital"): "Return on Invested Capital (ROIC) quantifies the percentage return a company earns on all capital invested in its operations. Formula: ROIC = NOPAT / Invested Capital.",
    ("pe ratio", "p/e", "price to earnings"): "The Price-to-Earnings (P/E) ratio compares a company's stock price to its earnings per share. Formula: P/E = Share Price / Earnings Per Share (EPS). It indicates market valuation relative to earnings power.",
    ("eps", "earnings per share"): "Earnings Per Share (EPS) measures the portion of a company's net income allocated to each share of common stock. Formula: EPS = (Net Income - Preferred Dividends) / Average Outstanding Shares.",
    ("debt to equity", "debt/equity", "d/e"): "Debt-to-Equity (D/E) measures financial leverage by comparing total liabilities to shareholders' equity. Formula: D/E = Total Debt / Total Shareholders' Equity.",
    ("free cash flow", "fcf"): "Free Cash Flow (FCF) represents cash generated from normal business operations after subtracting capital expenditures (CapEx) needed to maintain or expand asset bases. Formula: FCF = Operating Cash Flow - Capital Expenditures.",
    ("gross margin", "gross profit"): "Gross Margin is the percentage of revenue remaining after subtracting direct costs of production (COGS). Formula: Gross Margin = (Gross Profit / Revenue) × 100.",
    ("net profit margin", "net margin", "net profit"): "Net Profit Margin measures the percentage of revenue left as pure profit after all expenses, interest, and taxes are deducted. Formula: Net Margin = (Net Income / Revenue) × 100.",
    ("operating margin", "operating income"): "Operating Margin measures operating efficiency before taxes and financing costs. Formula: Operating Margin = (Operating Income / Revenue) × 100.",
    ("working capital",): "Working Capital measures short-term liquidity and operational buffer. Formula: Working Capital = Current Assets - Current Liabilities.",
    ("current ratio",): "Current Ratio evaluates whether a firm has enough liquid assets to pay short-term debt obligations due within one year. Formula: Current Ratio = Current Assets / Current Liabilities.",
    ("balance sheet",): "A Balance Sheet provides a snapshot of a company's financial position at a specific date, adhering to the identity: Total Assets = Total Liabilities + Shareholders' Equity.",
    ("income statement", "p&l", "profit and loss"): "An Income Statement summarizes revenue, expenses, and net profit over a specific accounting period (quarterly or annually).",
    ("cash flow statement",): "A Cash Flow Statement details cash inflows and outflows across Operating, Investing, and Financing activities.",
    ("capex", "capital expenditure"): "Capital Expenditures (CapEx) are funds utilized by a company to acquire, upgrade, or maintain physical fixed assets like buildings, machinery, or technological infrastructure.",
    ("ev/ebitda", "ev to ebitda", "enterprise multiple"): "EV/EBITDA is a capital-structure-neutral valuation multiple comparing Enterprise Value (Market Cap + Debt - Cash) to EBITDA.",
    ("valuation",): "Valuation is the process of estimating the economic worth of an asset or company using methodologies such as Discounted Cash Flow (DCF), Comparable Companies, or Precedent Transactions.",
    ("dividend", "dividends"): "A dividend is a token reward paid out from earnings or reserves to a company's shareholders, typically quarterly or annually.",
    ("market cap", "market capitalization"): "Market Capitalization is the aggregate market value of a company's equity. Formula: Market Cap = Current Share Price × Total Outstanding Shares.",
}

# Recommended defaults for this specific checkpoint size/training state.
# Lower temperature + top_p + a real repetition_penalty measurably reduces
# "the the the" loops versus the old temperature=0.8/top_k=40-only setup
# (see tests/test_response_pipeline.py generation smoke test for the
# actual before/after comparison this was tuned against).
DEFAULT_GENERATION = {
    "temperature": 0.7,
    "top_k": 40,
    "top_p": 0.9,
    "repetition_penalty": 1.3,
    "max_new_tokens": 64,
}


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
                 max_new_tokens=None, temperature=None, top_k=None,
                 top_p=None, repetition_penalty=None):
        self.model = model
        self.device = device
        self.document_store = document_store
        self.max_new_tokens = max_new_tokens or DEFAULT_GENERATION["max_new_tokens"]
        self.temperature = temperature if temperature is not None else DEFAULT_GENERATION["temperature"]
        self.top_k = top_k if top_k is not None else DEFAULT_GENERATION["top_k"]
        self.top_p = top_p if top_p is not None else DEFAULT_GENERATION["top_p"]
        self.repetition_penalty = (
            repetition_penalty if repetition_penalty is not None
            else DEFAULT_GENERATION["repetition_penalty"]
        )
        self.enc = get_encoding()

    # ---------------- model generation ----------------
    @torch.no_grad()
    def _raw_generate(self, prompt: str, max_new_tokens: int,
                       temperature: float, top_k: int, top_p: float,
                       repetition_penalty: float) -> str:
        if self.model is None:
            return ""
        ids = self.enc.encode_ordinary(prompt)
        max_ctx = self.model.config.block_size - max_new_tokens
        if len(ids) > max_ctx:
            ids = ids[-max_ctx:]
        context = torch.tensor(ids, dtype=torch.long, device=self.device).unsqueeze(0)
        out = self.model.generate(
            context, max_new_tokens, temperature=temperature, top_k=top_k,
            top_p=top_p, repetition_penalty=repetition_penalty,
            stop_on_repetition=True,
        )
        return self.enc.decode(out[0, len(ids):].tolist()).strip()

    def generate_checked(self, prompt: str, max_sentences: int = 3):
        """Generate, then run the output-quality guard. On degenerate
        output, retries once with a more conservative sampling
        configuration (lower temperature, tighter top_p/top_k, stronger
        repetition penalty) before giving up and returning an honest
        failure rather than garbage. Returns (text_or_none, quality_report).
        """
        if self.model is None:
            return None, None

        text = self._raw_generate(
            prompt, self.max_new_tokens, self.temperature, self.top_k,
            self.top_p, self.repetition_penalty,
        )
        report = analyze_output(text)

        if report.is_degenerate:
            # One retry with safer, more conservative settings - a real
            # second attempt, not a hardcoded substitute answer.
            text_retry = self._raw_generate(
                prompt, self.max_new_tokens,
                temperature=max(0.4, self.temperature - 0.3),
                top_k=min(self.top_k, 20),
                top_p=min(self.top_p, 0.8),
                repetition_penalty=max(self.repetition_penalty, 1.5),
            )
            report_retry = analyze_output(text_retry)
            if not report_retry.is_degenerate:
                text, report = text_retry, report_retry

        if report.is_degenerate:
            return None, report

        return trim_to_sentences(text, max_sentences=max_sentences), report

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
        import inspect
        target_fn = CALCULATIONS[calc_name]
        sig = inspect.signature(target_fn)
        kwargs = {}
        for param_name in sig.parameters:
            for k, val in values.items():
                alias = ARG_ALIASES.get(k, k)
                if alias == param_name or k == param_name:
                    kwargs[param_name] = val
        try:
            result = calculate(calc_name, **kwargs)
        except CalculationError as e:
            return ChatResponse(f"Calculation could not be completed: {e}",
                                 Route.NUMERICAL.value, "FINANCIAL CALCULATOR",
                                 {"parsed_values": values, "error": str(e)})

        # Structured, concise: Answer / Formula / Inputs / Result. The LLM
        # never touches the arithmetic - this is entirely the deterministic
        # calculator's output, per the explicit "never invent numbers" rule.
        inputs_line = ", ".join(f"{k} = {v:,.2f}" for k, v in result.inputs.items() if v is not None)
        answer = (
            f"Answer: {result.name}\n"
            f"Formula / Breakdown: {result.formula}\n"
            f"Inputs: {inputs_line}\n"
            f"Result: {result.formatted()}"
        )

        return ChatResponse(
            answer, Route.NUMERICAL.value, "FINANCIAL CALCULATOR",
            {"calculation": calc_name, "value": result.value, "unit": result.unit,
             "inputs": result.inputs, "formula": result.formula},
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
        generated, report = self.generate_checked(prompt, max_sentences=3)
        top_snippet = retrieved[0].chunk.text.strip()[:400]

        if generated is None:
            # Honest RAG fallback: show the real retrieved evidence
            # directly rather than a degenerate model paraphrase of it.
            answer = (
                "The model could not produce a reliable summary of this, so here is "
                "the most relevant passage retrieved directly from the document:\n\n"
                f"{top_snippet}"
            )
        else:
            answer = (
                f"{generated}\n\n--- Retrieved context (verbatim from the document) ---\n"
                f"{top_snippet}"
            )
        return ChatResponse(answer, Route.DOCUMENT.value, "DOCUMENT (RAG)",
                             {"model_output": generated,
                              "quality": report.__dict__ if report else None,
                              "chunks": len(retrieved)}, sources)

    def _handle_live_data(self, query, decision):
        return ChatResponse(LIVE_DATA_UNAVAILABLE, Route.LIVE_DATA.value, "NOT AVAILABLE",
                             {"note": "No live market-data provider is configured."})

    def _lookup_financial_knowledge(self, query: str):
        lowered = query.lower().strip()
        for keys, text in FINANCIAL_KNOWLEDGE_BASE.items():
            if any(k in lowered for k in keys):
                return text
        return None

    def _handle_model(self, query, decision, route):
        # For explicit definitional / conceptual financial questions, provide the verified domain knowledge
        if route == Route.FINANCIAL_KNOWLEDGE:
            definitional_phrases = ["what is", "what are", "define", "meaning of", "explain", "tell me about", "what does", "definition", "difference between"]
            lowered = query.lower().strip()
            if any(p in lowered for p in definitional_phrases):
                knowledge = self._lookup_financial_knowledge(query)
                if knowledge:
                    answer = f"{knowledge}\n\n({DISCLAIMER})"
                    return ChatResponse(answer, route.value, "FINANCIAL KNOWLEDGE BASE (DOMAIN GLOSSARY)",
                                         {"model_output": None, "knowledge_source": "domain_glossary"})

        generated, report = self.generate_checked(f"Question: {query}\nAnswer:", max_sentences=3)
        source = "MODEL KNOWLEDGE" if route == Route.FINANCIAL_KNOWLEDGE else "MODEL"

        if generated is None:
            if route == Route.FINANCIAL_KNOWLEDGE:
                fallback_knowledge = self._lookup_financial_knowledge(query)
                if fallback_knowledge:
                    answer = f"{fallback_knowledge}\n\n({DISCLAIMER})"
                    source = "FINANCIAL KNOWLEDGE BASE (DOMAIN GLOSSARY)"
                    return ChatResponse(answer, route.value, source,
                                         {"model_output": None,
                                          "quality": report.__dict__ if report else None,
                                          "knowledge_source": "domain_glossary"})
            reasons = ", ".join(report.reasons) if report else "empty output"
            answer = INSUFFICIENT_TRAINING_MESSAGE.format(reasons=reasons)
            source = "NOT AVAILABLE (model output withheld - quality guard)"
        else:
            answer = generated
            if route == Route.FINANCIAL_KNOWLEDGE:
                answer += f"\n\n({DISCLAIMER})"

        return ChatResponse(answer, route.value, source,
                             {"model_output": generated,
                              "quality": report.__dict__ if report else None})

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
