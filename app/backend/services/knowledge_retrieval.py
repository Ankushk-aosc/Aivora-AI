"""Retrieval over the curated finance glossary (the RAG step for definitions).

The glossary is matched by substring: "What is depreciation?" hits the
"depreciation" entry. That fails the moment someone asks in their own words -
"how do I know if a company can pay its short-term bills?" contains none of
the glossary's keys, so the question falls through to a 101M model that
answers it wrongly.

This indexes the same glossary with the project's TF-IDF embedder (rag/) and
retrieves the closest entry by cosine similarity, so paraphrases reach the
right definition. It is deliberately conservative: below MIN_SCORE it returns
nothing rather than a confidently wrong entry, and the answer is always
labelled as retrieved rather than generated.

Nothing new is installed: rag.TfidfEmbedder is numpy-only.
"""

import numpy as np

from evaluation.financial_metrics import content_words

# Below this cosine similarity the best entry is treated as "no match". Chosen
# so that clear paraphrases retrieve their entry while off-topic questions
# ("what is the weather") retrieve nothing - see tests/test_knowledge_retrieval.py.
MIN_SCORE = 0.18

# A question must look like finance before any definition is retrieved. Without
# this gate, TF-IDF happily returned "liquidity" for "how do I bake sourdough
# bread?" at a higher score than it gave real finance paraphrases: with 92 short
# entries, shared common words dominate the similarity.
FINANCE_HINTS = {
    "company", "companies", "business", "firm", "corporate", "money", "cash", "capital",
    "revenue", "sales", "profit", "profits", "loss", "income", "earnings", "cost", "costs",
    "price", "prices", "asset", "assets", "liability", "liabilities", "debt", "equity",
    "share", "shares", "shareholder", "stock", "stocks", "bond", "bonds", "dividend",
    "invest", "investor", "investment", "investments", "market", "markets", "fund",
    "interest", "tax", "taxes", "balance", "sheet", "ratio", "margin", "valuation",
    "accounting", "financial", "finance", "economy", "economic", "inflation", "bank",
    "loan", "borrow", "borrowing", "lend", "credit", "audit", "budget", "expense",
    "expenses", "customer", "supplier", "inventory", "depreciation", "amortization",
    "payment", "pay", "paid", "bills", "quarter", "annual", "fiscal", "portfolio",
}


def _content_text(text):
    """Stopword-stripped text for indexing and querying."""
    return " ".join(sorted(content_words(text)))


def _shares_word(asked, entry_words, stem=6):
    """Do the two word sets share a word, allowing for endings?

    Exact set intersection is too brittle for this check: a question about
    "borrowing" shares no exact word with an entry that says "borrowed", so a
    correct leverage match was being thrown away."""
    entry_stems = {w[:stem] for w in entry_words}
    return any(w[:stem] in entry_stems for w in asked)


class SemanticIndex:
    """Sentence-embedding index (all-MiniLM-L6-v2, ~90 MB, CPU).

    Measured against the TF-IDF index on the same paraphrases, embeddings
    separate finance from noise far better: off-topic questions score 0.07-0.14
    where real ones score 0.42-0.69, whereas TF-IDF gave "how do I bake
    sourdough bread?" 0.25 - above some genuine finance paraphrases.

    Still not perfect: "prices when money loses value over time" retrieves
    depreciation rather than inflation, which is the plain-English reading of
    those words. Hence MIN_SCORE is set for precision, not recall: a missed
    definition falls through to the rest of the pipeline, a wrong one is served
    to the user as fact."""

    # Accept outright above STRONG. Between SUPPORTED and STRONG, require the
    # entry to share at least one content word with the question as well.
    # That combination keeps the good matches ("financed by borrowing ..." ->
    # leverage, 0.46, shares "borrow") while rejecting the plausible-but-wrong
    # ones ("money loses value over time" -> depreciation, 0.48, shares
    # nothing: its text says "useful life", not "value" or "time").
    STRONG = 0.60
    SUPPORTED = 0.45
    MIN_SCORE = SUPPORTED
    MODEL = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(self, knowledge_base):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(self.MODEL)
        self.keys = list(knowledge_base.keys())
        self.texts = [knowledge_base[k] for k in self.keys]
        documents = [f"{' '.join(k)}: {t}" for k, t in zip(self.keys, self.texts)]
        self.matrix = self.model.encode(documents, normalize_embeddings=True,
                                        show_progress_bar=False)

    def search(self, question, top_k=1):
        if not question or not question.strip():
            return []
        vector = self.model.encode([question], normalize_embeddings=True,
                                   show_progress_bar=False)[0]
        scores = self.matrix @ vector
        asked = content_words(question)
        hits = []
        for i in np.argsort(-scores)[:top_k]:
            score = float(scores[i])
            if score >= self.STRONG:
                hits.append((score, self.keys[i], self.texts[i]))
            elif score >= self.SUPPORTED:
                entry_words = content_words(" ".join(self.keys[i]) + " " + self.texts[i])
                if _shares_word(asked, entry_words):
                    hits.append((score, self.keys[i], self.texts[i]))
        return hits


class GlossaryIndex:
    """TF-IDF index over the glossary, built once and queried per request."""

    def __init__(self, knowledge_base):
        from rag import TfidfEmbedder

        self.keys = []
        self.texts = []
        documents = []
        for keys, text in knowledge_base.items():
            self.keys.append(keys)
            self.texts.append(text)
            # Index the terms alongside the definition: a question usually
            # shares wording with the definition, but sometimes only with the
            # term itself.
            # Index content words only. Indexing whole sentences let "is", "the",
            # "company" and similar filler drive the similarity.
            documents.append(_content_text(" ".join(keys) + " " + text))

        self.embedder = TfidfEmbedder().fit(documents)
        self.matrix = self.embedder.encode(documents)

    def search(self, question, top_k=1):
        """[(score, keys, text)] best first, empty if nothing clears MIN_SCORE."""
        if not question or not question.strip():
            return []
        if not (content_words(question) & FINANCE_HINTS):
            return []
        vector = self.embedder.encode([_content_text(question)])[0]
        scores = self.matrix @ vector
        asked = content_words(question)
        hits = []
        for i in np.argsort(-scores)[:top_k]:
            if scores[i] < MIN_SCORE:
                continue
            # Same shared-word requirement the semantic index uses. Lexical
            # similarity alone put "prices when money loses value over time"
            # on "payback period", whose text happens to contain "time value
            # of money".
            entry_words = content_words(" ".join(self.keys[i]) + " " + self.texts[i])
            if _shares_word(asked, entry_words):
                hits.append((float(scores[i]), self.keys[i], self.texts[i]))
        return hits


_INDEX = None


def get_index(knowledge_base):
    """Build the index once per process; it is small but not free.

    Sentence embeddings when sentence-transformers is available, TF-IDF
    otherwise - the fallback keeps this working (less well) with no extra
    dependency, which matters for the lightweight Space bundle."""
    global _INDEX
    if _INDEX is None:
        try:
            _INDEX = SemanticIndex(knowledge_base)
        except Exception as e:  # not installed, or no network for the model
            print(f"knowledge_retrieval: falling back to TF-IDF ({type(e).__name__}: {e})")
            _INDEX = GlossaryIndex(knowledge_base)
    return _INDEX


def index_kind():
    return type(_INDEX).__name__ if _INDEX else "not built"


def retrieve_definition(question, knowledge_base):
    """Best glossary entry for a paraphrased question, or None."""
    hits = get_index(knowledge_base).search(question, top_k=1)
    if not hits:
        return None
    score, keys, text = hits[0]
    return {"text": text, "score": round(score, 3), "term": keys[0]}
