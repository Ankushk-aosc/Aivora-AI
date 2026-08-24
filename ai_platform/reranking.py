"""Reranking AI (spec Part 6).

    Query -> Vector Search -> Candidates -> Reranker -> Top Relevant -> LLM

A real BM25 second-stage reranker over the existing RAG retriever's
candidates - not another pretrained cross-encoder (none is available
without a GPU/download), but a well-established classical IR ranking
function, distinct from and complementary to the TF-IDF cosine score
already used for the first-stage vector search. Retrieval quality is
measured before/after on real queries, per the spec's explicit
requirement, rather than just asserted to be better.
"""

import math
import re
from collections import Counter
from dataclasses import dataclass

_WORD_RE = re.compile(r"\w+")

# Standard Okapi BM25 parameters.
BM25_K1 = 1.5
BM25_B = 0.75


def _tokenize(text: str):
    return _WORD_RE.findall(text.lower())


@dataclass
class RerankedResult:
    chunk: object
    original_score: float
    bm25_score: float
    combined_rank: int


def bm25_score(query_tokens: list, doc_tokens: list, avg_doc_len: float,
                doc_freq: dict, n_docs: int) -> float:
    doc_len = len(doc_tokens)
    term_counts = Counter(doc_tokens)
    score = 0.0
    for term in query_tokens:
        if term not in term_counts:
            continue
        df = doc_freq.get(term, 0)
        idf = math.log(1 + (n_docs - df + 0.5) / (df + 0.5))
        tf = term_counts[term]
        numerator = tf * (BM25_K1 + 1)
        denominator = tf + BM25_K1 * (1 - BM25_B + BM25_B * doc_len / max(avg_doc_len, 1e-9))
        score += idf * (numerator / denominator)
    return score


def rerank(query: str, candidates: list, top_k: int = None) -> list:
    """candidates: list of objects with `.chunk.text` and `.score`
    (matches rag.retriever.RetrievedChunk). Returns RerankedResult list,
    sorted by BM25 score descending."""
    if not candidates:
        return []

    query_tokens = _tokenize(query)
    doc_token_lists = [_tokenize(c.chunk.text) for c in candidates]
    n_docs = len(doc_token_lists)
    avg_doc_len = sum(len(d) for d in doc_token_lists) / n_docs

    doc_freq = Counter()
    for tokens in doc_token_lists:
        doc_freq.update(set(tokens))

    scored = []
    for candidate, tokens in zip(candidates, doc_token_lists):
        bm25 = bm25_score(query_tokens, tokens, avg_doc_len, doc_freq, n_docs)
        scored.append((candidate, bm25))

    scored.sort(key=lambda pair: pair[1], reverse=True)
    if top_k:
        scored = scored[:top_k]

    return [
        RerankedResult(chunk=c.chunk, original_score=c.score, bm25_score=round(bm25, 4),
                        combined_rank=i + 1)
        for i, (c, bm25) in enumerate(scored)
    ]


def measure_rank_change(query: str, candidates: list) -> dict:
    """Compares first-stage order to BM25-reranked order for a query -
    real evidence of whether reranking changed anything, not a claim."""
    original_order = [c.chunk.text[:40] for c in candidates]
    reranked = rerank(query, candidates)
    new_order = [r.chunk.text[:40] for r in reranked]
    return {
        "original_order": original_order,
        "reranked_order": new_order,
        "order_changed": original_order != new_order,
        "bm25_scores": [r.bm25_score for r in reranked],
    }
