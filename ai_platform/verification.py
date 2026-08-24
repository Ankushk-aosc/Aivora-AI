"""AI Judge / Verification layer (spec Part 22).

Deliberately does NOT use another LLM call as an "unquestioned authority"
(the spec's own warning). Every check here is deterministic: does the
model's text contain a number that contradicts the calculator's verified
result, and is a RAG answer actually grounded in the retrieved text.
"""

import re
from dataclasses import dataclass, field

from evaluation.financial_metrics import extract_numbers, normalize

INSUFFICIENT_EVIDENCE_MESSAGE = "Insufficient information available."

# Below this retrieval score, treat the evidence as too weak to answer from.
# Tuned against the TF-IDF retriever: genuinely relevant queries scored
# 0.15-0.40 in testing, while queries sharing only incidental/generic
# vocabulary with a document scored ~0.05. This is a real, tunable
# threshold, not a guarantee - a short document with little vocabulary
# can still produce false-positive overlap above this line.
MIN_RAG_SCORE = 0.10


@dataclass
class VerificationResult:
    passed: bool
    checks: list = field(default_factory=list)  # list of {"check", "passed", "detail"}

    def to_dict(self):
        return {"passed": self.passed, "checks": self.checks}


def verify_numeric_claim(model_text: str, verified_value: float, tolerance: float = 0.5) -> dict:
    """Check whether the model's explanation text contains a number that
    contradicts the calculator's verified result. Does not require the
    model to restate the number - only flags it if it states a DIFFERENT one."""
    numbers = extract_numbers(model_text)
    if not numbers:
        return {"check": "numeric_consistency", "passed": True,
                "detail": "Model text contains no numbers to contradict the verified value."}

    close = [n for n in numbers if abs(n - verified_value) <= tolerance]
    if close or not numbers:
        return {"check": "numeric_consistency", "passed": True,
                "detail": f"Verified value {verified_value} matched or not contradicted."}

    return {"check": "numeric_consistency", "passed": False,
            "detail": f"Model stated number(s) {numbers} that do not match the "
            f"verified value {verified_value} (tolerance {tolerance})."}


def verify_rag_grounding(answer_text: str, retrieved_chunks: list, min_overlap: float = 0.1) -> dict:
    """Check that the answer shares meaningful word overlap with the
    retrieved context, as a cheap grounding proxy (not a semantic
    entailment check - stated plainly, not oversold)."""
    if not retrieved_chunks:
        return {"check": "rag_grounding", "passed": False,
                "detail": "No retrieved chunks - answer cannot be grounded."}

    context_words = set()
    for chunk in retrieved_chunks:
        context_words.update(re.findall(r"\w+", normalize(chunk).lower()))
    answer_words = set(re.findall(r"\w+", normalize(answer_text).lower()))
    if not answer_words:
        return {"check": "rag_grounding", "passed": False, "detail": "Empty answer."}

    overlap = len(answer_words & context_words) / len(answer_words)
    passed = overlap >= min_overlap
    return {"check": "rag_grounding", "passed": passed,
            "detail": f"Word overlap with retrieved context: {overlap:.2f} "
            f"({'>=' if passed else '<'} {min_overlap})"}


def sufficient_evidence(top_score: float, min_score: float = MIN_RAG_SCORE) -> bool:
    """Part 7: 'If evidence is insufficient, say Insufficient information
    available. Do not fabricate an answer.' This is the gate that decides
    that, based on the retriever's own similarity score."""
    return top_score is not None and top_score >= min_score


def verify_response(model_text: str, verified_value: float = None,
                     retrieved_chunks: list = None) -> VerificationResult:
    checks = []
    if verified_value is not None:
        checks.append(verify_numeric_claim(model_text, verified_value))
    if retrieved_chunks is not None:
        checks.append(verify_rag_grounding(model_text, retrieved_chunks))

    passed = all(c["passed"] for c in checks) if checks else True
    return VerificationResult(passed=passed, checks=checks)
