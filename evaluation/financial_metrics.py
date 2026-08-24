"""Metric functions for the Financial LLM POC Evaluation (Part 17 / §33).

Deliberately simple and inspectable: exact match, normalized match,
keyword coverage, and numeric tolerance. Nothing here is a benchmark of
general intelligence.
"""

import re

_PUNCT_RE = re.compile(r"[^\w\s%./-]")
_WS_RE = re.compile(r"\s+")
_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


def normalize(text: str) -> str:
    if text is None:
        return ""
    text = text.lower().strip()
    text = _PUNCT_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text)
    return text.strip()


def exact_match(prediction: str, reference: str) -> bool:
    return (prediction or "").strip() == (reference or "").strip()


def normalized_match(prediction: str, reference: str, acceptable=None) -> bool:
    pred = normalize(prediction)
    if not pred:
        return False
    candidates = [reference] + list(acceptable or [])
    for candidate in candidates:
        ref = normalize(candidate)
        if ref and ref in pred:
            return True
    return False


def keyword_coverage(prediction: str, keywords) -> float:
    """Fraction of reference keywords present in the prediction."""
    if not keywords:
        return 0.0
    pred = normalize(prediction)
    hits = sum(1 for kw in keywords if normalize(kw) in pred)
    return hits / len(keywords)


def extract_numbers(text: str):
    return [float(n) for n in _NUMBER_RE.findall(text or "")]


def numeric_match(prediction: str, expected_value: float, tolerance: float = 0.01) -> bool:
    """True if any number in the prediction matches expected within tolerance."""
    if expected_value is None:
        return False
    for value in extract_numbers(prediction):
        if abs(value - expected_value) <= tolerance:
            return True
    return False


def aggregate(results):
    """results: list of dicts with a boolean "correct" key."""
    total = len(results)
    if total == 0:
        return {"total": 0, "correct": 0, "accuracy": None}
    correct = sum(1 for r in results if r["correct"])
    return {
        "total": total,
        "correct": correct,
        "accuracy": round(100.0 * correct / total, 2),
    }
