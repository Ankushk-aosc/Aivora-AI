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


# Words that carry no meaning for judging whether two definitions agree.
_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being", "of", "to", "in",
    "on", "for", "from", "by", "with", "as", "at", "it", "its", "this", "that", "these",
    "those", "and", "or", "but", "if", "then", "than", "so", "such", "which", "what",
    "when", "where", "who", "whom", "how", "why", "can", "could", "would", "should",
    "may", "might", "will", "shall", "do", "does", "did", "has", "have", "had", "not",
    "no", "yes", "you", "your", "they", "their", "we", "our", "i", "he", "she", "him",
    "her", "his", "them", "there", "here", "over", "under", "into", "out", "up", "down",
    "also", "means", "meaning", "refers", "used", "use", "using", "one", "two", "all",
    "any", "some", "more", "most", "other", "another", "each", "per", "about",
}


def content_words(text: str):
    """Meaning-bearing words, lower-cased, stopwords and short tokens removed."""
    return {w for w in normalize(text).split() if len(w) > 2 and w not in _STOPWORDS}


def concept_overlap(prediction: str, reference: str) -> float:
    """Share of the reference's content words that the prediction also uses.

    Recall, not F1: a correct answer is often longer than the reference (it
    adds examples or a formula), and should not be punished for that. The
    strict matcher requires the reference's exact wording, which marks real
    answers wrong - "Depreciation spreads the cost of a tangible asset
    (machinery, vehicles, buildings) across its useful life" scored 0 against
    "spreading the cost of a tangible asset over its useful life"."""
    ref = content_words(reference)
    if not ref:
        return 0.0
    pred = content_words(prediction)
    if not pred:
        return 0.0
    # Count a reference word as covered if it appears, or if a prediction word
    # starts with it (spread/spreads, calculate/calculated).
    covered = 0
    for word in ref:
        if word in pred or any(p.startswith(word[:6]) for p in pred):
            covered += 1
    return covered / len(ref)


def semantic_match(prediction: str, reference: str, threshold: float = 0.5) -> bool:
    """True when the prediction covers enough of the reference's concepts.

    A rough signal only - word overlap cannot judge meaning. Measured against
    hand-graded examples it erred both ways: it failed a correct dividend
    definition (the answer said "earnings", the reference "profits") and
    passed a wrong one ("Costco depreciation is an accounting term that has a
    useful life expectancy") because the words happened to coincide. Use
    rubric_match where the verdict matters."""
    return concept_overlap(prediction, reference) >= threshold


def rubric_match(prediction: str, required_any) -> bool:
    """Grade against a per-question rubric: a list of concept GROUPS.

    An answer passes when it expresses something from EVERY group, where a
    group lists interchangeable wordings:

        [["tangible", "physical", "machinery"], ["useful life", "over time"]]

    This is how a person marks a definition - "did it say the asset is
    physical, and that the cost is spread over time?" - so it accepts
    paraphrase without accepting nonsense that merely shares vocabulary. The
    rubric is written with the question, never tuned to a model's output."""
    if not required_any:
        return False
    pred = normalize(prediction)
    for group in required_any:
        options = [group] if isinstance(group, str) else group
        if not any(normalize(option) in pred for option in options):
            return False
    return True


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
