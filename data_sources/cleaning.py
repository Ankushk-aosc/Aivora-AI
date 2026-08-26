"""Text cleaning and deduplication.

Designed to be safe for financial text: the notation preserve-list is
checked before/after cleaning in tests, and none of the transformations
here touch currency symbols, percentages, or financial abbreviations.
"""

import hashlib
import re
import unicodedata

MIN_TEXT_LENGTH = 20

# Financial notation that must never be stripped or mangled.
FINANCIAL_NOTATION_PATTERN = re.compile(
    r"(₹|\$|€|%|EPS|P/E|EBITDA|ROE|ROIC|FCF|YoY|QoQ|FY\d{4}|Q[1-4]|\d+(\.\d+)?%)"
)

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_WHITESPACE_RE = re.compile(r"[ \t]+")
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")
_BOILERPLATE_PATTERNS = [
    re.compile(r"click here to (subscribe|read more)", re.IGNORECASE),
    re.compile(r"all rights reserved", re.IGNORECASE),
    re.compile(r"cookie(s)? policy", re.IGNORECASE),
    re.compile(r"terms (of|and) (service|conditions)", re.IGNORECASE),
]


def strip_html(text: str) -> str:
    return _HTML_TAG_RE.sub(" ", text)


def normalize_unicode(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    # Drop control characters but keep newlines/tabs.
    return "".join(ch for ch in text if ch >= " " or ch in "\n\t")


def normalize_whitespace(text: str) -> str:
    text = _WHITESPACE_RE.sub(" ", text)
    text = _MULTI_NEWLINE_RE.sub("\n\n", text)
    return text.strip()


def remove_boilerplate(text: str) -> str:
    for pattern in _BOILERPLATE_PATTERNS:
        text = pattern.sub("", text)
    return text


def clean_text(text: str) -> str:
    if not text:
        return ""
    text = strip_html(text)
    text = normalize_unicode(text)
    text = remove_boilerplate(text)
    text = normalize_whitespace(text)
    return text


def is_valid_record(text: str, min_length: int = MIN_TEXT_LENGTH) -> bool:
    if not text or not text.strip():
        return False
    if len(text.strip()) < min_length:
        return False
    # Malformed: mostly non-alphanumeric noise.
    alnum_count = sum(ch.isalnum() for ch in text)
    if alnum_count < min_length * 0.3:
        return False
    return True


def content_hash(text: str) -> str:
    """Hash used for exact-duplicate detection and train/eval leakage checks."""
    normalized = " ".join(text.split()).lower()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def near_dup_signature(text: str, shingle_size: int = 5) -> str:
    """Cheap near-duplicate signature: hash of a sorted set of word shingles.

    Not a full MinHash — good enough to catch boilerplate-heavy near
    duplicates without adding a heavy dependency.
    """
    words = re.findall(r"\w+", text.lower())
    if len(words) < shingle_size:
        return content_hash(text)
    shingles = {
        " ".join(words[i:i + shingle_size])
        for i in range(0, len(words) - shingle_size + 1, shingle_size)
    }
    signature_input = "|".join(sorted(shingles))
    return hashlib.sha256(signature_input.encode("utf-8")).hexdigest()


class Deduplicator:
    """Tracks seen exact-hashes and near-dup signatures across a stream."""

    def __init__(self):
        self._seen_exact = set()
        self._seen_near = set()

    def is_duplicate(self, text: str) -> bool:
        h = content_hash(text)
        if h in self._seen_exact:
            return True
        sig = near_dup_signature(text)
        if sig in self._seen_near:
            return True
        self._seen_exact.add(h)
        self._seen_near.add(sig)
        return False


def load_eval_questions() -> list:
    """Every evaluation question's exact text, lowercased, for leakage
    filtering during data prep (not just post-hoc detection).

    Imports evaluation.evaluator lazily: evaluator.py imports from
    data_sources (get_encoding), so an unconditional top-of-file import
    here would be circular. By the time this function is actually called
    (during a real prepare_dataset() run), both modules are fully loaded.
    """
    from evaluation.evaluator import EVAL_FILES, load_eval_set

    questions = []
    for category in EVAL_FILES:
        for item in load_eval_set(category):
            q = item.get("question")
            if q:
                questions.append(q.lower())
    return questions


def contains_eval_leakage(text: str, eval_questions: list) -> bool:
    lowered = text.lower()
    return any(q in lowered for q in eval_questions)


def clean_and_filter(records, min_length: int = MIN_TEXT_LENGTH, stats: dict = None,
                      eval_questions: list = None):
    """Generator: clean, validate, dedup, and leakage-filter a stream of
    {"text": ...} records.

    eval_questions, when given, excludes any record whose text contains an
    evaluation question verbatim - the same check_leakage() uses to detect
    leakage after the fact, applied here so leakage can't happen in the
    first place rather than only being caught (and wasting a prepare run)
    after. A real example this caught: a Dolly-15k record happened to
    contain "what is a balance sheet?" verbatim, coincidentally colliding
    with this project's own qa_004 evaluation item, at financial_poc's
    larger per-dataset token budget (the smaller `small` preset's budget
    never happened to sample that record).

    If `stats` is provided, it is updated in place as records are consumed,
    so the caller can inspect counts after the generator is exhausted.
    """
    dedup = Deduplicator()
    if stats is None:
        stats = {}
    stats.update({"seen": 0, "kept": 0, "removed_invalid": 0, "removed_duplicate": 0,
                  "removed_eval_leakage": 0})

    for record in records:
        stats["seen"] += 1
        text = clean_text(record["text"])

        if not is_valid_record(text, min_length=min_length):
            stats["removed_invalid"] += 1
            continue

        if dedup.is_duplicate(text):
            stats["removed_duplicate"] += 1
            continue

        if eval_questions and contains_eval_leakage(text, eval_questions):
            stats["removed_eval_leakage"] += 1
            continue

        stats["kept"] += 1
        cleaned = dict(record)
        cleaned["text"] = text
        yield cleaned
