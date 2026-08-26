"""Output-quality guard for LLM-generated text.

The base model is a small, lightly-trained checkpoint (101.7M params,
100 real training steps at last count - see checkpoints/base/*.json for
the actual measured loss). Left unchecked it will readily produce
degenerate output: "the the the the", empty strings, or a handful of
tokens looping forever. This module gives the chat pipeline a real,
inspectable way to detect that *after* generation, on top of the
in-generation repetition stopping in models/model.py's `generate()` -
belt and suspenders, since the in-generation check only sees a fixed
token window and can still let a subtler degeneration through.

Nothing here tries to make bad output look good. A degenerate result is
reported as degenerate; the caller decides whether to fall back to an
honest "not enough training yet" message rather than show it.
"""

import re
from dataclasses import dataclass, field


@dataclass
class QualityReport:
    is_empty: bool
    unique_word_ratio: float
    max_word_repeat: int
    max_repeated_ngram_count: int
    is_degenerate: bool
    reasons: list = field(default_factory=list)


def _words(text: str) -> list:
    return re.findall(r"[A-Za-z0-9']+", text.lower())


# Common English function words. A passage built almost entirely from these,
# with no substantive (content) words, is the OTHER failure mode this size
# of model produces alongside literal repetition: high word-level diversity
# ("the of a in that on for was") that is still meaningless. Deliberately a
# plain, readable list rather than an imported stopword corpus, so the rule
# stays inspectable.
_FUNCTION_WORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "of", "in", "on", "at", "to", "for", "and", "or", "but", "nor", "so",
    "that", "this", "these", "those", "with", "as", "by", "it", "its",
    "from", "not", "no", "do", "does", "did", "have", "has", "had",
    "will", "would", "can", "could", "should", "may", "might", "must",
    "i", "you", "he", "she", "we", "they", "them", "his", "her", "their",
    "our", "your", "what", "which", "who", "whom", "if", "then", "than",
    "there", "here", "when", "where", "why", "how", "up", "out", "about",
    "into", "over", "after", "before", "such", "some", "any", "all",
    "one", "s", "t", "re", "ve", "ll", "d", "m",
}


def _ngram_counts(words: list, n: int) -> dict:
    counts = {}
    for i in range(len(words) - n + 1):
        gram = tuple(words[i:i + n])
        counts[gram] = counts.get(gram, 0) + 1
    return counts


def analyze_output(text: str, min_words: int = 3) -> QualityReport:
    """Score generated text for the degeneration patterns undertrained
    small LMs actually produce: single-token loops, short-phrase loops,
    and near-zero lexical diversity. Every threshold below is a plain,
    inspectable rule - not a learned classifier - so a report can always
    be explained by pointing at the number that tripped it.
    """
    reasons = []
    stripped = text.strip()

    if not stripped:
        return QualityReport(True, 0.0, 0, 0, True, ["empty output"])

    words = _words(stripped)
    if len(words) < min_words:
        reasons.append(f"only {len(words)} word(s) produced")
        return QualityReport(False, 0.0, 0, 0, True, reasons)

    unique_ratio = len(set(words)) / len(words)
    if unique_ratio < 0.35:
        reasons.append(f"low lexical diversity (unique word ratio {unique_ratio:.2f})")

    # Content-word check: catches the OTHER degeneration pattern this size
    # of model produces - a diverse-looking but meaningless string of
    # function words with no substantive content, e.g. ".The in of that a."
    content_words = [w for w in words if w not in _FUNCTION_WORDS and len(w) >= 3]
    content_ratio = len(content_words) / len(words)
    if len(words) >= 5 and content_ratio < 0.25:
        reasons.append(
            f"almost no substantive content ({len(content_words)}/{len(words)} "
            "words are content words, rest are function words/fragments)"
        )

    word_counts = {}
    for w in words:
        word_counts[w] = word_counts.get(w, 0) + 1
    max_word_repeat = max(word_counts.values())
    # A single filler word dominating a short passage ("the the the ...")
    # is the single most common failure mode for this size of model.
    if max_word_repeat >= 4 and max_word_repeat / len(words) > 0.3:
        top_word = max(word_counts, key=word_counts.get)
        reasons.append(f"'{top_word}' repeated {max_word_repeat}/{len(words)} words")

    max_ngram_count = 0
    for n in (2, 3, 4):
        if len(words) < n:
            continue
        counts = _ngram_counts(words, n)
        if not counts:
            continue
        top_count = max(counts.values())
        max_ngram_count = max(max_ngram_count, top_count)
        if top_count >= 3:
            top_gram = max(counts, key=counts.get)
            reasons.append(f"phrase {' '.join(top_gram)!r} repeated {top_count} times")
            break

    is_degenerate = bool(reasons)
    return QualityReport(
        is_empty=False,
        unique_word_ratio=round(unique_ratio, 3),
        max_word_repeat=max_word_repeat,
        max_repeated_ngram_count=max_ngram_count,
        is_degenerate=is_degenerate,
        reasons=reasons,
    )


def trim_to_sentences(text: str, max_sentences: int = 3, max_chars: int = 480) -> str:
    """Enforce conciseness: cut at a sentence boundary rather than an
    arbitrary character count, then hard-cap length as a fallback for
    text with no punctuation at all (which degenerate output often is)."""
    stripped = text.strip()
    if not stripped:
        return stripped
    sentences = re.split(r"(?<=[.!?])\s+", stripped)
    kept = " ".join(sentences[:max_sentences]).strip()
    if len(kept) > max_chars:
        kept = kept[:max_chars].rsplit(" ", 1)[0].rstrip(",;:") + "..."
    return kept
