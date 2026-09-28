"""What is a question actually about, and what kind of answer does it want?

The baseline's remaining failures were nearly all one defect: the knowledge
layer answered with the closest entry it held rather than declining.

    "What is discounting in a valuation?"  -> the definition of VALUATION
    "A company's current ratio is 0.7. What does that mean?"
                                          -> the definition of CURRENT RATIO
    "A company is profitable but keeps running out of cash. How?"
                                          -> the definition of EPS

Two different mistakes are visible there. The first is focus: a glossary key
matched a word in the question that was not its subject. The second is kind: an
interpretation question was answered with a definition, which is not an answer
to it at all.

So this module answers two questions about a question:

* `focus_phrases()` - the phrases a definition would have to be ABOUT. A
  glossary hit is only acceptable if it matches one of these.
* `is_interpretive()` / `is_definitional()` - whether the asker wants a
  definition or an explanation of something they have observed.

Nothing here knows any finance: it is grammar, kept away from the glossary so
each can be tested on its own.
"""

import re

# Nouns that head a question without being its subject: in "the purpose of an
# annual report" the subject is the annual report, not the purpose.
GENERIC_HEADS = {
    "purpose", "meaning", "difference", "differences", "role", "effect",
    "effects", "impact", "point", "idea", "definition", "significance",
    "importance", "function", "use", "uses", "benefit", "benefits",
    "advantage", "advantages", "disadvantage", "disadvantages", "problem",
    "reason", "reasons", "cause", "causes", "implication", "implications",
    "consequence", "consequences", "example", "examples", "kind", "kinds",
    "type", "types", "sort", "section", "sections", "part", "parts",
}

# Openers that ask for a definition.
DEFINITIONAL_OPENERS = [
    "what is", "what are", "what was", "what does", "define", "definition of",
    "meaning of", "explain", "tell me about", "describe",
]

# Phrases that ask for an explanation of something the asker has just stated.
# "what does it mean when ..." is deliberately absent: that is a definition
# asked in plain words, and retrieval should still answer it.
INTERPRETIVE_MARKERS = [
    "what does that mean", "what does this mean", "what does that tell",
    "what does this tell", "what could explain", "what might explain",
    "what should you check", "what should i check", "what would you check",
    "likely cause", "plausible reason", "plausible explanation",
    "how is that possible", "how can that be", "what is the implication",
    "what are the implications", "what does that imply", "what does that suggest",
    "why might", "why would", "why do", "why does", "why can", "why is", "why are",
    "what happened", "where did", "what is going on", "what is happening",
    "should you worry", "might that worry", "does that worry",
]

_SPLIT_TAIL = re.compile(
    r"\s+(?:in|of|on|for|within|under|from|about|across|during|when|with)\s+")


def _strip_question_words(clause: str) -> str:
    text = clause.strip().rstrip("?.!").strip()
    lowered = text.lower()
    for opener in sorted(DEFINITIONAL_OPENERS, key=len, reverse=True):
        if lowered.startswith(opener):
            text = text[len(opener):]
            break
    text = text.strip()
    for article in ("a ", "an ", "the "):
        if text.lower().startswith(article):
            text = text[len(article):]
    return text.strip()


def asking_clause(question: str) -> str:
    """The sentence that asks, ignoring any data supplied alongside it."""
    if not question:
        return ""
    lines = [line for line in question.strip().splitlines() if line.strip()]
    clause = lines[-1] if lines else question
    sentences = [s for s in re.split(r"(?<=[.?!])\s+", clause) if s.strip()]
    asking = [s for s in sentences if "?" in s] or sentences[-1:]
    return (asking[-1] if asking else clause).strip()


def is_interpretive(question: str) -> bool:
    """True when the asker has stated an observation and wants it explained."""
    text = (question or "").lower()
    if any(marker in text for marker in INTERPRETIVE_MARKERS):
        return True
    clause = asking_clause(question).lower()
    # A bare "How?" or "Why?" after a described situation.
    return bool(re.search(r"\b(how|why)\s*\?", clause))


def is_definitional(question: str) -> bool:
    if is_interpretive(question):
        return False
    clause = asking_clause(question).lower()
    return any(clause.startswith(opener) or f" {opener} " in clause
               for opener in DEFINITIONAL_OPENERS)


def focus_phrases(question: str):
    """Phrases a definition must be about, most specific first.

    "What is discounting in a valuation?"          -> ["discounting"]
    "What is the purpose of an annual report?"     -> ["annual report"]
    "Difference between IFRS and US GAAP?"         -> ["ifrs", "us gaap"]
    """
    clause = asking_clause(question)
    subject = _strip_question_words(clause)
    if not subject:
        return []

    lowered = subject.lower()

    # "the difference between X and Y" - either side is a legitimate subject.
    between = re.search(r"between\s+(.+?)\s+and\s+(.+)$", lowered)
    if between:
        sides = []
        for side in between.groups():
            side = side.strip()
            for article in ("a ", "an ", "the "):
                if side.startswith(article):
                    side = side[len(article):]
            sides.append(side.strip())
        return [s for s in sides if s]

    # "X of Y" / "X in Y": the head is the subject unless the head is a word
    # like "purpose" that carries no subject of its own.
    parts = _SPLIT_TAIL.split(lowered, maxsplit=1)
    head = parts[0].strip()
    tail = parts[1].strip() if len(parts) > 1 else ""
    head_words = [w for w in re.findall(r"[a-z&]+", head)]
    # Only when the head is NOTHING BUT a generic word does the subject move to
    # the tail: "the purpose of an annual report" is about the annual report,
    # but "the MD&A section of an annual report" is about the MD&A.
    if tail and len(head_words) == 1 and head_words[-1] in GENERIC_HEADS:
        for article in ("a ", "an ", "the "):
            if tail.startswith(article):
                tail = tail[len(article):]
        return [tail.strip(), head]
    # Only the head, never head + tail: including the tail let a term that
    # matches the tail alone look like the subject ("valuation" for "discounting
    # in a valuation"). A multi-word term whose head matches is handled in
    # matches_focus, which checks the rest of its words against the subject.
    if tail and head_words and head_words[-1] in GENERIC_HEADS:
        return [head, tail.strip()]
    return [head]


def _stem(word: str) -> str:
    """Crude plural stripping - enough for "liabilities"/"liability"."""
    for suffix, replacement in (("ies", "y"), ("ses", "s"), ("es", ""), ("s", "")):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[:-len(suffix)] + replacement
    return word


def matches_focus(term: str, question: str) -> bool:
    """Is `term` what the question is about, rather than merely mentioned in it?

    A term LONGER than the question's head ("cost of goods sold" for a question
    headed "cost") is accepted only if its remaining words are in the subject
    too - otherwise "cost of debt" would answer a question about the cost of
    equity.
    """
    if not term:
        return False
    term = term.lower().strip()
    subject = _strip_question_words(asking_clause(question)).lower()
    subject_words = set(re.findall(r"[a-z&]+", subject))
    term_words = set(re.findall(r"[a-z&]+", term))

    for phrase in (p.lower() for p in focus_phrases(question)):
        if not phrase:
            continue
        if term == phrase:
            return True
        if phrase in term:
            # term is the longer, more specific name: every word of it must
            # actually appear in what was asked.
            if term_words <= subject_words:
                return True
            continue
        if term in phrase:
            return True
        # Plural/singular and stem agreement: "current liability" asks about the
        # entry named "current liabilities", and comparing whole words missed it.
        phrase_words = set(re.findall(r"[a-z&]+", phrase))
        term_stems = {_stem(w) for w in term_words}
        phrase_stems = {_stem(w) for w in phrase_words}
        shared = {w for w in term_stems & phrase_stems if len(w) > 3}
        if shared and len(shared) >= min(len(term_stems), len(phrase_stems)):
            return True
    return False

# Openers that introduce a noun phrase as the subject. "what does" is absent on
# purpose: "What does it mean when a company buys back its own shares?" has no
# noun subject to check, and it is exactly the kind of paraphrase the semantic
# retrieval layer exists to answer.
_EXPLICIT_OPENERS = ["what is", "what are", "define", "definition of", "meaning of"]

# Words that cannot be a subject on their own.
_NON_SUBJECT = {"it", "that", "this", "they", "you", "we", "i", "he", "she",
                "mean", "means", "meant", "happen", "happens", "there", "one"}


def has_explicit_subject(question: str) -> bool:
    """True when the question names its subject as a noun phrase.

    The focus check is only fair on these. A paraphrase that describes a
    situation instead of naming a term ("How is a company financed by borrowing
    rather than issuing shares?") has no subject to compare a retrieved term
    against, and requiring one would disable paraphrase retrieval entirely -
    which is what it was added to do.
    """
    clause = asking_clause(question).lower()
    if re.search(r"difference between .+ and .+", clause):
        return True
    if not any(clause.startswith(opener) for opener in _EXPLICIT_OPENERS):
        return False
    phrases = focus_phrases(question)
    if not phrases:
        return False
    words = set(re.findall(r"[a-z&0-9-]+", phrases[0]))
    return bool(words - _NON_SUBJECT)
