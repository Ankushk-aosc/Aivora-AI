"""Retrieval must answer paraphrases WITHOUT inventing matches.

The risk with retrieval over 92 short glossary entries is a confident wrong
definition: earlier iterations answered "how do I bake sourdough bread?" with
the liquidity entry, and "what happens to prices when money loses value over
time?" with depreciation. Both are worse than saying nothing, because the
pipeline's fallback (a 101M model, clearly labelled as unreliable) is honest
about not knowing.

Run: python tests/test_knowledge_retrieval.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.backend.services.chat_service import FINANCIAL_KNOWLEDGE_BASE  # noqa: E402
from app.backend.services.knowledge_retrieval import (  # noqa: E402
    index_kind, retrieve_definition,
)

# (question, expected term or None for "must not answer")
CASES = [
    ("What does it mean when a company buys back its own shares?", "share buyback"),
    ("How is a company financed by borrowing rather than issuing shares?", "leverage"),
    ("Why would a company hold cash instead of investing it?", "liquidity"),
    # Must NOT answer: plain-English reading points at the wrong entry
    # ("loses value over time" sounds like depreciation, but this is inflation).
    ("What happens to prices when money loses value over time?", None),
    # Must NOT answer: nothing to do with finance.
    ("How do I bake sourdough bread?", None),
    ("Tell me a joke", None),
    ("What is the capital of France?", None),
    ("What is the weather like in Paris today?", None),
]

PASSED, FAILED = [], []


def check(name, ok, detail=""):
    (PASSED if ok else FAILED).append(name)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" - {detail}" if detail else ""))


def main():
    for question, expected in CASES:
        hit = retrieve_definition(question, FINANCIAL_KNOWLEDGE_BASE)
        got = hit["term"] if hit else None
        if expected is None:
            check(f"declines: {question[:46]}", got is None, f"returned {got!r}")
        else:
            check(f"retrieves {expected}: {question[:34]}", got == expected,
                  f"got {got!r}" + (f" score {hit['score']}" if hit else ""))

    # Exact terms must still go through the cheap keyword path unchanged.
    from app.backend.services.chat_service import FinancialChat

    chat = FinancialChat(model=None)
    for question, must_contain in (("What is EBITDA?", "Earnings Before Interest"),
                                   ("What is depreciation?", "tangible"),
                                   ("What is a stock?", "fractional ownership")):
        answer = chat.ask(question).answer
        check(f"glossary still answers: {question}", must_contain in answer, answer[:70])

    print(f"\nindex in use: {index_kind()}")
    print(f"{len(PASSED)}/{len(PASSED) + len(FAILED)} passed")
    sys.exit(1 if FAILED else 0)


if __name__ == "__main__":
    main()
