"""Regression battery for scripts/eval_heldout.numbers_in.

Guards the two defects found while diagnosing Phase 4's calculation score:
"2.50x" read as 2.0 (silent truncation, which can score a wrong answer
correct), and a number matched out of the middle of a version string.
"""
import sys

sys.path.insert(0, ".")
from scripts.eval_heldout import numbers_in  # noqa: E402

CASES = [
    ("2.50x", [2.5], "ratio carries an x unit"),
    ("2.5x", [2.5], "ratio carries an x unit"),
    ("2.50X", [2.5], "upper case unit"),
    ("2.50x (revenue = 6,000.00, total assets = 2,400.00)",
     [2.5, 6000.0, 2400.0], "the pipeline's calculation format"),
    ("2.50xy", [], "unreadable token is rejected, NOT truncated to 2.0"),
    ("1.2.3", [], "a version string is not a number"),
    ("Q3", [], "quarter label"),
    ("FY2024", [], "fiscal year label"),
    ("ABC123", [], "alphanumeric id"),
    ("Stock code: BL-9", [9.0], "hyphenated id - unchanged behaviour"),
    ("1,204.50", [1204.5], "thousands separator"),
    ("2,60.65", [], "malformed separator stays rejected"),
    ("38.33%", [38.33], "percentage"),
    ("The value is 2.50.", [2.5], "sentence-final period"),
    ("(2,110.00)", [2110.0], "parenthesised"),
    ("Revenue: 12,000.00 and 7,400.00", [12000.0, 7400.0], "two figures"),
    ("-5.5", [-5.5], "negative"),
    ("", [], "empty"),
    (None, [], "none"),
]


def main():
    failed = 0
    for text, expected, why in CASES:
        got = numbers_in(text)
        ok = got == expected
        failed += not ok
        print(f"[{'PASS' if ok else 'FAIL'}] {str(text)[:46]!r:<50} -> {got}  ({why})")
        if not ok:
            print(f"         expected {expected}")
    print(f"\n{len(CASES) - failed}/{len(CASES)} passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
