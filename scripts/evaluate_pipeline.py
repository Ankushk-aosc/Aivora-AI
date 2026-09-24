"""Score the DEPLOYED PIPELINE (run: python scripts/evaluate_pipeline.py)

Scores (calculator + glossary + model), not the raw model.

Every score so far measured the 101M model generating alone: 1-5 of 45. The
Space actually answers through a pipeline, and the pipeline is what a user
experiences. This measures the same 45 questions through it, and reports which
component answered each one.
"""
import os
import sys
import types
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUNDLE = os.path.join(ROOT, "hf_export")
sys.path.insert(0, ROOT)
sys.path.insert(0, BUNDLE)

# Stub gradio: not installed here, and the demo only needs its UI at runtime.
g = types.ModuleType("gradio")


class _W:
    def __init__(self, *a, **k):
        pass


class _I:
    def __init__(self, **k):
        self.fn = k.get("fn")

    def launch(self, *a, **k):
        pass


g.Textbox = g.Slider = g.Markdown = _W
g.Interface = _I
sys.modules["gradio"] = g

import app  # noqa: E402  (the Space's own answer pipeline)

from evaluation import print_report  # noqa: E402
from evaluation.generic import evaluate_generator  # noqa: E402

SOURCES = Counter()


def answer(question: str) -> str:
    text = app.answer(question, 48, 0.7)
    if "Computed exactly by the calculator" in text:
        SOURCES["calculator"] += 1
    elif "curated glossary" in text:
        SOURCES["glossary"] += 1
    elif "was withheld" in text:
        SOURCES["withheld"] += 1
    else:
        SOURCES["model"] += 1
    return text


if __name__ == "__main__":
    results = evaluate_generator(answer, verbose=True,
                                 label="Aivora pipeline (calculator + glossary + 101M model)")
    print_report(results)
    print("answered by:", dict(SOURCES))
