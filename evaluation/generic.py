"""Score ANY model on the project's 45 evaluation questions.

evaluation/evaluator.py's evaluate_model() is tied to this repo's DeepSeekV3
(it calls model.generate with this project's signature). Fine-tuning an
external model (e.g. a Hugging Face checkpoint) needs the SAME questions and
the SAME scoring, or before/after numbers cannot be compared - so this takes a
plain callable instead of a model.

    from evaluation.generic import evaluate_generator
    results = evaluate_generator(lambda q: my_model_answer(q), verbose=True)
    print_report(results)
"""

from evaluation.evaluator import EVAL_FILES, load_eval_set, score_item
from evaluation.financial_metrics import aggregate


def evaluate_generator(generate_fn, categories=None, verbose: bool = False,
                       label: str = "Financial LLM POC Evaluation") -> dict:
    """generate_fn(question: str) -> answer: str.

    Returns the same structure evaluate_model() does, so print_report() and
    any stored result JSON stay comparable across models."""
    categories = categories or list(EVAL_FILES)
    results = {"categories": {}, "details": []}

    for category in categories:
        items = load_eval_set(category)
        if not items:
            results["categories"][category] = {
                "total": 0, "correct": 0, "accuracy": None,
                "note": "No evaluation items found",
            }
            continue

        scored = []
        for item in items:
            prediction = generate_fn(item["question"])
            record = score_item(category, item, prediction)
            scored.append(record)
            results["details"].append(record)
            if verbose:
                mark = "OK " if record["correct"] else "XX "
                print(f"  {mark}{item['id']}: {prediction[:70]!r}")
        results["categories"][category] = aggregate(scored)

    results["overall"] = aggregate(results["details"])
    results["label"] = label
    return results
