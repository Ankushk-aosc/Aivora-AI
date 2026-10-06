"""Phase 4: score the tool pipeline, with a pre-trained model doing extraction.

Chosen configuration (owner's decision, 2026-10-05): Qwen2.5-1.5B-Instruct,
zero-shot. Phase 3 put 0-shot and 3-shot within one wording item of each other,
and 0-shot avoids the few-shot abstention artifact documented in
reports/phase3/PHASE3_REPORT.md section 3. The pipeline supplies its own
structured prompt, so a few-shot block would be redundant anyway.

The model is given ONE job - read the context and copy a value with the span it
came from. Python does every piece of arithmetic. See pipeline/tool_pipeline.py.

This script does not take the span rule on trust. The pipeline is built to
guarantee it, but a guarantee that is never measured is an assumption, so every
answer is re-checked here against the context it came from, independently of the
pipeline's own validator.

    python scripts/run_tool_pipeline_eval.py --model Qwen/Qwen2.5-1.5B-Instruct
    python scripts/run_tool_pipeline_eval.py --model ... --split selection --limit 10
"""

import argparse
import json
import os
import sys
import time
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from pipeline.schema import normalise  # noqa: E402
from pipeline.tool_pipeline import ToolPipeline  # noqa: E402
from scripts.eval_base_models import LICENCES, free_memory_gb  # noqa: E402

# The pipeline's prompts already state the task and the output format, so the
# system prompt only has to stop the model adding commentary around the JSON.
SYSTEM_PROMPT = (
    "You extract values from financial text. You reply with a single JSON "
    "object and nothing else - no explanation, no markdown fence, no commentary. "
    "Copy values exactly as they appear. Never calculate. Never invent a number."
)


def build_llm(model_name, max_new_tokens=160):
    """A raw llm(prompt) -> str callable.

    Phase 3's helper returns a (question, context) function, which is the wrong
    shape here: the pipeline writes its own prompts and must hand them over
    untouched, or it is no longer the pipeline being measured.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    print(f"loading {model_name} on {device} ({dtype}) ...")
    started = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype)
    model.to(device)
    model.eval()
    print(f"loaded in {time.time() - started:.0f}s, "
          f"{sum(p.numel() for p in model.parameters()) / 1e6:.0f}M parameters")

    calls = {"n": 0}

    def llm(prompt):
        calls["n"] += 1
        messages = [{"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False,
                                             add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens,
                                 do_sample=False, temperature=None, top_p=None,
                                 pad_token_id=tokenizer.eos_token_id)
        generated = out[0, inputs["input_ids"].shape[1]:]
        return tokenizer.decode(generated, skip_special_tokens=True).strip()

    return llm, calls


def audit_spans(answers, items):
    """Re-check the span rule independently of the pipeline that produced it.

    EXPERIMENT_RULES_v2 section 1: every extracted value needs a validated
    source span, at 100%. A violation is any answer that states a value while
    citing no span, or citing a span that is not in the context it was given.
    """
    checked, violations = 0, []
    for answer, item in zip(answers, items):
        if answer.abstained:
            continue
        checked += 1
        context = normalise(item["context"] or "")
        spans = [s for s in (answer.source_spans or []) if s]
        if not spans:
            violations.append({"id": item["id"], "fault": "value with no span",
                               "answer": answer.answer[:120]})
            continue
        for span in spans:
            if normalise(span) not in context:
                violations.append({"id": item["id"],
                                   "fault": "span absent from the context",
                                   "span": span[:120],
                                   "answer": answer.answer[:120]})
    return {"answers_checked": checked, "violations": violations,
            "violation_count": len(violations),
            "span_validated_pct": round(
                100.0 * (checked - len(violations)) / max(checked, 1), 2),
            "rule": "100% required; any violation blocks ship"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--items-file", default=None,
                        help="score an arbitrary JSONL item file instead of a "
                             "named split - used for the fresh calculation "
                             "split, which is not a frozen-set split and has no "
                             "manifest entry")
    parser.add_argument("--split", default="selection",
                        choices=("frozen", "selection"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--tag", default=None,
                        help="suffix for the output file, so a new "
                             "configuration never overwrites an earlier run")
    parser.add_argument("--stub", action="store_true",
                        help="wire-check with a scripted model, no download")
    args = parser.parse_args()

    from scripts.eval_heldout import (FROZEN, SELECTION, evaluate, print_summary,
                                      verify_split)

    if args.items_file:
        # A split outside the frozen set: there is no manifest hash to verify
        # against, and that is stated rather than silently skipped.
        with open(args.items_file, encoding="utf-8") as handle:
            items = [json.loads(line) for line in handle if line.strip()]
        if args.limit:
            items = items[:args.limit]
        print(f"items: {len(items)} from {args.items_file}")
        print("integrity: NOT VERIFIED - this file is not a manifested split")
        digest, status = None, "not a manifested split"
        source = args.items_file
    else:
        source = FROZEN if args.split == "frozen" else SELECTION
        # evaluate() verifies the split itself, but only when it loads the
        # items. This script hands it a list (so --limit works), which would
        # skip the check - so it is done here, explicitly, before scoring.
        digest, status = verify_split(source, args.split)
        print(f"{args.split} set: {status}, content_sha256 {digest[:16]}...")
        with open(source, encoding="utf-8") as handle:
            items = [json.loads(line) for line in handle if line.strip()]
        if args.limit:
            items = items[:args.limit]

    if args.stub:
        # Proves the wiring end to end without a 3 GB download: a model that
        # always claims it cannot find anything. Every answer must then be an
        # abstention, and the span audit must find zero violations.
        calls = {"n": 0}

        def llm(prompt):
            calls["n"] += 1
            return '{"found": false, "value": null}'

        label = "tool pipeline (stub model)"
    else:
        licence, needed = LICENCES.get(args.model, (None, 8.0))
        print(f"licence: {licence or 'UNVERIFIED - check before using the result'}")
        free = free_memory_gb()
        print(f"free memory: {free:.1f} GB, this model needs about {needed} GB")
        if free < needed and not args.force:
            sys.exit(
                f"\nREFUSING TO RUN: {free:.1f} GB free, about {needed} GB "
                f"needed. This is an unmet requirement, not a result.\n"
                f"  * run it on Kaggle with training/kaggle/phase4/ (same code)\n"
                f"  * or --stub to check the wiring without the model\n"
                f"  * or --force to try anyway")
        llm, calls = build_llm(args.model, args.max_new_tokens)
        label = f"tool pipeline + {args.model} (0-shot)"
        if args.tag:
            label += f" [{args.tag}]"

    pipeline = ToolPipeline(llm)

    # Answers are kept as objects so the span audit and the component breakdown
    # can see inside them; the scorer only ever sees the answer string.
    produced = []

    def answer_fn(question, context):
        result = pipeline.answer(question, context)
        produced.append(result)
        return result.answer

    slug = (args.model.replace("/", "_") + f"_pipeline_{args.split}"
            if not args.stub else f"stub_pipeline_{args.split}")
    if args.tag:
        slug += f"_{args.tag}"
    out = os.path.join("reports", "heldout", f"{slug}.json")

    started = time.time()
    summary, records = evaluate(answer_fn, items=items, label=label,
                                split=args.split, save_raw=out)
    elapsed = time.time() - started

    summary["seconds"] = round(elapsed, 1)
    summary["seconds_per_item"] = round(elapsed / max(len(records), 1), 2)
    summary["model"] = args.model if not args.stub else "stub"
    summary["shots"] = 0
    summary["llm_calls"] = calls["n"]
    summary["llm_calls_per_item"] = round(calls["n"] / max(len(records), 1), 2)
    summary["span_audit"] = audit_spans(produced, items)
    # evaluate() records this only when it loads the items itself; this script
    # hands it a list, so the digest verified above is recorded here.
    summary["eval_set_integrity"] = {
        "split": args.items_file or args.split,
        "content_sha256": digest, "status": status, "verified_by": __file__}
    summary["components"] = dict(
        Counter(a.component for a in produced).most_common())
    summary["abstained_pct"] = round(
        100.0 * sum(1 for a in produced if a.abstained) / max(len(produced), 1), 2)
    summary["arithmetic_done_by"] = "python (pipeline/tool_pipeline.py OPERATIONS)"
    summary["configuration_tag"] = args.tag
    resolved = [a for a in produced
                if any(o.get("resolved_by_synonym") for o in a.operands.values())]
    summary["operands_resolved_by_synonym"] = {
        "answers": len(resolved),
        "note": "calculations where at least one operand was found under a "
                "caption other than the formula's own name"}

    # The per-item detail is what makes a failure diagnosable later.
    with open(out, encoding="utf-8") as handle:
        payload = json.load(handle)
    payload["summary"] = summary
    for record, produced_answer in zip(payload["records"], produced):
        record["component"] = produced_answer.component
        record["abstained"] = produced_answer.abstained
        record["source_spans"] = produced_answer.source_spans
        record["computed_value"] = produced_answer.value
        record["operation"] = produced_answer.operation
        record["pipeline_reason"] = produced_answer.reason
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    print_summary(summary)
    audit = summary["span_audit"]
    print(f"\ncomponents: {summary['components']}")
    print(f"llm calls: {calls['n']} ({summary['llm_calls_per_item']} per item)")
    print(f"abstained: {summary['abstained_pct']}%")
    print(f"span rule: {audit['span_validated_pct']}% of "
          f"{audit['answers_checked']} stated values carry a validated span")
    if audit["violations"]:
        print(f"\nSPAN RULE VIOLATED {audit['violation_count']}x - this blocks "
              f"ship under EXPERIMENT_RULES_v2 section 1:")
        for violation in audit["violations"][:10]:
            print(f"  {violation['id']}: {violation['fault']}")
    else:
        print("span rule: no violations")
    print(f"\n{summary['seconds_per_item']}s per item, {summary['seconds']}s total")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
