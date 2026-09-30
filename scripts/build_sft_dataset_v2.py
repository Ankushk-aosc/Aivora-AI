"""SFT_002 dataset: built to fix each measured cause of the SFT_001 failure.

SFT_001 destroyed the base model's copy-from-context ability (5/6 -> 0/6) and
left extraction at 0%. The causes were measured, not guessed, and each one has a
corresponding change here:

  measured cause                             fix in this file
  -----------------------------------------  --------------------------------
  extraction: 30% of rows, 10% of the loss   balance by ANSWER TOKENS, with a
  (7.5-token answers vs 41 for definitions)  target share per task; extraction
                                             answers are padded with a short
                                             citation so they carry real weight
  956 rows shared 13 answer strings          every canned sentence is replaced
  (interpretation 6, abstention 7), which    by a template bank; the builder
  made emitting them the cheapest loss       REFUSES to emit a dataset whose
  reduction available                        per-task distinct-answer ratio is
                                             below a floor
  copying was not supported by any row       a replay slice of plain financial
                                             text, plus explicit copy rows with
                                             no finance at all
  memorisable positions                      the asked field's position, the
                                             label spelling, the separator and
                                             the scale all vary; distractor
                                             numbers are always present

The builder fails loudly rather than producing a dataset with the old defect:
run it and it either meets the entropy and token-balance floors or it raises.

    python scripts/build_sft_dataset_v2.py
"""

import argparse
import hashlib
import json
import os
import random
import sys
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts.build_sft_dataset import (  # noqa: E402
    CALCULATIONS, CALCULATOR_EQUIVALENT, DOMAIN_OF_FIELD, FIELD_LABELS,
    load_benchmark_questions, money, render,
)

OUT_DEFAULT = os.path.join("data", "sft", "financial_sft_v2.jsonl")
SEED = 20260930

# Target share of ANSWER TOKENS per task type. SFT_001's failure was that these
# shares emerged by accident from row counts; here they are the specification.
TOKEN_TARGETS = {
    "extraction": 0.38,      # was 10.0% - the task that did not learn
    "calculation": 0.24,
    "reasoning": 0.10,
    "interpretation": 0.08,
    "abstention": 0.08,
    "copy": 0.05,            # new: copying with no finance involved
    "grounded": 0.04,
    # A definition has ONE right answer, so this task cannot be made
    # high-entropy: 123 glossary terms are 123 possible targets however many ways
    # the question is phrased. The lesson from SFT_001 is that a low-entropy task
    # must therefore hold a SMALL share of the gradient, not that its answers
    # should be padded into artificial variety.
    "definition": 0.02,
    "replay": 0.02,          # plain financial text continuation
    "explanation": 0.01,
}

# A task whose answers repeat more than this is a shortcut waiting to be found.
MIN_DISTINCT_ANSWER_RATIO = 0.55

# ---------------------------------------------------------------- extraction

CITATION_TEMPLATES = [
    "{value} ({label} in the figures provided).",
    "{value}. That is the {label} line.",
    "{label} is {value}.",
    "The figures give {label} as {value}.",
    "{value}, read from the {label} line.",
    "{value} - the {label} entry.",
    "Taking the {label} line: {value}.",
    "{label}: {value}, as stated.",
]

QUESTION_TEMPLATES = [
    "What is {label_l}?", "What was {label_l}?", "What is the {label_l} figure?",
    "State {label_l} from the figures above.", "How much is {label_l}?",
    "Give the {label_l} value.", "What does the statement show for {label_l}?",
    "Read off {label_l}.",
]

HEADERS = ["Financial summary:", "Selected figures:", "Extract from the accounts:",
           "Reported figures:", "Financial data (unaudited):", "Summary of results:",
           "Key figures (in $000s):", "From the filing:", "Statement extract:"]


def extraction_rows_v2(rng, count):
    """Every axis a model could memorise is varied, and the answer carries a
    citation so the row is not 7 tokens of gradient."""
    rows = []
    field_names = list(FIELD_LABELS)
    for index in range(count):
        # 5-11 fields: always several distractor numbers, never a fixed position.
        chosen = rng.sample(field_names, rng.randint(5, 11))
        rng.shuffle(chosen)
        currency = rng.choice(["$", "", "£", "€"])
        scale = rng.choice(["", "", "m", " million", "k", " thousand"])
        separator = rng.choice([": ", " = ", "  ", " ", ": ", " -- "])
        indent = rng.choice(["", "  ", "\t"])
        values, labels = {}, {}
        for field in chosen:
            # Distinct values, so no two lines share a number and the model
            # cannot be right by picking any figure.
            while True:
                candidate = money(rng, 10, 9900)
                if candidate not in values.values():
                    break
            values[field] = candidate
            labels[field] = rng.choice(FIELD_LABELS[field])
        lines = [f"{indent}{labels[f]}{separator}{currency}{values[f]:,.2f}{scale}"
                 for f in chosen]
        context = rng.choice(HEADERS) + "\n" + "\n".join(lines)

        asked = rng.choice(chosen)
        label = labels[asked]
        value_str = f"{currency}{values[asked]:,.2f}{scale}"
        question = rng.choice(QUESTION_TEMPLATES).format(label_l=label.lower())
        answer = rng.choice(CITATION_TEMPLATES).format(value=value_str, label=label)

        if value_str not in context or value_str not in answer:
            continue
        rows.append({
            "id": f"v2_extract_{index:05d}",
            "task_type": "extraction",
            "domain": DOMAIN_OF_FIELD.get(asked, "statements"),
            "difficulty": 1 if len(chosen) <= 7 else 2,
            "question": question, "context": context, "answer": answer,
            "numeric_answer": values[asked],
            "source": "generated_statement_v2",
            "verification_status": "verified_present_in_context",
            "verification_detail": (f"field={asked}; {len(chosen) - 1} distractor "
                                    f"figures; position {chosen.index(asked) + 1} "
                                    f"of {len(chosen)}"),
        })
    return rows


# ---------------------------------------------------------------------- copy

COPY_SUBJECTS = ["access code", "reference number", "batch id", "ticket number",
                 "serial number", "confirmation code", "tracking id", "case number",
                 "invoice number", "policy number"]
COPY_NOISE = ["The office is closed on Friday.", "Filed by the accounts team.",
              "See appendix B for details.", "Received in the afternoon post.",
              "Logged by the duty officer.", "Copied to the regional office."]


def copy_rows(rng, count):
    """Copying with no finance in it at all.

    SFT_001 destroyed this, and nothing in that dataset supported it. These rows
    exist purely so the behaviour has gradient support of its own.
    """
    alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
    rows = []
    for index in range(count):
        subject = rng.choice(COPY_SUBJECTS)
        code = "".join(rng.choice(alphabet) for _ in range(rng.randint(3, 6)))
        others = [(rng.choice([s for s in COPY_SUBJECTS if s != subject]),
                   "".join(rng.choice(alphabet) for _ in range(rng.randint(3, 6))))
                  for _ in range(rng.randint(1, 3))]
        lines = [f"The {subject} is {code}."]
        lines += [f"The {other} is {value}." for other, value in others]
        lines.append(rng.choice(COPY_NOISE))
        rng.shuffle(lines)
        context = " ".join(lines)
        rows.append({
            "id": f"v2_copy_{index:05d}",
            "task_type": "copy", "domain": "general", "difficulty": 1,
            "question": f"What is the {subject}?",
            "context": context,
            "answer": f"The {subject} is {code}.",
            "source": "generated_copy",
            "verification_status": "verified_present_in_context",
            "verification_detail": f"{len(others)} distractor codes",
        })
    return rows


# -------------------------------------------------------------------- replay



def replay_rows(rng, count):
    """Plain continuation of REAL financial prose, sampled from the corpus.

    This is the replay slice: it keeps ordinary language modelling in the
    gradient so SFT does not overwrite general behaviour, which is what SFT_001
    did to the copy ability. An earlier version of this function used six
    authored sentences, which is not a replay slice - it is six more canned
    targets, the very defect being fixed.

    Windows are decoded from the local shards, so the text is the same
    distribution the model was pretrained on.
    """
    import glob

    import numpy as np

    from data_sources.shard_writer import load_shard_index
    from data_sources.tokenizer import get_encoding

    enc = get_encoding()
    rows, sources = [], []
    for index_path in sorted(glob.glob(os.path.join("data", "shards", "*",
                                                    "train", "index.json"))):
        shard_dir = os.path.dirname(index_path)
        dataset = os.path.basename(os.path.dirname(shard_dir))
        # Financial text only: replay should reinforce the domain, not TinyStories.
        if not dataset.startswith("financial"):
            continue
        for shard in load_shard_index(shard_dir)["shards"]:
            sources.append((dataset, os.path.join(shard_dir, shard["file"]),
                            shard["tokens"]))
    if not sources:
        print("  replay: no local financial shards found - slice will be empty")
        return []

    for index in range(count):
        dataset, path, total = sources[index % len(sources)]
        window = rng.randint(90, 150)
        if total <= window + 2:
            continue
        start = rng.randrange(0, total - window - 1)
        tokens = np.memmap(path, dtype=np.uint16, mode="r")[start:start + window]
        text = enc.decode(tokens.tolist()).strip()
        if len(text) < 120:
            continue
        rows.append({
            "id": f"v2_replay_{index:05d}",
            "task_type": "replay", "domain": "concepts", "difficulty": 1,
            "question": "", "context": "",
            "answer": text,
            "source": f"corpus_replay:{dataset}",
            "verification_status": "sourced_corpus_text",
            "verification_detail": (f"{window} tokens from {os.path.basename(path)} "
                                    f"at offset {start}; trained without a prompt "
                                    f"template"),
        })
    return rows


# ------------------------------------------- high-entropy abstention/interp

ABSTAIN_OPENERS = [
    "That cannot be determined from what is given:",
    "There is not enough information here:",
    "This one cannot be answered as stated:",
    "The figures provided are insufficient:",
    "Not determinable from these figures:",
    "That calculation cannot be completed:",
    "I cannot work that out from this:",
    "The data given does not support that:",
]
ABSTAIN_CLOSERS = [
    "so no figure is given.", "so I will not guess at it.",
    "and guessing would be worse than saying so.",
    "so the answer has to be left open.", "which is why no number follows.",
    "so nothing is asserted here.",
]


def abstention_rows_v2(rng, count):
    """Same behaviour as v1, but no two rows share a sentence.

    The opener, the naming of what is missing, and the closer are all sampled, so
    the low-loss shortcut of memorising one refusal sentence does not exist.
    """
    missing = [
        ("the net profit margin", "net income", "revenue"),
        ("ROE", "net income", "shareholders' equity"),
        ("the current ratio", "current assets", "current liabilities"),
        ("EPS", "net income", "the share count"),
        ("free cash flow", "operating cash flow", "capital expenditure"),
        ("the debt-to-equity ratio", "total debt", "equity"),
        ("the EBITDA margin", "EBITDA", "revenue"),
        ("asset turnover", "revenue", "total assets"),
        ("interest coverage", "EBIT", "interest expense"),
        ("the quick ratio", "current assets and inventory", "current liabilities"),
        ("the payout ratio", "dividends", "net income"),
        ("ROIC", "NOPAT", "invested capital"),
    ]
    current = [
        ("the Federal Reserve policy rate", "today"),
        ("Apple's share price", "right now"),
        ("the inflation rate", "this month"),
        ("the S&P 500 level", "today"),
        ("the 10-year Treasury yield", "currently"),
        ("the price of gold", "right now"),
        ("Microsoft's quarterly earnings", "this quarter"),
        ("the unemployment rate", "this month"),
    ]
    rows = []
    for index in range(count):
        if index % 3 < 2:
            metric, supplied, withheld = missing[index % len(missing)]
            value = money(rng)
            opener = rng.choice(ABSTAIN_OPENERS)
            closer = rng.choice(ABSTAIN_CLOSERS)
            phrasing = rng.choice([
                f"{metric} needs {withheld} as well, which is not here",
                f"{withheld} is missing, and {metric} cannot be computed without it",
                f"without {withheld} there is no way to reach {metric}",
                f"{metric} requires both figures; only {supplied} is given",
            ])
            rows.append({
                "id": f"v2_abstain_{index:05d}",
                "task_type": "abstention", "domain": "ratios", "difficulty": 3,
                "question": f"{supplied.capitalize()} is {value:,.2f}. "
                            f"What is {metric}?",
                "context": "",
                "answer": f"{opener} {phrasing}, {closer}",
                "source": "generated_verified_v2",
                "verification_status": "verified_insufficient",
                "verification_detail": f"withheld={withheld}",
            })
        else:
            subject, when = current[(index // 3) % len(current)]
            opener = rng.choice([
                "I cannot verify that from the available data:",
                "That is not something I can confirm here:",
                "No verified source for that is connected:",
                "I have no way to check that figure:",
            ])
            closer = rng.choice([
                "so no figure is given here.",
                "and a stale number would be worse than none.",
                "so it needs a live source rather than a guess.",
                "so I will not state a value.",
            ])
            rows.append({
                "id": f"v2_abstain_{index:05d}",
                "task_type": "abstention", "domain": "markets", "difficulty": 3,
                "question": f"What is {subject} {when}?",
                "context": "",
                "answer": f"{opener} {subject} changes over time, {closer}",
                "source": "generated_verified_v2",
                "verification_status": "verified_current_data",
                "verification_detail": "requires a live source",
            })
    return rows


INTERP_CASES = [
    ("Revenue rose {a:.1f}% while gross margin fell {b:.1f} points.",
     ["input costs rising faster than prices", "discounting used to win volume",
      "a sales mix shifting toward lower-margin products",
      "a one-off cost landing in cost of sales"]),
    ("Operating cash flow fell {b:.1f}% while net income rose {a:.1f}%.",
     ["receivables growing faster than sales", "inventory being built ahead of demand",
      "payables being settled earlier than before",
      "profit carried by non-cash or one-off items"]),
    ("EBITDA rose {a:.1f}% while EBIT fell {b:.1f}%.",
     ["depreciation rising after heavy capital expenditure",
      "amortization of intangibles from an acquisition",
      "an impairment charge landing below EBITDA"]),
    ("Total debt rose {a:.1f}% while EBITDA was flat.",
     ["borrowing used to fund an acquisition",
      "capital expenditure funded by debt rather than cash flow",
      "distributions paid out of borrowing"]),
    ("Inventory rose {a:.1f}% while sales were flat.",
     ["demand weakening after the stock was ordered",
      "over-ordering against an optimistic forecast",
      "a deliberate build ahead of a launch"]),
    ("Days sales outstanding rose from {c:.0f} to {d:.0f}.",
     ["credit terms loosened to support sales",
      "collections slipping on existing accounts",
      "revenue recognised ahead of cash collection"]),
    ("Return on equity rose while margins were unchanged.",
     ["buybacks shrinking the equity base", "more leverage in the capital structure",
      "asset disposals reducing the denominator"]),
    ("Free cash flow turned negative while profit grew.",
     ["capital expenditure stepping up", "working capital absorbing cash",
      "tax paid catching up with reported profit"]),
]
INTERP_FRAMES = [
    "{fact} Possible explanations include {causes}. The figures given do not "
    "settle which applies.",
    "{fact} That pattern is usually caused by {causes} - which of those it is "
    "cannot be told from these numbers alone.",
    "{fact} Candidates would be {causes}. Nothing here identifies the cause.",
    "{fact} It could be {causes}; the statement does not say which.",
    "{fact} The usual reasons are {causes}, and more detail would be needed to "
    "choose between them.",
]


def interpretation_rows_v2(rng, count):
    """Each answer samples its frame and a different subset of causes, so no
    sentence recurs often enough to be worth memorising."""
    rows = []
    for index in range(count):
        template, causes = INTERP_CASES[index % len(INTERP_CASES)]
        numbers = {"a": rng.uniform(4, 40), "b": rng.uniform(2, 25),
                   "c": rng.randrange(25, 45), "d": rng.randrange(55, 95)}
        observation = template.format(**numbers)
        picked = rng.sample(causes, min(len(causes), rng.randint(2, 3)))
        joined = ", ".join(picked[:-1]) + (" or " + picked[-1] if len(picked) > 1
                                           else picked[0])
        answer = rng.choice(INTERP_FRAMES).format(fact=observation, causes=joined)
        rows.append({
            "id": f"v2_interp_{index:05d}",
            "task_type": "interpretation", "domain": "interpretation",
            "difficulty": 4,
            "question": f"{observation} What could explain this?",
            "context": "", "answer": answer,
            "source": "generated_from_pattern_v2",
            "verification_status": "verified_pattern",
            "verification_detail": f"{len(picked)} causes offered, none asserted",
        })
    return rows


# ---------------------------------------------------------------- assembly

def answer_tokens(enc, row):
    return len(enc.encode_ordinary(" " + row["answer"])) + 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=OUT_DEFAULT)
    parser.add_argument("--target-tokens", type=int, default=260_000,
                        help="total answer tokens to aim for")
    args = parser.parse_args()

    from data_sources.tokenizer import get_encoding

    from scripts.build_sft_dataset import (
        calculation_rows, curated_rows, grounded_rows, reasoning_rows,
    )

    enc = get_encoding()
    rng = random.Random(SEED)
    print("building v2, balanced by answer tokens...\n")

    # Generate generously, then trim each task to its token target.
    pools = {
        "extraction": extraction_rows_v2(rng, 6000),
        "copy": copy_rows(rng, 1200),
        "replay": replay_rows(rng, 900),
        "abstention": abstention_rows_v2(rng, 1400),
        "interpretation": interpretation_rows_v2(rng, 1400),
        "reasoning": reasoning_rows(rng, 900),
        "grounded": grounded_rows(rng, 700),
    }
    calc, mismatches = calculation_rows(rng, 3000)
    pools["calculation"] = calc
    curated = curated_rows()
    # One row per distinct definition: the five phrasings in v1 shared a single
    # answer each, which is exactly the repetition the entropy floor rejects.
    # Keeping one phrasing per term still avoids the benchmark's wording.
    definitions, seen_answers = [], set()
    for row in curated:
        if row["task_type"] != "definition" or row["answer"] in seen_answers:
            continue
        seen_answers.add(row["answer"])
        definitions.append(row)
    pools["definition"] = definitions
    pools["explanation"] = [r for r in curated if r["task_type"] == "explanation"]
    print(f"calculator cross-check rejected {mismatches} rows")

    benchmark = load_benchmark_questions()
    kept, seen, duplicates, leaked = [], set(), 0, 0
    report_rows = {}

    for task, pool in pools.items():
        target = int(args.target_tokens * TOKEN_TARGETS.get(task, 0.01))
        rng.shuffle(pool)
        used, tokens = [], 0
        for row in pool:
            if tokens >= target:
                break
            # The answer is part of the key: replay rows have no question and
            # no context, so keying on those alone collapsed the whole slice to
            # a single row.
            key = hashlib.sha256((row["question"] + "||" + row.get("context", "") +
                                  "||" + row["answer"]).lower().encode()).hexdigest()
            if key in seen:
                duplicates += 1
                continue
            if row["question"].strip().lower() in benchmark:
                leaked += 1
                continue
            seen.add(key)
            tokens += answer_tokens(enc, row)
            used.append(row)
        kept.extend(used)
        report_rows[task] = {"rows": len(used), "answer_tokens": tokens,
                             "target_tokens": target,
                             "distinct_answers": len({r["answer"] for r in used})}

    # --- the floors that make the SFT_001 failure impossible to repeat -----
    total_tokens = sum(v["answer_tokens"] for v in report_rows.values())
    problems = []
    for task, stats in report_rows.items():
        if not stats["rows"]:
            continue
        ratio = stats["distinct_answers"] / stats["rows"]
        share = stats["answer_tokens"] / total_tokens
        stats["distinct_ratio"] = round(ratio, 3)
        stats["token_share"] = round(100 * share, 2)
        if ratio < MIN_DISTINCT_ANSWER_RATIO:
            problems.append(f"{task}: only {stats['distinct_answers']} distinct "
                            f"answers across {stats['rows']} rows "
                            f"(ratio {ratio:.2f} < {MIN_DISTINCT_ANSWER_RATIO}) - "
                            f"this is the SFT_001 shortcut")
    extraction_share = report_rows.get("extraction", {}).get("token_share", 0)
    if extraction_share < 25:
        problems.append(f"extraction holds only {extraction_share}% of answer "
                        f"tokens; SFT_001 failed at 10% and the target is 34%")

    if problems:
        print("\nREFUSING TO WRITE THE DATASET:")
        for problem in problems:
            print(f"  - {problem}")
        sys.exit(1)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        for row in kept:
            handle.write(json.dumps(row) + "\n")

    manifest = {
        "path": args.out, "version": "financial_sft_v2",
        "sha256": hashlib.sha256(open(args.out, "rb").read()).hexdigest(),
        "seed": SEED, "examples": len(kept),
        "total_answer_tokens": total_tokens,
        "duplicates_dropped": duplicates,
        "benchmark_collisions_dropped": leaked,
        "calculator_mismatches_rejected": mismatches,
        "per_task": report_rows,
        "token_targets": TOKEN_TARGETS,
        "min_distinct_answer_ratio": MIN_DISTINCT_ANSWER_RATIO,
        "fixes_applied": [
            "balanced by answer tokens rather than row count",
            "extraction token share raised from 10% to ~34%",
            "extraction answers carry a citation so they are not 7 tokens",
            "5-11 fields per context with distinct values: distractors always present",
            "asked field position, label spelling, separator, indent, currency and "
            "scale all randomised",
            "abstention and interpretation answers assembled from sampled parts, so "
            "no sentence is worth memorising",
            "copy rows with no finance in them, to support the behaviour SFT_001 "
            "destroyed",
            "replay slice of plain financial prose, trained without a template",
            "builder refuses to write a dataset that repeats the SFT_001 defect",
        ],
    }
    manifest_path = args.out.replace(".jsonl", "_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    print(f"\n{'task':<16}{'rows':>7}{'ans tokens':>12}{'share':>9}"
          f"{'distinct':>10}{'ratio':>8}")
    for task, stats in sorted(report_rows.items(), key=lambda kv: -kv[1]["answer_tokens"]):
        print(f"{task:<16}{stats['rows']:>7}{stats['answer_tokens']:>12,}"
              f"{stats['token_share']:>8.1f}%{stats['distinct_answers']:>10}"
              f"{stats.get('distinct_ratio', 0):>8.2f}")
    print(f"\n{len(kept):,} rows, {total_tokens:,} answer tokens")
    print(f"dropped: {duplicates} duplicates, {leaked} benchmark collisions")
    print(f"\nwrote {args.out}\nwrote {manifest_path}")


if __name__ == "__main__":
    main()
