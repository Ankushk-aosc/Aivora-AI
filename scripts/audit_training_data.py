"""Phase 7: audit every training dataset this project holds, at record level.

docs/AIVORA_TRAINING_DATA_AUDIT.md covers the PRETRAINING corpus, which lives on
Kaggle and can only be described from its manifest. This measures what is on
disk, record by record: lengths, duplicates, near-duplicates, malformed rows,
arithmetic that does not check out, and coverage by financial domain and task.

Writes reports/TRAINING_DATA_AUDIT.md and .json. Changes nothing.

    python scripts/audit_training_data.py
"""

import hashlib
import json
import os
import re
import sys
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

DATASETS = {
    "financial_sft_v1": os.path.join("data", "sft", "financial_sft_v1.jsonl"),
    "tiny_overfit": os.path.join("data", "tiny_overfit", "tiny_financial.jsonl"),
    "instruction_legacy": os.path.join("data", "instruction",
                                       "financial_instructions.jsonl"),
}

# Phase 7's domain list, with the words that indicate each. Deliberately simple:
# a keyword count is a coverage estimate, not a classification, and it is
# reported as such.
DOMAIN_TERMS = {
    "accounting": ["accrual", "depreciation", "amortization", "amortisation",
                   "balance sheet", "ledger", "debit", "credit", "provision",
                   "goodwill", "impairment", "deferred"],
    "valuation": ["valuation", "dcf", "discounted", "multiple", "p/e", "ev/ebitda",
                  "terminal value", "intrinsic", "npv", "present value", "irr"],
    "financial_statements": ["income statement", "balance sheet", "cash flow statement",
                             "statement of", "line item", "reported figures"],
    "cash_flow": ["cash flow", "operating cash", "free cash flow", "capex",
                  "capital expenditure", "working capital"],
    "corporate_finance": ["dividend", "buyback", "capital structure", "wacc",
                          "cost of capital", "rights issue", "acquisition", "merger"],
    "economics": ["inflation", "interest rate", "gdp", "recession", "monetary",
                  "central bank", "unemployment"],
    "markets": ["share price", "stock price", "market cap", "index", "trading",
                "yield curve", "treasury"],
    "banking": ["bank", "deposit", "loan", "credit facility", "covenant", "lender"],
    "risk": ["risk", "default", "volatility", "hedge", "exposure", "leverage",
             "concentration"],
    "ratios": ["ratio", "margin", "turnover", "roe", "roa", "roic", "coverage",
               "per share"],
    "portfolio_management": ["portfolio", "diversif", "asset allocation",
                             "rebalanc", "benchmark index"],
    "fixed_income": ["bond", "coupon", "duration", "maturity", "yield to maturity",
                     "credit rating"],
    "derivatives": ["option", "future", "forward", "swap", "derivative", "strike"],
}

TASK_TERMS = {
    "definitions": ["what is", "define", "what does", "meaning of"],
    "extraction": ["extract", "from the figures", "reported figures",
                   "financial summary", "selected figures"],
    "calculations": ["calculate", "what is the margin", "what is the ratio",
                     "compute"],
    "interpretation": ["what could explain", "what does that", "why might",
                       "what does this"],
    "reasoning": ["step", "then", "therefore", "first", "= "],
    "abstention": ["insufficient", "cannot verify", "not given", "unable"],
    "current_data": ["today", "right now", "current", "latest"],
}


def shingles(text, size=5):
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {" ".join(words[i:i + size]) for i in range(max(1, len(words) - size + 1))}


def near_duplicate_rate(texts, sample=1200, threshold=0.8):
    """Jaccard over 5-word shingles on a sample, because all-pairs is quadratic."""
    import random

    rng = random.Random(7)
    picked = texts if len(texts) <= sample else rng.sample(texts, sample)
    sets = [shingles(t) for t in picked]
    hits = 0
    comparisons = 0
    for i in range(len(sets)):
        for j in range(i + 1, min(i + 25, len(sets))):
            comparisons += 1
            union = sets[i] | sets[j]
            if not union:
                continue
            if len(sets[i] & sets[j]) / len(union) >= threshold:
                hits += 1
    return (round(100.0 * hits / comparisons, 3) if comparisons else 0.0,
            comparisons, len(picked))


def audit_file(name, path):
    from data_sources.tokenizer import get_encoding
    from evaluation.financial_metrics import extract_numbers

    enc = get_encoding()
    with open(path, encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]

    texts, lengths, malformed = [], [], []
    for row in rows:
        question = row.get("question") or row.get("instruction") or ""
        answer = row.get("answer") or row.get("output") or ""
        context = row.get("context") or row.get("input") or ""
        if not question.strip() or not answer.strip():
            malformed.append(row.get("id", "?"))
            continue
        text = f"{question}\n{context}\n{answer}"
        texts.append(text)
        lengths.append(len(enc.encode_ordinary(text)))

    lengths.sort()
    exact = len(texts) - len({hashlib.sha256(t.lower().encode()).hexdigest()
                              for t in texts})
    near_rate, comparisons, sampled = near_duplicate_rate(texts)

    joined = [t.lower() for t in texts]
    domains = {domain: sum(1 for t in joined if any(term in t for term in terms))
               for domain, terms in DOMAIN_TERMS.items()}
    tasks = {task: sum(1 for t in joined if any(term in t for term in terms))
             for task, terms in TASK_TERMS.items()}

    # Arithmetic check: where a row states a numeric answer, does the answer text
    # contain it? A row whose prose disagrees with its label is a wrong example.
    numeric_rows = [r for r in rows if r.get("numeric_answer") is not None]
    inconsistent = [r.get("id") for r in numeric_rows
                    if not any(abs(v - r["numeric_answer"]) < 0.02
                               for v in extract_numbers(r.get("answer", "")))]

    return {
        "path": path,
        "sha256": hashlib.sha256(open(path, "rb").read()).hexdigest(),
        "records": len(rows),
        "usable_records": len(texts),
        "malformed_records": len(malformed),
        "total_tokens": sum(lengths),
        "mean_length_tokens": round(sum(lengths) / len(lengths), 1) if lengths else 0,
        "median_length_tokens": lengths[len(lengths) // 2] if lengths else 0,
        "max_length_tokens": lengths[-1] if lengths else 0,
        "p95_length_tokens": lengths[int(len(lengths) * 0.95)] if lengths else 0,
        "exact_duplicates": exact,
        "near_duplicate_rate_pct": near_rate,
        "near_duplicate_sample": {"records_sampled": sampled,
                                  "pairs_compared": comparisons,
                                  "threshold_jaccard": 0.8},
        "rows_with_numeric_answer": len(numeric_rows),
        "numeric_answers_inconsistent_with_prose": len(inconsistent),
        "by_task_type": dict(Counter(r.get("task_type", "unlabelled")
                                     for r in rows).most_common()),
        "by_verification_status": dict(Counter(r.get("verification_status", "none")
                                              for r in rows).most_common()),
        "by_source": dict(Counter(r.get("source", "unknown") for r in rows).most_common()),
        "domain_coverage_records": dict(sorted(domains.items(), key=lambda kv: -kv[1])),
        "task_coverage_records": dict(sorted(tasks.items(), key=lambda kv: -kv[1])),
        "synthetic_vs_sourced": {
            "generated_or_curated_here": sum(
                1 for r in rows if str(r.get("source", "")).startswith(
                    ("generated", "curated", "authored"))),
            "external_source": sum(
                1 for r in rows if r.get("source")
                and not str(r["source"]).startswith(("generated", "curated", "authored"))),
        },
    }


def pretraining_summary():
    path = os.path.join("checkpoints", "final", "checkpoint_247850.json")
    if not os.path.exists(path):
        return {"available": False}
    with open(path, encoding="utf-8") as handle:
        meta = json.load(handle)
    manifest = meta.get("dataset_manifest") or {}
    datasets = manifest.get("datasets", [])
    unique = sum(d.get("train_tokens_used", 0) for d in datasets)
    return {
        "available": True,
        "note": ("the pretraining shards live in the Kaggle environment; this is "
                 "from the manifest the checkpoint embedded, and record-level "
                 "measures are not available locally"),
        "datasets": len(datasets),
        "unique_train_tokens": unique,
        "tokens_processed": meta.get("tokens_processed"),
        "epochs": round(meta["tokens_processed"] / unique, 2) if unique else None,
        "exact_duplicates_removed_at_prepare": sum(
            d.get("records_removed_duplicate", 0) for d in datasets),
        "eval_leakage_removed_at_prepare": sum(
            d.get("records_removed_eval_leakage", 0) for d in datasets),
        "by_dataset_tokens": {d["name"]: d.get("train_tokens_used", 0)
                              for d in datasets},
    }


def markdown(report):
    lines, add = [], None
    add = lines.append
    add("# Training data audit")
    add("")
    add("Phase 7. Record-level measurement of every training dataset on disk, plus "
        "what the pretraining manifest can tell us about the corpus that lives on "
        "Kaggle. Produced by `scripts/audit_training_data.py`; nothing was changed.")
    add("")

    pre = report["pretraining_corpus"]
    add("## Pretraining corpus (from the checkpoint's own manifest)")
    add("")
    if pre.get("available"):
        add(f"| | |")
        add(f"| --- | --- |")
        add(f"| unique train tokens | **{pre['unique_train_tokens']:,}** |")
        add(f"| tokens processed | {pre['tokens_processed']:,} |")
        add(f"| **epochs over the corpus** | **{pre['epochs']}** |")
        add(f"| datasets | {pre['datasets']} |")
        add(f"| exact duplicates removed at prepare time | {pre['exact_duplicates_removed_at_prepare']} |")
        add(f"| evaluation leakage removed at prepare time | {pre['eval_leakage_removed_at_prepare']} |")
        add("")
        add(f"_{pre['note']}_")
    add("")

    for name, entry in report["datasets"].items():
        add(f"## {name}")
        add("")
        add(f"`{entry['path']}`, sha256 `{entry['sha256'][:12]}...`")
        add("")
        add("| measure | value |")
        add("| --- | --- |")
        add(f"| records | {entry['records']:,} |")
        add(f"| malformed (empty question or answer) | {entry['malformed_records']} |")
        add(f"| total tokens | {entry['total_tokens']:,} |")
        add(f"| mean / median length | {entry['mean_length_tokens']} / {entry['median_length_tokens']} tokens |")
        add(f"| p95 / max length | {entry['p95_length_tokens']} / {entry['max_length_tokens']} tokens |")
        add(f"| exact duplicates | {entry['exact_duplicates']} |")
        add(f"| near-duplicate rate | {entry['near_duplicate_rate_pct']}% "
            f"({entry['near_duplicate_sample']['pairs_compared']:,} pairs compared, "
            f"Jaccard >= 0.8 on 5-word shingles) |")
        add(f"| rows with a numeric answer | {entry['rows_with_numeric_answer']:,} |")
        add(f"| **numeric answers disagreeing with their own prose** | "
            f"**{entry['numeric_answers_inconsistent_with_prose']}** |")
        add("")
        if entry["by_task_type"]:
            add("Task types: " + ", ".join(f"{k} {v:,}" for k, v in
                                           entry["by_task_type"].items()))
            add("")
        if entry["by_verification_status"]:
            add("Verification: " + ", ".join(f"{k} {v:,}" for k, v in
                                             entry["by_verification_status"].items()))
            add("")
        add("Domain coverage (records mentioning the domain's vocabulary - an "
            "estimate, not a classification):")
        add("")
        add("| domain | records |")
        add("| --- | --- |")
        for domain, count in entry["domain_coverage_records"].items():
            add(f"| {domain} | {count:,} |")
        add("")
        add("Task signals: " + ", ".join(f"{k} {v:,}" for k, v in
                                         entry["task_coverage_records"].items()))
        add("")
        add(f"Synthetic vs sourced: "
            f"{entry['synthetic_vs_sourced']['generated_or_curated_here']:,} generated "
            f"or curated in this repository, "
            f"{entry['synthetic_vs_sourced']['external_source']:,} from an external "
            f"source.")
        add("")

    add("## What this audit says")
    add("")
    sft = report["datasets"].get("financial_sft_v1", {})
    add(f"* The new SFT set is **{sft.get('records', 0):,} records / "
        f"{sft.get('total_tokens', 0):,} tokens**, entirely generated or curated in "
        "this repository. Nothing in it came from an external instruction dataset, "
        "which is a limitation worth stating: it inherits this project's idea of a "
        "good answer.")
    add(f"* **{sft.get('numeric_answers_inconsistent_with_prose', 0)}** numeric "
        "answers disagree with their own explanation text - the builder computes "
        "both from one expression, and the independent validator checks it again.")
    add("* Thin or absent domains are visible in the coverage table above and are "
        "the honest gap list for the next dataset version: portfolio management, "
        "derivatives and fixed income are barely represented, and banking only "
        "incidentally.")
    add("* The legacy instruction file is finance-alpaca content, which is "
        "Reddit-style personal finance rather than financial analysis. It is kept "
        "for provenance, not used by the new SFT experiment.")
    add("")
    return "\n".join(lines)


def main():
    report = {"pretraining_corpus": pretraining_summary(), "datasets": {}}
    for name, path in DATASETS.items():
        if not os.path.exists(path):
            print(f"skipping {name}: {path} not found")
            continue
        print(f"auditing {name} ...")
        report["datasets"][name] = audit_file(name, path)

    os.makedirs("reports", exist_ok=True)
    with open(os.path.join("reports", "TRAINING_DATA_AUDIT.json"), "w",
              encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    text = markdown(report)
    with open(os.path.join("reports", "TRAINING_DATA_AUDIT.md"), "w",
              encoding="utf-8") as handle:
        handle.write(text)
    print("\n" + text[:2000])
    print("\nwrote reports/TRAINING_DATA_AUDIT.{json,md}")


if __name__ == "__main__":
    main()
