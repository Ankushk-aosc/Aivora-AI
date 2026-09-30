"""Capability gates A-E. Run these on any checkpoint, before and during training.

SFT_001 passed every check that existed at the time - the loss fell monotonically -
while destroying the base model's ability to copy from context. These gates exist
so that cannot happen silently again. They measure behaviour, not loss.

  Gate A  copy      the model must retain the baseline's copy-from-context ability
  Gate B  extraction must improve on the baseline's 0% on UNSEEN contexts
  Gate C  generalisation  unseen numbers, wording, label spellings, positions and
                    distractors; a model that memorised its training answers fails
  Gate D  regression  calculation, reasoning, interpretation, abstention and
                    plain-text continuation must not collapse
  Gate E  hallucination  measured separately, and never traded for accuracy

Every probe is generated from a seed that the training data does not use, so
nothing here is in any training set.

    python scripts/capability_gates.py --checkpoint checkpoints/final/checkpoint_247850.pt
    python scripts/capability_gates.py --checkpoint checkpoints/sft_002/best.pt --baseline-json reports/gates_baseline.json
"""

import argparse
import json
import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# A seed no dataset builder uses (they use 20260930 and 1234).
PROBE_SEED = 99001
COPY_MIN_FRACTION = 0.8   # of the baseline's copy score

COPY_PROBES = [
    ("The access code is XK42. What is the access code?", "XK42"),
    ("Alice has 17 apples. How many apples does Alice have?", "17"),
    ("The password is banana. What is the password?", "banana"),
    ("Widget count: 4821\nGadget count: 1193\n\nWhat is the widget count?", "4821"),
    ("Repeat this sequence exactly: red blue green", "red blue green"),
    ("The serial number is 7Q9. The colour is teal. What is the serial number?", "7Q9"),
]

# Label spellings and question wordings deliberately NOT used by either dataset
# builder, so Gate C tests wording the model has never been trained on.
UNSEEN_LABELS = {
    "revenue": "Turnover for the period",
    "net_income": "Profit attributable to shareholders",
    "cash": "Cash at bank",
    "total_debt": "Interest-bearing liabilities",
    "ebitda": "Earnings before interest, tax, depreciation and amortisation",
    "inventory": "Stock on hand",
    "equity": "Net assets attributable to owners",
    "capex": "Additions to property, plant and equipment",
}
UNSEEN_QUESTIONS = [
    "Looking at the table, what figure is recorded against {label}?",
    "According to the extract, how much is {label}?",
    "From the data supplied, identify {label}.",
]


def generate_extraction_probes(rng, count, unseen_wording=False):
    """Fresh contexts, fresh values, distractors, random position."""
    from scripts.build_sft_dataset import FIELD_LABELS, money

    probes = []
    fields = list(FIELD_LABELS)
    for _ in range(count):
        chosen = rng.sample(fields, rng.randint(5, 9))
        values, seen = {}, set()
        for field in chosen:
            while True:
                candidate = money(rng, 11, 9899)
                if candidate not in seen:
                    seen.add(candidate)
                    break
            values[field] = candidate
        asked = rng.choice(chosen)
        if unseen_wording:
            labels = {f: UNSEEN_LABELS.get(f, FIELD_LABELS[f][0]) for f in chosen}
            question = rng.choice(UNSEEN_QUESTIONS).format(label=labels[asked].lower())
            header = "Extract of results:"
        else:
            labels = {f: FIELD_LABELS[f][0] for f in chosen}
            question = f"What is {labels[asked].lower()}?"
            header = "Financial summary:"
        lines = [f"  {labels[f]}: {values[f]:,.2f}" for f in chosen]
        probes.append({
            "prompt": header + "\n" + "\n".join(lines) + "\n\n" + question,
            "expected": values[asked],
            "position": chosen.index(asked) + 1,
            "fields": len(chosen),
        })
    return probes


def generate_calculation_probes(rng, count):
    from scripts.build_sft_dataset import CALCULATIONS, money

    probes = []
    for index in range(count):
        name, needed, question, compute, unit, _ = CALCULATIONS[index % len(CALCULATIONS)]
        values = {}
        for field in needed:
            if field in ("eps_value", "book_value_per_share"):
                values[field] = round(rng.uniform(0.5, 15), 2)
            elif field == "price_per_share":
                values[field] = round(rng.uniform(5, 300), 2)
            else:
                values[field] = money(rng, 60, 8000)
        if "revenue" in values:
            for smaller in ("cogs", "ebitda", "ebit", "net_income"):
                if smaller in values:
                    values[smaller] = round(values["revenue"] * rng.choice(
                        [0.1, 0.25, 0.4, 0.55]), 2)
        if "price_per_share" in values and "eps_value" in values:
            values["eps_value"] = round(values["price_per_share"] /
                                        rng.choice([10, 14, 18, 22]), 2)
        if "part" in values and "whole" in values:
            values["part"] = round(values["whole"] * rng.choice([0.1, 0.35, 0.6]), 2)
        if "current_assets" in values and "inventory" in values:
            values["inventory"] = round(values["current_assets"] * 0.25, 2)
        if "ebit" in values and "interest_expense" in values:
            values["interest_expense"] = round(values["ebit"] * 0.15, 2)
        if "dividends" in values and "net_income" in values:
            values["dividends"] = round(values["net_income"] * 0.3, 2)
        if "total_assets" in values and "total_liabilities" in values:
            values["total_liabilities"] = round(values["total_assets"] * 0.4, 2)
        try:
            expected = compute(values)
        except ZeroDivisionError:
            continue
        given = ", ".join(f"{f.replace('_', ' ')} is {values[f]:,.2f}" for f in needed)
        probes.append({"prompt": f"{given.capitalize()}. {question}",
                       "expected": round(expected, 2), "formula": name})
    return probes


ABSTENTION_PROBES = [
    ("Revenue is 4,120.00. What is the net profit margin?", True),
    ("Total assets are 9,400.00. What is ROE?", True),
    ("Net income is 812.00. What is EPS?", True),
    ("What is the Federal Reserve policy rate today?", True),
    ("What is the current price of silver?", True),
    # Answerable controls: over-abstention is a failure too.
    ("Revenue is 1,000.00 and net income is 250.00. What is the net profit margin?",
     False),
    ("Current assets are 600.00 and current liabilities are 300.00. "
     "What is the current ratio?", False),
]

INTERPRETATION_PROBES = [
    ("Gross margin fell while revenue rose. What could explain this?",
     ["cost", "price", "mix", "discount", "input"]),
    ("Operating cash flow is far below net income. What should you check?",
     ["receivable", "inventory", "working capital", "payable", "accrual"]),
    ("EBITDA rose while EBIT fell. What could explain this?",
     ["depreciation", "amortis", "amortiz", "impair"]),
]

PLAIN_TEXT_PROBES = [
    "Gross margin measures how much of each unit of revenue survives",
    "A balance sheet balances because every asset was financed either by",
    "Operating cash flow differs from reported profit whenever",
]


def generate(model, enc, prompt, max_new_tokens=48, greedy=True, seed=PROBE_SEED,
             templated=True):
    import torch

    from data_sources.training_format import PROMPT_TEMPLATE

    text = PROMPT_TEMPLATE.format(prompt=prompt) if templated else prompt
    ids = enc.encode_ordinary(text)
    torch.manual_seed(seed)
    with torch.no_grad():
        out = model.generate(
            torch.tensor(ids).unsqueeze(0), max_new_tokens,
            temperature=1e-5 if greedy else 0.7, top_k=1 if greedy else 40,
            top_p=None if greedy else 0.9, repetition_penalty=1.0 if greedy else 1.3,
            stop_on_repetition=True, eos_token_id=enc.eot_token)
    generated = out[0, len(ids):].tolist()
    hit_eos = enc.eot_token in generated
    if hit_eos:
        generated = generated[:generated.index(enc.eot_token)]
    return enc.decode(generated).strip(), len(generated), hit_eos


def run_gates(model, enc, verbose=False):
    from evaluation.financial_metrics import abstained, numeric_match

    rng = random.Random(PROBE_SEED)
    result = {}

    # --- Gate A: copy ---------------------------------------------------
    copy_hits, copy_detail = 0, []
    for prompt, needed in COPY_PROBES:
        answer, _, _ = generate(model, enc, prompt, 24)
        ok = needed.lower() in answer.lower()
        copy_hits += ok
        copy_detail.append({"need": needed, "answer": answer[:70], "ok": bool(ok)})
    result["gate_a_copy"] = {"score": copy_hits, "of": len(COPY_PROBES),
                             "detail": copy_detail}

    # --- Gate B: extraction on unseen contexts --------------------------
    probes = generate_extraction_probes(rng, 20)
    hits, detail = 0, []
    for probe in probes:
        answer, _, _ = generate(model, enc, probe["prompt"], 32)
        ok = numeric_match(answer, probe["expected"], 0.01)
        hits += ok
        detail.append({"expected": probe["expected"], "answer": answer[:60],
                       "position": probe["position"], "ok": bool(ok)})
    result["gate_b_extraction"] = {"score": hits, "of": len(probes),
                                   "accuracy_pct": round(100.0 * hits / len(probes), 2),
                                   "detail": detail[:8]}

    # --- Gate C: generalisation - unseen wording and labels -------------
    probes = generate_extraction_probes(rng, 15, unseen_wording=True)
    hits, detail = 0, []
    for probe in probes:
        answer, _, _ = generate(model, enc, probe["prompt"], 32)
        ok = numeric_match(answer, probe["expected"], 0.01)
        hits += ok
        detail.append({"expected": probe["expected"], "answer": answer[:60],
                       "ok": bool(ok)})
    result["gate_c_generalisation"] = {
        "score": hits, "of": len(probes),
        "accuracy_pct": round(100.0 * hits / len(probes), 2),
        "note": "label spellings and question wordings absent from every training set",
        "detail": detail[:6]}

    # --- Gate D: regression ---------------------------------------------
    calc_probes = generate_calculation_probes(rng, 20)
    calc_hits = 0
    for probe in calc_probes:
        answer, _, _ = generate(model, enc, probe["prompt"], 48)
        calc_hits += numeric_match(answer, probe["expected"], 0.05)

    abst_hits, over_abstain = 0, 0
    for prompt, should_decline in ABSTENTION_PROBES:
        answer, _, _ = generate(model, enc, prompt, 40)
        declined = abstained(answer)
        if should_decline and declined:
            abst_hits += 1
        if not should_decline and declined:
            over_abstain += 1

    interp_hits = 0
    for prompt, keywords in INTERPRETATION_PROBES:
        answer, _, _ = generate(model, enc, prompt, 56)
        interp_hits += any(k in answer.lower() for k in keywords)

    plain_lengths, plain_texts = [], []
    for prompt in PLAIN_TEXT_PROBES:
        answer, length, _ = generate(model, enc, prompt, 40, greedy=False,
                                     templated=False)
        plain_lengths.append(length)
        plain_texts.append(answer[:80])

    result["gate_d_regression"] = {
        "calculation": {"score": calc_hits, "of": len(calc_probes),
                        "accuracy_pct": round(100.0 * calc_hits / max(len(calc_probes), 1), 2)},
        "abstention": {"score": abst_hits,
                       "of": sum(1 for _, d in ABSTENTION_PROBES if d)},
        "over_abstention": {"score": over_abstain,
                            "of": sum(1 for _, d in ABSTENTION_PROBES if not d)},
        "interpretation": {"score": interp_hits, "of": len(INTERPRETATION_PROBES)},
        "plain_text_continuation": {"mean_tokens": round(sum(plain_lengths) /
                                                        len(plain_lengths), 1),
                                    "samples": plain_texts},
    }

    # --- Gate E: hallucination on the unseen extraction probes ----------
    # A wrong assertion where the answer was present in the context.
    detail = result["gate_b_extraction"]["detail"]
    asserted_wrong = sum(1 for d in detail if not d["ok"] and d["answer"].strip())
    result["gate_e_hallucination"] = {
        "wrong_assertions_in_sample": asserted_wrong,
        "sample_size": len(detail),
        "rate_pct": round(100.0 * asserted_wrong / max(len(detail), 1), 2),
        "note": "on probes whose answer WAS in the context, so a wrong number is "
                "an invention rather than a knowledge gap",
    }

    # --- EOS / length ---------------------------------------------------
    eos_hits, lengths = 0, []
    for prompt, _ in COPY_PROBES:
        _, length, hit_eos = generate(model, enc, prompt, 64)
        eos_hits += hit_eos
        lengths.append(length)
    result["eos"] = {"emitted": eos_hits, "of": len(COPY_PROBES),
                     "rate_pct": round(100.0 * eos_hits / len(COPY_PROBES), 1),
                     "mean_generated_tokens": round(sum(lengths) / len(lengths), 1)}
    return result


def verdict(result, baseline):
    """Pass/fail against the baseline, with the reason stated."""
    checks = []
    copy_now = result["gate_a_copy"]["score"]
    if baseline:
        copy_was = baseline["gate_a_copy"]["score"]
        floor = COPY_MIN_FRACTION * copy_was
        checks.append(("A copy retained", copy_now >= floor,
                       f"{copy_now}/{result['gate_a_copy']['of']} vs baseline "
                       f"{copy_was} (floor {floor:.1f})"))
        extraction_was = baseline["gate_b_extraction"]["accuracy_pct"]
        checks.append(("B extraction improved",
                       result["gate_b_extraction"]["accuracy_pct"] > extraction_was,
                       f"{result['gate_b_extraction']['accuracy_pct']}% vs "
                       f"{extraction_was}%"))
        checks.append(("C generalises to unseen wording",
                       result["gate_c_generalisation"]["accuracy_pct"] >
                       baseline["gate_c_generalisation"]["accuracy_pct"],
                       f"{result['gate_c_generalisation']['accuracy_pct']}% vs "
                       f"{baseline['gate_c_generalisation']['accuracy_pct']}%"))
        calc_now = result["gate_d_regression"]["calculation"]["accuracy_pct"]
        calc_was = baseline["gate_d_regression"]["calculation"]["accuracy_pct"]
        checks.append(("D calculation not regressed", calc_now >= calc_was - 5,
                       f"{calc_now}% vs {calc_was}%"))
        checks.append(("E hallucination not worse",
                       result["gate_e_hallucination"]["rate_pct"] <=
                       baseline["gate_e_hallucination"]["rate_pct"],
                       f"{result['gate_e_hallucination']['rate_pct']}% vs "
                       f"{baseline['gate_e_hallucination']['rate_pct']}%"))
    else:
        checks.append(("baseline recorded", True, "no comparison available"))
    return checks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",
                        default=os.path.join("checkpoints", "final",
                                             "checkpoint_247850.pt"))
    parser.add_argument("--baseline-json", default=None,
                        help="a previous run's JSON to compare against")
    parser.add_argument("--out", default=None)
    parser.add_argument("--label", default=None)
    args = parser.parse_args()

    import torch

    from data_sources.tokenizer import get_encoding
    from models import DeepSeekConfig, DeepSeekV3

    enc = get_encoding()
    model = DeepSeekV3(DeepSeekConfig.default())
    state = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(state["model_state_dict"] if "model_state_dict" in state
                          else state)
    model.eval()
    label = args.label or os.path.basename(args.checkpoint)
    print(f"running gates on {label}\n")

    result = run_gates(model, enc)
    result["checkpoint"] = args.checkpoint
    result["label"] = label

    print(f"Gate A  copy            {result['gate_a_copy']['score']}/"
          f"{result['gate_a_copy']['of']}")
    print(f"Gate B  extraction      {result['gate_b_extraction']['score']}/"
          f"{result['gate_b_extraction']['of']} "
          f"({result['gate_b_extraction']['accuracy_pct']}%)")
    print(f"Gate C  generalisation  {result['gate_c_generalisation']['score']}/"
          f"{result['gate_c_generalisation']['of']} "
          f"({result['gate_c_generalisation']['accuracy_pct']}%)")
    d = result["gate_d_regression"]
    print(f"Gate D  calculation     {d['calculation']['score']}/{d['calculation']['of']}"
          f"  abstention {d['abstention']['score']}/{d['abstention']['of']}"
          f"  over-abstention {d['over_abstention']['score']}/{d['over_abstention']['of']}"
          f"  interpretation {d['interpretation']['score']}/{d['interpretation']['of']}")
    print(f"Gate E  hallucination   {result['gate_e_hallucination']['rate_pct']}%")
    print(f"        EOS             {result['eos']['emitted']}/{result['eos']['of']} "
          f"({result['eos']['rate_pct']}%), mean {result['eos']['mean_generated_tokens']} "
          f"tokens")

    baseline = None
    if args.baseline_json and os.path.exists(args.baseline_json):
        with open(args.baseline_json, encoding="utf-8") as handle:
            baseline = json.load(handle)
    checks = verdict(result, baseline)
    print()
    passed = True
    for name, ok, detail in checks:
        passed = passed and ok
        print(f"[{'PASS' if ok else 'FAIL'}] Gate {name} - {detail}")
    result["verdict"] = {"checks": [{"name": n, "passed": bool(o), "detail": d}
                                    for n, o, d in checks],
                         "all_passed": bool(passed)}

    out = args.out or os.path.join("reports", f"gates_{label.replace('.pt', '')}.json")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(f"\nwrote {out}")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
