"""Score any answer function against the frozen held-out set.

One scorer for every system - Aivora, zero-shot base models, the tool pipeline,
LoRA - so the numbers are comparable. The correctness rules are those
pre-registered in EXPERIMENT_RULES.md section 7, and they live here so no system
can be scored by its own standard.

    from scripts.eval_heldout import evaluate
    result = evaluate(lambda q, c: my_system(q, c), label="qwen-0.5b zero-shot")

    python scripts/eval_heldout.py --system aivora_baseline
"""

import argparse
import hashlib
import io
import json
import math
import os
import re
import sys
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

FROZEN = os.path.join("data", "eval_heldout", "heldout_frozen.jsonl")
SELECTION = os.path.join("data", "eval_heldout", "selection.jsonl")
MANIFEST = os.path.join("data", "eval_heldout", "heldout_manifest.json")


def content_sha256(path):
    """A hash of the ITEMS, not the bytes.

    A plain byte hash of a text file is platform-dependent: with
    core.autocrlf=true a Windows checkout has CRLF line endings and a Linux one
    has LF, so the same frozen set hashes differently on the two machines. That
    made the integrity check raise a false alarm when Phase 3 ran on Kaggle, and
    - worse - it could not have detected a real change in a cross-platform run,
    because every run was expected to disagree. This hashes the parsed items in
    a canonical order instead, so it is identical wherever the file is checked
    out and still changes if any item changes.
    """
    with io.open(path, encoding="utf-8") as handle:
        items = [json.loads(line) for line in handle if line.strip()]
    canonical = json.dumps(sorted(items, key=lambda item: item["id"]),
                           sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def verify_split(path, split):
    """Returns (content_sha, status). Never silently continues on a mismatch."""
    digest = content_sha256(path)
    if not os.path.exists(MANIFEST):
        return digest, "UNVERIFIED: no manifest"
    with io.open(MANIFEST, encoding="utf-8") as handle:
        manifest = json.load(handle)
    recorded = manifest.get(split, {}).get("content_sha256")
    if recorded is None:
        return digest, "UNVERIFIED: manifest predates content hashing"
    if recorded != digest:
        raise SystemExit(
            f"\nREFUSING TO SCORE: {path} does not match the manifest.\n"
            f"  manifest content_sha256 {recorded}\n"
            f"  file     content_sha256 {digest}\n"
            f"The {split} set changed after it was frozen. Any number produced "
            f"from it would not be comparable with earlier results. Investigate "
            f"before scoring - do not rebuild to make this pass.")
    return digest, "VERIFIED"


NUMBER = re.compile(r"-?\d{1,3}(?:,\d{3})+(?:\.\d+)?|-?\d+(?:\.\d+)?")


def wilson(k, n, z=1.96):
    """95% Wilson interval - honest at small n and at 0, where Wald is not."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (round(100 * max(0.0, centre - margin), 2),
            round(100 * min(1.0, centre + margin), 2))


def substantive_answer(text):
    if not text:
        return ""
    # Strip prompt continuation / echoed Q&A loops
    parts = re.split(r"(?:\n+|^|\b)Question:\s*", text)
    first_part = parts[0].strip()
    return first_part if first_part else text.strip()


# A numeric token: not glued to a letter or digit on either side, optionally
# carrying a ratio's "x" unit, and never a prefix of something longer.
#
# The trailing (?!\.\d) matters as much as the boundary. With the boundary
# lookahead alone the engine backtracks instead of giving up: on "2.50x" it
# cannot match "2.50" because "x" follows, so it matched "2" and returned 2.0 -
# silently truncating. That is not a near miss, it is a wrong number that can
# score a wrong answer CORRECT (expected 2.0, answer "2.50x"). A token that
# cannot be read whole is now rejected outright, the same discipline
# pipeline/tool_pipeline.py numeric() already applies to "2,60.65".
#
# "x" is accepted as a unit because a ratio is legitimately written "2.50x" and
# the suffix carries no numeric meaning. Letters still cannot begin or sit
# inside a token, so "Q3", "FY2024" and "BL-9" remain non-numeric.
# The (?<!\d\.) guard is the mirror image: without it "1.2.3" matches "2.3" in
# the middle and reports 2.3 as a number in the text.
NUMBER_TOKEN = re.compile(
    r"(?<![A-Za-z0-9])(?<!\d\.)-?\d[\d,]*(?:\.\d+)?[xX]?(?![A-Za-z0-9])(?!\.\d)")


def numbers_in(text):
    if not text:
        return []
    raw_tokens = NUMBER_TOKEN.findall(text)
    nums = []
    for t in raw_tokens:
        t = t.strip().rstrip("xX")
        if not t:
            continue
        if "," in t:
            # Valid thousands format requires groups of 3 digits after comma
            if re.match(r"^-?\d{1,3}(?:,\d{3})+(?:\.\d+)?$", t):
                nums.append(float(t.replace(",", "")))
            else:
                # Malformed comma placement (e.g. 2,60.65) - do not extract valid numbers from it
                pass
        elif re.match(r"^-?\d+(?:\.\d+)?$", t):
            nums.append(float(t))
    return nums


def score_item(item, answer):
    """Returns (correct, category). Categories follow EXPERIMENT_RULES.md s7."""
    from evaluation.financial_metrics import abstained

    answer = (answer or "").strip()
    clean_ans = substantive_answer(answer)
    expected = item["expected"]
    gate = item["gate"]

    if gate == "abstention":
        declined = abstained(clean_ans) or abstained(answer)
        if expected is None:
            return declined, ("correct_abstention" if declined
                              else "hallucinated_instead_of_abstaining")
        # Answerable control: must NOT refuse.
        return (not declined), ("answered_control" if not declined
                                else "over_refusal")

    if gate == "calculation":
        if not clean_ans:
            return False, "empty"
        target = float(expected)
        tolerance = max(0.05, abs(target) * 0.005)
        found = numbers_in(clean_ans)
        if any(abs(v - target) <= tolerance for v in found):
            return True, "correct"
        if abstained(clean_ans):
            return False, "incorrect_abstention"
        return False, ("wrong_value" if found else "no_number")

    # copy / extraction / wording: the expected string, or its numeric value
    expected_text = str(expected)
    if not clean_ans:
        return False, "empty"
    if abstained(clean_ans):
        return False, "incorrect_abstention"

    # Alphanumeric identifiers (e.g. "4H58-E2", "TRV-8841", "MX-5520", "RZZR")
    # must match exactly with token boundaries and must NOT fall back to numbers.
    is_alphanumeric_id = any(c.isalpha() for c in expected_text)
    if is_alphanumeric_id:
        pattern = r"(?<![A-Za-z0-9])" + re.escape(expected_text) + r"(?![A-Za-z0-9])"
        if re.search(pattern, clean_ans, re.IGNORECASE):
            return True, "correct"
        return False, "wrong_text"

    if expected_text.lower() in clean_ans.lower():
        # Guard against an answer that also contains a contradicting value.
        context_numbers = numbers_in(item["context"])
        expected_numbers = numbers_in(expected_text)
        if expected_numbers:
            others = [v for v in numbers_in(clean_ans)
                      if all(abs(v - e) > 0.001 for e in expected_numbers)
                      and any(abs(v - c) < 0.001 for c in context_numbers)]
            if others:
                return False, "ambiguous_multiple_values"
        return True, "correct"

    expected_numbers = numbers_in(expected_text)
    if expected_numbers:
        if any(abs(v - expected_numbers[0]) < 0.01 for v in numbers_in(clean_ans)):
            return True, "correct"
        answer_numbers = numbers_in(clean_ans)
        context_numbers = numbers_in(item["context"])
        if answer_numbers and any(any(abs(v - c) < 0.001 for c in context_numbers)
                                  for v in answer_numbers):
            return False, "wrong_field"
        if answer_numbers:
            return False, "invented_value"
        return False, "no_number"
    return False, "wrong_text"


def evaluate(answer_fn, items=None, label="system", split="frozen", verbose=False,
             save_raw=None):
    """answer_fn(question, context) -> answer string."""
    path = FROZEN if split == "frozen" else SELECTION
    integrity = None
    if items is None:
        digest, status = verify_split(path, split)
        integrity = {"split": split, "content_sha256": digest, "status": status}
        print(f"  {split} set: {status}, content_sha256 {digest[:16]}...")
    if items is None:
        with open(path, encoding="utf-8") as handle:
            items = [json.loads(line) for line in handle if line.strip()]

    records = []
    for index, item in enumerate(items, 1):
        answer = answer_fn(item["question"], item["context"])
        correct, category = score_item(item, answer)
        records.append({"id": item["id"], "gate": item["gate"],
                        "origin": item["origin"], "correct": bool(correct),
                        "category": category, "answer": answer,
                        "expected": item["expected"], "note": item.get("note")})
        if verbose:
            print(f"  {'OK ' if correct else 'XX '}{item['id']}: {answer[:60]!r}")
        elif index % 50 == 0:
            print(f"    {index}/{len(items)}")

    by_gate = {}
    for gate in sorted({r["gate"] for r in records}):
        subset = [r for r in records if r["gate"] == gate]
        k = sum(1 for r in subset if r["correct"])
        low, high = wilson(k, len(subset))
        by_gate[gate] = {"k": k, "n": len(subset),
                         "accuracy_pct": round(100.0 * k / len(subset), 2),
                         "ci95": [low, high],
                         "categories": dict(Counter(r["category"] for r in subset
                                                    if not r["correct"]).most_common())}

    # The abstention gate mixes two OPPOSITE tasks: items that must be refused
    # and answerable controls that must not be. A combined accuracy gives a model
    # that never refuses a free 16/16 on the controls, which read as 30.77%
    # "abstention" for the Aivora baseline whose true refusal rate is 0/36. The
    # gate is therefore reported split, and the combined figure is dropped.
    must_abstain = [r for r in records if r["gate"] == "abstention"
                    and r["expected"] is None]
    if must_abstain:
        k = sum(1 for r in must_abstain if r["correct"])
        low, high = wilson(k, len(must_abstain))
        by_gate["abstention"] = {
            "k": k, "n": len(must_abstain),
            "accuracy_pct": round(100.0 * k / len(must_abstain), 2),
            "ci95": [low, high],
            "note": ("refusals on items that MUST be refused; the answerable "
                     "controls are reported as over_refusal, never mixed in"),
            "categories": dict(Counter(r["category"] for r in must_abstain
                                       if not r["correct"]).most_common()),
        }

    # Over-refusal and hallucination, called out separately per the rules.
    controls = [r for r in records if r["gate"] == "abstention"
                and r["expected"] is False]
    over_refusal = sum(1 for r in controls if r["category"] == "over_refusal")
    answerable = [r for r in records if r["gate"] != "abstention"]
    invented = sum(1 for r in answerable if r["category"] == "invented_value")

    # The overall figure counts each item once, using the split abstention
    # definition: controls contribute through over_refusal, not through the gate.
    scored_records = [r for r in records
                      if not (r["gate"] == "abstention" and r["expected"] is False)]
    total_k = sum(1 for r in scored_records if r["correct"])
    summary = {
        "label": label, "split": split, "items": len(records),
        "overall": {"k": total_k, "n": len(scored_records),
                    "accuracy_pct": round(100.0 * total_k / len(scored_records), 2),
                    "ci95": list(wilson(total_k, len(scored_records))),
                    "note": ("answerable abstention controls excluded; they are "
                             "reported as over_refusal")},
        "by_gate": by_gate,
        "over_refusal": {"k": over_refusal, "n": len(controls),
                         "pct": round(100.0 * over_refusal / max(len(controls), 1), 2)},
        "hallucination_invented_values": {
            "k": invented, "n": len(answerable),
            "pct": round(100.0 * invented / max(len(answerable), 1), 2)},
        "handwritten_only": {},
        "eval_set_integrity": integrity,
    }
    for gate in by_gate:
        hand = [r for r in records if r["gate"] == gate and r["origin"] == "handwritten"]
        if hand:
            k = sum(1 for r in hand if r["correct"])
            summary["handwritten_only"][gate] = {
                "k": k, "n": len(hand),
                "accuracy_pct": round(100.0 * k / len(hand), 2)}

    if save_raw:
        os.makedirs(os.path.dirname(save_raw) or ".", exist_ok=True)
        with open(save_raw, "w", encoding="utf-8") as handle:
            json.dump({"summary": summary, "records": records}, handle, indent=2)
    return summary, records


def print_summary(summary):
    print(f"\n{summary['label']} ({summary['split']}, n={summary['items']})")
    print(f"{'gate':<14}{'k/n':>10}{'acc':>9}{'95% CI':>18}")
    for gate, stats in summary["by_gate"].items():
        interval = f"[{stats['ci95'][0]}, {stats['ci95'][1]}]"
        print(f"{gate:<14}{stats['k']:>4}/{stats['n']:<5}"
              f"{stats['accuracy_pct']:>8.2f}%{interval:>18}")
    overall = summary["overall"]
    print(f"{'OVERALL':<14}{overall['k']:>4}/{overall['n']:<5}"
          f"{overall['accuracy_pct']:>8.2f}%")
    print(f"over-refusal {summary['over_refusal']['k']}/"
          f"{summary['over_refusal']['n']} "
          f"({summary['over_refusal']['pct']}%)   "
          f"invented values {summary['hallucination_invented_values']['k']}/"
          f"{summary['hallucination_invented_values']['n']} "
          f"({summary['hallucination_invented_values']['pct']}%)")


def aivora_answer_fn(checkpoint):
    import torch

    from data_sources.tokenizer import get_encoding
    from data_sources.training_format import PROMPT_TEMPLATE
    from models import DeepSeekConfig, DeepSeekV3

    enc = get_encoding()
    model = DeepSeekV3(DeepSeekConfig.default())
    state = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state["model_state_dict"] if "model_state_dict" in state
                          else state)
    model.eval()

    def answer(question, context):
        prompt = f"{context}\n\n{question}" if context else question
        ids = enc.encode_ordinary(PROMPT_TEMPLATE.format(prompt=prompt))
        torch.manual_seed(1234)
        with torch.no_grad():
            out = model.generate(torch.tensor(ids).unsqueeze(0), 160,
                                 temperature=1e-5, top_k=1,
                                 stop_on_repetition=True,
                                 eos_token_id=enc.eot_token)
        generated = out[0, len(ids):].tolist()
        if enc.eot_token in generated:
            generated = generated[:generated.index(enc.eot_token)]
        return enc.decode(generated).strip()

    return answer


SYSTEMS = {
    "aivora_baseline": os.path.join("checkpoints", "final", "checkpoint_247850.pt"),
    "aivora_sft_002": os.path.join("checkpoints", "sft_002", "sft_002_best.pt"),
    "aivora_sft_001": os.path.join("checkpoints", "sft_001", "sft_001_best.pt"),
    "aivora_sft_003": os.path.join("checkpoints", "sft_003", "sft_003_best.pt"),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--system", required=True, choices=list(SYSTEMS))
    parser.add_argument("--split", default="frozen", choices=("frozen", "selection"))
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    path = SYSTEMS[args.system]
    if not os.path.exists(path):
        sys.exit(f"missing checkpoint: {path}")
    print(f"scoring {args.system} on the {args.split} split")

    items = None
    if args.limit:
        source = FROZEN if args.split == "frozen" else SELECTION
        with open(source, encoding="utf-8") as handle:
            items = [json.loads(line) for line in handle if line.strip()][:args.limit]

    out = os.path.join("reports", "heldout", f"{args.system}_{args.split}.json")
    summary, _ = evaluate(aivora_answer_fn(path), items=items, label=args.system,
                          split=args.split, save_raw=out)
    print_summary(summary)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
