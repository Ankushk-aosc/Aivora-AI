"""Compare two systems on the same items with the exact McNemar test.

EXPERIMENT_RULES_v2 section 2: systems are evaluated on identical items, so the
observations are PAIRED. Two-proportion and Fisher tests treat them as
independent, which is the wrong model and overstates uncertainty. McNemar uses
only the discordant pairs, which is what actually carries the evidence.

  b = items system A got right and system B got wrong
  c = items system A got wrong and system B got right

Exact two-sided p = 2 * P(X <= min(b,c)) under Binomial(b+c, 0.5), capped at 1.

    python scripts/compare_systems.py reports/heldout/a.json reports/heldout/b.json
"""

import argparse
import json
import math
import os
import sys
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def mcnemar_exact(b, c):
    """Exact two-sided McNemar. Returns (p, n_discordant)."""
    n = b + c
    if n == 0:
        return 1.0, 0
    smaller = min(b, c)
    tail = sum(math.comb(n, k) for k in range(smaller + 1)) / (2 ** n)
    return min(1.0, 2 * tail), n


def minimum_detectable(n_discordant):
    """The most lopsided split that would still NOT reach p<0.05 - a plain way
    to say whether a null result means anything."""
    if n_discordant == 0:
        return None
    for smaller in range(n_discordant // 2 + 1):
        p, _ = mcnemar_exact(n_discordant - smaller, smaller)
        if p < 0.05:
            return {"discordant": n_discordant,
                    "significant_from_split": f"{n_discordant - smaller}:{smaller}"}
    return {"discordant": n_discordant,
            "significant_from_split": "no split at this n reaches p<0.05"}


def load(path):
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    records = {r["id"]: r for r in payload["records"]}
    return payload["summary"], records


def compare(path_a, path_b):
    summary_a, records_a = load(path_a)
    summary_b, records_b = load(path_b)
    shared = sorted(set(records_a) & set(records_b))
    if not shared:
        sys.exit("no shared items - these systems were not scored on the same set")

    gates = sorted({records_a[i]["gate"] for i in shared})
    result = {"system_a": summary_a["label"], "system_b": summary_b["label"],
              "shared_items": len(shared), "test": "exact McNemar, two-sided",
              "by_gate": {}}

    for gate in list(gates) + ["ALL"]:
        ids = [i for i in shared if gate == "ALL" or records_a[i]["gate"] == gate]
        b = sum(1 for i in ids if records_a[i]["correct"] and not records_b[i]["correct"])
        c = sum(1 for i in ids if not records_a[i]["correct"] and records_b[i]["correct"])
        both = sum(1 for i in ids if records_a[i]["correct"] and records_b[i]["correct"])
        neither = len(ids) - b - c - both
        p, discordant = mcnemar_exact(b, c)
        result["by_gate"][gate] = {
            "n": len(ids),
            "a_correct": both + b, "b_correct": both + c,
            "both_correct": both, "neither_correct": neither,
            "a_only": b, "b_only": c, "discordant": discordant,
            "p_value": round(p, 4),
            "significant_at_0.05": bool(p < 0.05),
            "verdict": ("B better" if c > b and p < 0.05 else
                        "A better" if b > c and p < 0.05 else
                        "not distinguishable at this sample size"),
            "power_note": minimum_detectable(discordant),
        }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("a")
    parser.add_argument("b")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    result = compare(args.a, args.b)
    print(f"A = {result['system_a']}")
    print(f"B = {result['system_b']}")
    print(f"{result['shared_items']} shared items, exact McNemar (two-sided)\n")
    print(f"{'gate':<14}{'A':>6}{'B':>6}{'A only':>8}{'B only':>8}{'p':>9}  verdict")
    for gate, stats in result["by_gate"].items():
        print(f"{gate:<14}{stats['a_correct']:>6}{stats['b_correct']:>6}"
              f"{stats['a_only']:>8}{stats['b_only']:>8}{stats['p_value']:>9.4f}"
              f"  {stats['verdict']}")
    print("\nWhere a gate says 'not distinguishable', the discordant count and the "
          "split that WOULD have been significant are in the JSON, so a null "
          "result is not mistaken for equivalence.")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
