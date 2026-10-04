"""Phase 3: zero-shot and few-shot baselines for open instruct models.

Identical prompts, identical greedy decoding, identical scorer for every model -
the only variable is which model, as pre-registered in EXPERIMENT_RULES.md.

Licences verified before use (Phase 3 requirement):

  Qwen/Qwen2.5-0.5B-Instruct    Apache-2.0     no gate, commercial use allowed
  Qwen/Qwen2.5-1.5B-Instruct    Apache-2.0     no gate
  HuggingFaceTB/SmolLM2-1.7B-Instruct  Apache-2.0  no gate
  meta-llama/Llama-3.2-1B-Instruct     Llama 3.2 Community Licence - GATED,
                                       requires accepting terms on the Hub, so it
                                       is NOT included by default. Add it only
                                       after you have accepted the licence.

Memory: this machine has 8 GB total. 0.5B in float32 needs roughly 2 GB plus
activations; 1.5B needs about 6 GB and will not fit alongside anything else.
The script measures free memory and refuses rather than thrashing, naming what
it needs. Larger models run on Kaggle with the same code.

    python scripts/eval_base_models.py --model Qwen/Qwen2.5-0.5B-Instruct
    python scripts/eval_base_models.py --model Qwen/Qwen2.5-0.5B-Instruct --shots 3
    python scripts/eval_base_models.py --model ... --split selection
"""

import argparse
import json
import os
import platform
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

LICENCES = {
    # name: (licence, approximate RAM in GB for float32 CPU inference)
    "HuggingFaceTB/SmolLM2-135M-Instruct": ("Apache-2.0", 1.1),
    "Qwen/Qwen2.5-0.5B-Instruct": ("Apache-2.0", 2.5),
    "Qwen/Qwen2.5-1.5B-Instruct": ("Apache-2.0", 7.0),
    "HuggingFaceTB/SmolLM2-360M-Instruct": ("Apache-2.0", 2.0),
    "HuggingFaceTB/SmolLM2-1.7B-Instruct": ("Apache-2.0", 8.0),
}

SYSTEM_PROMPT = (
    "You are a financial analysis assistant. Answer only from the information "
    "given. If the information needed is not present, say that it is "
    "insufficient rather than guessing. Never invent a number. Answer with the "
    "value alone, as briefly as possible."
)

# Few-shot examples. Written here rather than sampled from training data, and
# deliberately NOT drawn from any gate in the evaluation set.
FEW_SHOT = [
    ("Stock code: BL-9\nQuantity held: 42\n\nWhat is the stock code?", "BL-9"),
    ("Sales: 500.00\nCosts: 300.00\n\nWhat is sales?", "500.00"),
    ("Headcount: 88\n\nWhat is the average salary?",
     "Insufficient information: the average salary is not given."),
]


def free_memory_gb():
    if platform.system() == "Windows":
        import ctypes

        class Status(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong),
                        ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

        status = Status()
        status.dwLength = ctypes.sizeof(Status)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
        return status.ullAvailPhys / 1024 ** 3
    try:
        import psutil

        return psutil.virtual_memory().available / 1024 ** 3
    except Exception:
        return float("inf")


def build_answer_fn(model_name, shots, max_new_tokens):
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

    def answer(question, context):
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for example_user, example_assistant in FEW_SHOT[:shots]:
            messages.append({"role": "user", "content": example_user})
            messages.append({"role": "assistant", "content": example_assistant})
        user = f"{context}\n\n{question}" if context else question
        messages.append({"role": "user", "content": user})

        text = tokenizer.apply_chat_template(messages, tokenize=False,
                                             add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens,
                                 do_sample=False, temperature=None, top_p=None,
                                 pad_token_id=tokenizer.eos_token_id)
        generated = out[0, inputs["input_ids"].shape[1]:]
        return tokenizer.decode(generated, skip_special_tokens=True).strip()

    return answer, model, tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--shots", type=int, default=0, choices=(0, 1, 2, 3))
    parser.add_argument("--split", default="selection",
                        choices=("frozen", "selection"),
                        help="selection for model choice; frozen for reporting only")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--force", action="store_true",
                        help="run even if free memory looks insufficient")
    args = parser.parse_args()

    licence, needed_gb = LICENCES.get(args.model, (None, 8.0))
    if licence is None:
        print(f"WARNING: {args.model} is not in the verified licence list. "
              f"Verify its licence before using the result.")
    else:
        print(f"licence: {licence} (verified, ungated)")

    free = free_memory_gb()
    print(f"free memory: {free:.1f} GB, this model needs about {needed_gb} GB")
    if free < needed_gb and not args.force:
        sys.exit(
            f"\nREFUSING TO RUN: {free:.1f} GB free, about {needed_gb} GB needed.\n"
            f"This is not a result - it is an unmet requirement. Options:\n"
            f"  * close other work and retry\n"
            f"  * use a smaller model (SmolLM2-360M or Qwen2.5-0.5B)\n"
            f"  * run the same command on Kaggle, where the code is unchanged\n"
            f"  * pass --force to try anyway")

    from scripts.eval_heldout import evaluate, print_summary

    answer_fn, _, _ = build_answer_fn(args.model, args.shots, args.max_new_tokens)

    items = None
    if args.limit:
        source = os.path.join("data", "eval_heldout",
                              "heldout_frozen.jsonl" if args.split == "frozen"
                              else "selection.jsonl")
        with open(source, encoding="utf-8") as handle:
            items = [json.loads(line) for line in handle if line.strip()][:args.limit]

    label = f"{args.model} ({args.shots}-shot)"
    slug = args.model.replace("/", "_") + f"_{args.shots}shot_{args.split}"
    out = os.path.join("reports", "heldout", f"{slug}.json")
    started = time.time()
    summary, records = evaluate(answer_fn, items=items, label=label,
                                split=args.split, save_raw=out)
    summary["seconds"] = round(time.time() - started, 1)
    summary["seconds_per_item"] = round(summary["seconds"] / max(len(records), 1), 2)
    summary["licence"] = licence
    summary["shots"] = args.shots
    with open(out, encoding="utf-8") as handle:
        payload = json.load(handle)
    payload["summary"] = summary
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    print_summary(summary)
    print(f"\n{summary['seconds_per_item']}s per item, {summary['seconds']}s total")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
