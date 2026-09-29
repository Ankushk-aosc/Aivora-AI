"""Step 8: Aivora vs Qwen2.5-1.5B-Instruct vs Qwen + the project's finance LoRA.

Same benchmark, same scorer, same prompt handling for all three, so the numbers
are comparable. The point is an external-model baseline, not a verdict about
which project is better.

WHERE THIS CAN RUN. Not on the development machine: 8 GB RAM total with ~0.4 GB
free, against ~3.1 GB of Qwen weights. Run it on Kaggle (the repository already
has notebook infrastructure under training/kaggle/) or any host with a GPU or
~8 GB of free RAM. The script refuses rather than thrashing if memory is short.

    python scripts/compare_models.py --split blind
    python scripts/compare_models.py --split blind --models aivora
    python scripts/compare_models.py --external benchmarks/vendor.jsonl

Nothing is tuned against the evaluation set: the Qwen runs use the model's own
chat template and greedy decoding, which is what HFBackend already does.
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

AIVORA_CHECKPOINT = os.path.join("checkpoints", "final", "checkpoint_247850.pt")
QWEN_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
FINANCE_LORA = os.path.join("checkpoints", "qwen_finance_lora")
SEED = 1234
MIN_FREE_GB_FOR_QWEN = 6.0


def free_memory_gb():
    try:
        import psutil

        return psutil.virtual_memory().available / 1024 ** 3
    except Exception:
        pass
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
    return float("inf")


def peak_memory_gb():
    try:
        import resource

        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2
    except Exception:
        try:
            import psutil

            return psutil.Process().memory_info().rss / 1024 ** 3
        except Exception:
            return None


def aivora_answer_fn(checkpoint):
    """Model-only: the checkpoint, nothing else."""
    import torch

    from app.backend.services.generation import DeepSeekBackend
    from inference import load_model_for_inference

    model, _ = load_model_for_inference(checkpoint, device="cpu")
    backend = DeepSeekBackend(model, device="cpu", checkpoint=checkpoint)

    def answer(question):
        torch.manual_seed(SEED)
        return backend.generate(f"Question: {question}\nAnswer:", max_new_tokens=128,
                                temperature=0.7, top_k=40, top_p=0.9,
                                repetition_penalty=1.3)

    return answer, {"name": "Aivora 101.7M (model only)", "parameters": 101_723_264}


def qwen_answer_fn(adapter=None):
    """Qwen through its own chat template, greedy, no tuning against the set."""
    from app.backend.services.generation import HFBackend

    backend = HFBackend(QWEN_MODEL, adapter_dir=adapter)
    described = backend.describe()

    def answer(question):
        return backend.generate(question, max_new_tokens=128, temperature=0.0)

    label = "Qwen2.5-1.5B-Instruct" + (" + finance LoRA" if adapter else " (base)")
    return answer, {"name": label, "parameters": described.get("parameters"),
                    "adapter": adapter}


def run_one(label, answer_fn, items, evaluate):
    started = time.time()
    result = evaluate(items, answer_fn, label=label)
    elapsed = time.time() - started
    result["seconds"] = round(elapsed, 1)
    result["seconds_per_question"] = round(elapsed / max(len(items), 1), 3)
    result["peak_memory_gb"] = peak_memory_gb()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="blind",
                        help="a benchmark in data/benchmark/ (dev, test, blind)")
    parser.add_argument("--external", default=None,
                        help="path to an external benchmark JSONL instead of --split")
    parser.add_argument("--models", nargs="*",
                        default=["aivora", "qwen_base", "qwen_lora"])
    parser.add_argument("--checkpoint", default=AIVORA_CHECKPOINT)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", default=os.path.join("reports", "model_comparison.json"))
    args = parser.parse_args()

    if args.external:
        from evaluation.external import evaluate_external, load_external

        items = load_external(args.external)
        evaluate = evaluate_external
        source = args.external
    else:
        from evaluation.benchmark import load_benchmark, score_item, summarise

        items = load_benchmark(args.split)
        source = f"data/benchmark/{args.split}.jsonl"

        def evaluate(items, answer_fn, label=""):
            records = []
            for index, item in enumerate(items, 1):
                records.append(score_item(item, answer_fn(item["question"])))
                if index % 50 == 0:
                    print(f"    {index}/{len(items)}")
            return {"label": label, "examples": len(records),
                    "summary": summarise(records), "details": records}

    if args.limit:
        items = items[:args.limit]

    free = free_memory_gb()
    print(f"benchmark: {source}, {len(items)} items; free memory {free:.1f} GB")

    report = {"benchmark": source, "examples": len(items), "seed": SEED,
              "free_memory_gb_at_start": round(free, 2), "models": {}}

    for name in args.models:
        if name == "aivora":
            answer_fn, meta = aivora_answer_fn(args.checkpoint)
        elif name in ("qwen_base", "qwen_lora"):
            if free < MIN_FREE_GB_FOR_QWEN:
                message = (f"skipped: {free:.1f} GB free, needs ~{MIN_FREE_GB_FOR_QWEN} GB "
                           f"for {QWEN_MODEL}. Run this on Kaggle or a host with more "
                           f"memory - it is not a result, it is an unmet requirement.")
                print(f"  {name}: {message}")
                report["models"][name] = {"skipped": message}
                continue
            adapter = FINANCE_LORA if name == "qwen_lora" else None
            if adapter and not os.path.isdir(adapter):
                report["models"][name] = {"skipped": f"no adapter at {adapter}"}
                print(f"  {name}: no adapter at {adapter}")
                continue
            answer_fn, meta = qwen_answer_fn(adapter)
        else:
            sys.exit(f"unknown model {name!r}")

        print(f"\n{meta['name']}")
        result = run_one(meta["name"], answer_fn, items, evaluate)
        result["model"] = meta
        report["models"][name] = result
        summary = result["summary"]
        print(f"  accuracy {summary['overall']['accuracy']}%  "
              f"hallucination {summary['hallucination_rate']}%  "
              f"{result['seconds_per_question']} s/question")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
