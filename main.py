"""
DeepSeek-V3-Inspired Financial LLM — CLI entry point.

python main.py demo
python main.py dataset list
python main.py dataset download --name <id>
python main.py dataset prepare --name <id> --max-tokens N [--max-records N]
python main.py dataset stats [--name <id>]
python main.py dataset dashboard
python main.py train --preset tiny_debug|small|financial_poc [--wandb]
python main.py train --resume <checkpoint.pt>
python main.py train --stage instruction --checkpoint <checkpoint.pt>
python main.py evaluate --checkpoint <checkpoint.pt>
python main.py compare --base <ckpt> --instruction <ckpt>
python main.py chat --checkpoint <checkpoint.pt> [--document FILE]
python main.py inspect tokens|architecture|moe|forward [--checkpoint <ckpt>]
"""

import argparse
import os
import sys

import torch

from models import DeepSeekConfig, DeepSeekV3


def demo():
    """Run a simple demo using the unified model configuration."""
    print("=" * 50)
    print("DEEPSEEK-V3 FINANCIAL LLM DEMO")
    print("=" * 50)

    config = DeepSeekConfig.default()
    model = DeepSeekV3(config)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Created model with {total_params:,} parameters (from configs/model_config.yaml)")

    batch_size, seq_len = 2, 32
    test_input = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    test_targets = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    with torch.no_grad():
        logits, total_loss, main_loss, mtp_loss = model(test_input, test_targets)

    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {logits.shape}")
    print(f"Main loss: {main_loss:.4f}")
    if mtp_loss is not None:
        print(f"MTP loss: {mtp_loss:.4f}")
        print(f"Total loss: {total_loss:.4f}")

    prompt = torch.randint(0, config.vocab_size, (1, 5))
    with torch.no_grad():
        generated = model.generate(prompt, max_new_tokens=20, temperature=1.0, top_k=10)

    print(f"Generated {generated.shape[1] - prompt.shape[1]} tokens")
    print("Demo completed!")


def cmd_runtime(args):
    from tools.runtime_detect import detect_runtime, print_report
    report = detect_runtime()
    print_report(report)


def cmd_dataset_list(args):
    from data_sources import list_entries

    entries = list_entries()
    print(f"{'name':<32} {'category':<22} {'license':<18} {'status':<28} hf_id")
    print("-" * 130)
    for e in sorted(entries, key=lambda e: (e.category, e.name)):
        print(f"{e.name:<32} {e.category:<22} {e.license:<18} {e.verification_status:<28} {e.hf_id}")


def cmd_dataset_download(args):
    # "Download" for a streaming HF source means: pull a small probe batch
    # to confirm access, without materializing the dataset.
    from data_sources import get_entry
    from data_sources.huggingface_loader import stream_records

    entry = get_entry(args.name)
    print(f"Probing {entry.hf_id} (subset={entry.subset}, split={entry.split}) ...")
    count = 0
    for _ in stream_records(entry, max_records=5):
        count += 1
    print(f"OK - streamed {count} sample record(s) from {entry.hf_id}.")


def cmd_dataset_prepare(args):
    from data_sources import prepare_dataset

    summary = prepare_dataset(
        name=args.name,
        max_tokens=args.max_tokens,
        max_records=args.max_records,
    )
    print("Prepared dataset:")
    for k, v in summary.items():
        print(f"  {k}: {v}")


def cmd_dataset_dashboard(args):
    """Dataset dashboard (Part 30 / §44) - every value from actual processing."""
    import json as _json

    from data_sources import REGISTRY
    from data_sources.dataset_mixer import BUCKET_TO_CATEGORY
    from data_sources.manifest import read_manifest
    from data_sources.shard_writer import load_shard_index

    manifest = read_manifest()
    if not manifest["datasets"]:
        print("No datasets prepared yet. Run `python main.py dataset prepare` first.")
        return

    print("=" * 100)
    print("DATASET DASHBOARD (measured from actual prepare runs)")
    print("=" * 100)

    for record in manifest["datasets"]:
        entry = REGISTRY.get(record["name"])
        removed = record["records_removed_invalid"] + record["records_removed_duplicate"]
        used = record["train_records_used"] + record["validation_records_used"]
        shards_root = os.path.join("data", "shards", record["name"])
        train_index = load_shard_index(os.path.join(shards_root, "train"))
        avg_len = (record["train_tokens_used"] / record["train_records_used"]
                    if record["train_records_used"] else 0)

        print(f"\n{record['name']}")
        print(f"  Hugging Face ID   : {record['dataset_id']}")
        print(f"  Subset            : {record['subset'] or '(none)'}")
        print(f"  Revision          : {record['revision'] or 'Not pinned'}")
        print(f"  License           : {record['license']}")
        print(f"  Source URL        : {record['source_url']}")
        print(f"  Category          : {entry.category if entry else 'Not available'}")
        print(f"  Verification      : {record['verification_status']}")
        print(f"  Records seen      : {record['records_seen']:,}")
        print(f"  Records used      : {used:,}")
        print(f"  Records removed   : {removed:,} "
              f"(invalid {record['records_removed_invalid']:,}, "
              f"duplicate {record['records_removed_duplicate']:,})")
        print(f"  Training tokens   : {record['train_tokens_used']:,}")
        print(f"  Validation tokens : {record['validation_tokens_used']:,}")
        print(f"  Avg tokens/record : {avg_len:.1f}")
        print(f"  Shards (train)    : {len(train_index.get('shards', []))}")
        print(f"  Prepared          : {record['download_date']}")

    print("\n" + "=" * 100)
    print("CONFIGURED MIXTURE")
    print("=" * 100)
    from training.trainer import load_preset
    preset = load_preset(args.preset)
    prepared = {r["name"] for r in manifest["datasets"]}
    for bucket, weight in preset["dataset_mix"].items():
        category = BUCKET_TO_CATEGORY[bucket]
        members = [n for n, e in REGISTRY.items() if e.category == category]
        ready = [n for n in members if n in prepared]
        status = f"prepared: {', '.join(ready)}" if ready else "NOT PREPARED (dropped from mix)"
        print(f"  {bucket:<26}{weight * 100:>5.0f}%   {status}")


def cmd_dataset_stats(args):
    from data_sources import compute_shard_stats
    from data_sources.tokenizer_stats import print_stats

    shards_root = os.path.join("data", "shards")
    if args.name:
        names = [args.name]
    else:
        names = sorted(d for d in os.listdir(shards_root)) if os.path.isdir(shards_root) else []
        if not names:
            print("No prepared datasets found under data/shards/. Run `dataset prepare` first.")
            return

    for name in names:
        for split in ("train", "validation"):
            shard_dir = os.path.join(shards_root, name, split)
            stats = compute_shard_stats(shard_dir)
            print(f"\n=== {name} / {split} ===")
            print_stats(stats)


def cmd_train(args):
    if args.stage == "instruction":
        from training.instruction_trainer import DEFAULT_DATA, train_instruction

        if not args.checkpoint:
            raise SystemExit(
                "--stage instruction requires --checkpoint <base checkpoint .pt> "
                "(Stage A weights to initialize from)"
            )
        train_instruction(
            base_checkpoint=args.checkpoint,
            data_path=args.instruction_data or DEFAULT_DATA,
            max_steps=args.max_steps,
        )
        return

    from training import train_model

    # Preset is required to know the training schedule/mix even when
    # resuming; it is not persisted separately from the checkpoint
    # metadata in this phase.
    preset_name = args.preset or "tiny_debug"
    train_model(preset_name=preset_name, resume=args.resume, use_wandb=args.wandb)


def _load_checkpoint_model(checkpoint_path):
    from inference import load_model_for_inference
    from training.trainer import detect_device

    device = detect_device()
    device = "cpu" if device == "mps" else device  # MPS lacks some ops used here
    model, config = load_model_for_inference(checkpoint_path, device=device)
    return model, config, device


def cmd_evaluate(args):
    import json as _json

    from evaluation import check_leakage, evaluate_model, print_report

    model, config, device = _load_checkpoint_model(args.checkpoint)
    print(f"Loaded {args.checkpoint} on {device} "
          f"({sum(p.numel() for p in model.parameters()):,} parameters)\n")

    leak = check_leakage()
    print(f"Leakage check: {'CLEAN' if leak.get('clean') else 'LEAKS FOUND'} "
          f"({leak.get('eval_questions')} eval questions vs "
          f"{leak.get('checked_shards')} training shards)\n")

    results = evaluate_model(model, device=device, max_new_tokens=args.max_new_tokens,
                              verbose=args.verbose)
    print_report(results)

    if args.output:
        with open(args.output, "w") as f:
            _json.dump({"checkpoint": args.checkpoint, "leakage": leak, "results": results},
                        f, indent=2)
        print(f"\nWrote results to {args.output}")


def cmd_compare(args):
    """Base vs Financial vs Instruction comparison (Part 18 / §34).

    Every number is produced by actually running the evaluation on each
    checkpoint. Missing checkpoints report "Not available" - never a
    placeholder score.
    """
    import json as _json

    from evaluation import evaluate_model

    targets = []
    for label, path in (("Base", args.base), ("Financial", args.financial),
                         ("Instruction", args.instruction)):
        if path:
            targets.append((label, path))

    if not targets:
        raise SystemExit("Provide at least one of --base / --financial / --instruction")

    columns = {}
    for label, path in targets:
        if not os.path.exists(path):
            columns[label] = None
            print(f"{label}: checkpoint not found at {path} - reporting Not available")
            continue
        model, config, device = _load_checkpoint_model(path)
        print(f"Evaluating {label} ({path}) on {device} ...")
        columns[label] = evaluate_model(model, device=device,
                                         max_new_tokens=args.max_new_tokens)

    categories = ["financial_qa", "numerical", "terminology", "reasoning"]
    labels = [label for label, _ in targets]

    print()
    print("=" * (26 + 14 * len(labels)))
    print("MODEL COMPARISON - Financial LLM POC Evaluation (measured)")
    print("=" * (26 + 14 * len(labels)))
    header = f"{'Metric':<26}" + "".join(f"{lab:>14}" for lab in labels)
    print(header)
    print("-" * len(header))
    for category in categories:
        row = f"{category:<26}"
        for label in labels:
            results = columns.get(label)
            if not results:
                row += f"{'Not available':>14}"
            else:
                acc = results["categories"].get(category, {}).get("accuracy")
                row += f"{'Not available' if acc is None else f'{acc:.2f}%':>14}"
        print(row)
    print("-" * len(header))
    row = f"{'OVERALL':<26}"
    for label in labels:
        results = columns.get(label)
        acc = results["overall"]["accuracy"] if results else None
        row += f"{'Not available' if acc is None else f'{acc:.2f}%':>14}"
    print(row)
    print("=" * len(header))

    if args.output:
        with open(args.output, "w") as f:
            _json.dump({lab: (columns[lab] or "Not available") for lab in labels}, f, indent=2)
        print(f"\nWrote comparison to {args.output}")


def cmd_chat(args):
    from app.backend.services.chat_service import FinancialChat

    model, config, device = _load_checkpoint_model(args.checkpoint)
    print(f"Loaded {args.checkpoint} on {device} "
          f"({sum(p.numel() for p in model.parameters()):,} parameters)")

    store = None
    if args.document:
        from rag import DocumentStore
        store = DocumentStore()
        info = store.add_document(args.document)
        print(f"Loaded document: {info}")

    chat = FinancialChat(model=model, device=device, document_store=store,
                          max_new_tokens=args.max_new_tokens)

    print("\nFinancial Assistant (type 'exit' to quit)")
    print("Generation comes from this repository's own PyTorch model.\n")

    if args.query:
        response = chat.ask(args.query)
        print(f"USER: {args.query}\n")
        print(f"MODEL:\n{response.render()}")
        return

    while True:
        try:
            query = input("USER: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if query.lower() in {"exit", "quit"}:
            break
        if not query:
            continue
        response = chat.ask(query)
        print(f"\nMODEL:\n{response.render()}\n")


def cmd_inspect(args):
    import json as _json

    from app.backend.services.inspector import (
        inspect_architecture, inspect_forward, inspect_moe_routing, inspect_tokens,
    )

    if args.what == "tokens":
        print(_json.dumps(inspect_tokens(args.text), indent=2))
        return

    model, config, device = _load_checkpoint_model(args.checkpoint)

    if args.what == "architecture":
        print(_json.dumps(inspect_architecture(model, device), indent=2))
    elif args.what == "moe":
        print(_json.dumps(inspect_moe_routing(model, args.text, layer=args.layer,
                                               device=device), indent=2))
    elif args.what == "forward":
        print(_json.dumps(inspect_forward(model, args.text, device=device), indent=2))


def build_parser():
    parser = argparse.ArgumentParser(description="DeepSeek-V3-Inspired Financial LLM")
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("demo", help="Run a quick forward-pass demo with the unified config")
    subparsers.add_parser("runtime", help="Detect the actually accessible compute runtime")

    dataset_parser = subparsers.add_parser("dataset", help="Dataset operations")
    dataset_sub = dataset_parser.add_subparsers(dest="dataset_command")

    dataset_sub.add_parser("list", help="List all registered datasets")

    p_download = dataset_sub.add_parser("download", help="Probe a dataset source")
    p_download.add_argument("--name", required=True)

    p_prepare = dataset_sub.add_parser("prepare", help="Clean, split, tokenize, and shard a dataset")
    p_prepare.add_argument("--name", required=True)
    p_prepare.add_argument("--max-tokens", type=int, required=True)
    p_prepare.add_argument("--max-records", type=int, default=None)

    p_stats = dataset_sub.add_parser("stats", help="Show tokenizer statistics for prepared shards")
    p_stats.add_argument("--name", default=None)

    p_dash = dataset_sub.add_parser("dashboard", help="Dataset dashboard: provenance, counts, mixture")
    p_dash.add_argument("--preset", default="tiny_debug")

    p_train = subparsers.add_parser("train", help="Train the model")
    p_train.add_argument("--preset", choices=["tiny_debug", "small", "financial_poc"], default=None)
    p_train.add_argument("--resume", default=None, help="Path to a checkpoint_*.pt to resume from")
    p_train.add_argument("--stage", choices=["base", "instruction"], default="base",
                          help="base = pretraining, instruction = financial instruction tuning")
    p_train.add_argument("--checkpoint", default=None,
                          help="Stage A checkpoint to initialize instruction tuning from")
    p_train.add_argument("--instruction-data", default=None,
                          help="Instruction JSONL for --stage instruction")
    p_train.add_argument("--max-steps", type=int, default=100,
                          help="Steps for --stage instruction")
    p_train.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")

    p_eval = subparsers.add_parser("evaluate", help="Run the Financial LLM POC Evaluation")
    p_eval.add_argument("--checkpoint", required=True)
    p_eval.add_argument("--max-new-tokens", type=int, default=48)
    p_eval.add_argument("--output", default=None, help="Write results JSON here")
    p_eval.add_argument("--verbose", action="store_true")

    p_chat = subparsers.add_parser("chat", help="Financial chat using the trained model")
    p_chat.add_argument("--checkpoint", required=True)
    p_chat.add_argument("--document", default=None, help="Financial document to load for RAG")
    p_chat.add_argument("--query", default=None, help="Single query (non-interactive)")
    p_chat.add_argument("--max-new-tokens", type=int, default=60)

    p_cmp = subparsers.add_parser("compare", help="Compare base vs financial vs instruction models")
    p_cmp.add_argument("--base", default=None)
    p_cmp.add_argument("--financial", default=None)
    p_cmp.add_argument("--instruction", default=None)
    p_cmp.add_argument("--max-new-tokens", type=int, default=16)
    p_cmp.add_argument("--output", default=None)

    p_inspect = subparsers.add_parser("inspect", help="Inspect model internals")
    p_inspect.add_argument("what", choices=["tokens", "architecture", "moe", "forward"])
    p_inspect.add_argument("--checkpoint", default=None)
    p_inspect.add_argument("--text", default="What is EBITDA?")
    p_inspect.add_argument("--layer", type=int, default=0)

    return parser


def main():
    parser = build_parser()

    if len(sys.argv) < 2:
        demo()
        return

    args = parser.parse_args()

    if args.command == "demo":
        demo()
    elif args.command == "runtime":
        cmd_runtime(args)
    elif args.command == "dataset":
        if args.dataset_command == "list":
            cmd_dataset_list(args)
        elif args.dataset_command == "download":
            cmd_dataset_download(args)
        elif args.dataset_command == "prepare":
            cmd_dataset_prepare(args)
        elif args.dataset_command == "stats":
            cmd_dataset_stats(args)
        elif args.dataset_command == "dashboard":
            cmd_dataset_dashboard(args)
        else:
            parser.parse_args(["dataset", "--help"])
    elif args.command == "train":
        cmd_train(args)
    elif args.command == "evaluate":
        cmd_evaluate(args)
    elif args.command == "chat":
        cmd_chat(args)
    elif args.command == "compare":
        cmd_compare(args)
    elif args.command == "inspect":
        cmd_inspect(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
