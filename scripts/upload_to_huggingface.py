"""Publish the hf_export bundle to a free Hugging Face account.

Creates (or updates) two repos:

  * a MODEL repo  - the weights, config, model code and model card
  * a SPACE       - the Gradio demo, which runs on free CPU hardware

Log in first, in your own terminal (this script never handles your token -
paste it at the CLI's own hidden prompt, or export it as HF_TOKEN):

    Windows:  .\\deepseek_env\\Scripts\\hf.exe auth login
    Linux/Mac: hf auth login

huggingface_hub 1.x dropped the old `python -m huggingface_hub.commands.*`
entry point, hence the explicit path to the venv's hf executable.

usage:
    python scripts/upload_to_huggingface.py --name aivora-financial-llm
    python scripts/upload_to_huggingface.py --name aivora-financial-llm --private
    python scripts/upload_to_huggingface.py --name aivora-financial-llm --model-only
"""

import argparse
import os
import sys


def login_command():
    """The exact login command for THIS install.

    huggingface_hub 1.x removed `python -m huggingface_hub.commands.*`, and on
    Windows the `hf` executable lives inside the venv rather than on PATH, so
    print the path that actually exists here instead of a generic hint."""
    candidates = [
        os.path.join("deepseek_env", "Scripts", "hf.exe"),
        os.path.join("deepseek_env", "Scripts", "huggingface-cli.exe"),
        os.path.join("deepseek_env", "bin", "hf"),
        os.path.join(os.path.dirname(sys.executable), "hf.exe"),
        os.path.join(os.path.dirname(sys.executable), "hf"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            prefix = ".\\" if os.name == "nt" and not os.path.isabs(path) else ""
            return f"{prefix}{path} auth login"
    return "hf auth login"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="hf_export", help="bundle built by export_for_huggingface.py")
    ap.add_argument("--name", required=True, help="repo name, e.g. aivora-financial-llm")
    ap.add_argument("--private", action="store_true", help="create private repos")
    ap.add_argument("--model-only", action="store_true", help="skip the Space")
    ap.add_argument("--space-only", action="store_true", help="skip the model repo")
    args = ap.parse_args()

    from huggingface_hub import HfApi
    from huggingface_hub.errors import HfHubHTTPError

    if not os.path.isdir(args.dir):
        sys.exit(f"{args.dir}/ not found - run scripts/export_for_huggingface.py first")

    api = HfApi()
    try:
        me = api.whoami()
    except Exception as e:
        sys.exit(f"Not logged in to Hugging Face.\n\n"
                 f"Run this in your terminal, then re-run this script:\n"
                 f"    {login_command()}\n\n"
                 f"Paste your token at its 'Enter your token:' prompt (it stays hidden).\n"
                 f"Alternatively set HF_TOKEN in your shell before running this script.\n"
                 f"Create a token at https://huggingface.co/settings/tokens (type: Write).\n"
                 f"(details: {type(e).__name__}: {e})")
    user = me["name"]
    print(f"Logged in as: {user}")

    visibility = "private" if args.private else "PUBLIC"
    model_id, space_id = f"{user}/{args.name}", f"{user}/{args.name}-demo"
    print(f"About to publish ({visibility}):")
    if not args.space_only:
        print(f"  model: https://huggingface.co/{model_id}")
    if not args.model_only:
        print(f"  space: https://huggingface.co/spaces/{space_id}")

    size = sum(os.path.getsize(os.path.join(dp, f))
               for dp, _, fs in os.walk(args.dir) for f in fs)
    print(f"  uploading {size / 1e6:.0f} MB from {args.dir}/")

    if not args.space_only:
        api.create_repo(model_id, repo_type="model", private=args.private, exist_ok=True)
        api.upload_folder(folder_path=args.dir, repo_id=model_id, repo_type="model",
                          ignore_patterns=["__pycache__/*", "*.pyc"])
        print(f"model published: https://huggingface.co/{model_id}")

    if not args.model_only:
        try:
            api.create_repo(space_id, repo_type="space", private=args.private,
                            exist_ok=True, space_sdk="gradio")
        except HfHubHTTPError as e:
            sys.exit(f"Could not create the Space: {e}")
        api.upload_folder(folder_path=args.dir, repo_id=space_id, repo_type="space",
                          ignore_patterns=["__pycache__/*", "*.pyc"])
        print(f"space published: https://huggingface.co/spaces/{space_id}")
        print("The Space builds for a few minutes before it answers.")


if __name__ == "__main__":
    main()
