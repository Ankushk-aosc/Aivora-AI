"""
Simple script to run inference with the DeepSeek-style financial LLM.
Change the checkpoint path and prompts below and run this file.
"""

import sys

from inference import generate_text

DEFAULT_CHECKPOINT = "checkpoints/base/checkpoint_latest.pt"


if __name__ == "__main__":
    print("=" * 60)
    print("DEEPSEEK-V3 FINANCIAL LLM TEXT GENERATION")
    print("=" * 60)

    checkpoint_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CHECKPOINT

    # ============================================
    # CHANGE THESE PROMPTS TO WHATEVER YOU WANT!
    # ============================================

    my_prompts = [
        "What is EBITDA?",
        "Revenue is 500 and EBITDA is 100. Calculate the EBITDA margin.",
        "The little girl found a magic",
        "In the future, artificial intelligence will",
    ]

    max_tokens = 80
    temperature = 0.8
    top_k = 50

    try:
        for i, prompt in enumerate(my_prompts, 1):
            print(f"\n{i}. Prompt: '{prompt}'")
            print("-" * 40)
            result = generate_text(checkpoint_path, prompt, max_tokens=max_tokens,
                                    temperature=temperature, top_k=top_k)
            print(f"Generated: {result}")
            print()
    except FileNotFoundError:
        print(f"Checkpoint '{checkpoint_path}' not found. Train a model first:")
        print("  python main.py train --preset tiny_debug")

    print("=" * 60)
    print("DONE!")
    print("=" * 60)
