import json
import os

import torch
import tiktoken
from models import DeepSeekConfig, DeepSeekV3


def load_model_for_inference(checkpoint_path, device=None):
    """Load a model + its own saved config from a checkpoint_step.pt /
    checkpoint_step.json pair (Part 2: inference must use the exact
    config the checkpoint was trained with, not a hardcoded one)."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    meta_path = checkpoint_path.rsplit(".pt", 1)[0] + ".json"
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            metadata = json.load(f)
        config = DeepSeekConfig.from_dict(metadata["model_config"])
    else:
        config = DeepSeekConfig.default()

    model = DeepSeekV3(config)
    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model, config


def generate_text(checkpoint_path, prompt, max_tokens=100, temperature=0.8, top_k=50, device=None):
    """Generate text from a prompt using a trained checkpoint.

    Loads the model config that the checkpoint was actually trained with
    (see load_model_for_inference), rather than assuming a hardcoded config.
    """
    model, config = load_model_for_inference(checkpoint_path, device=device)
    device = next(model.parameters()).device

    # Tokenize input
    enc = tiktoken.get_encoding("gpt2")
    context = torch.tensor(enc.encode_ordinary(prompt)).unsqueeze(0).to(device)

    # Generate
    with torch.no_grad():
        generated = model.generate(context, max_tokens, temperature, top_k)

    # Decode and return
    result = enc.decode(generated.squeeze().tolist())
    return result


def run_inference_examples(checkpoint_path):
    """Run inference examples with different prompts using a trained checkpoint."""
    try:
        test_prompts = [
            "Once upon a time",
            "The little girl",
            "In a magical forest",
            "The brave knight",
        ]

        print("=" * 50)
        print("DEEPSEEK-V3 FINANCIAL LLM INFERENCE EXAMPLES")
        print("=" * 50)

        for prompt in test_prompts:
            result = generate_text(checkpoint_path, prompt, max_tokens=80, temperature=0.7, top_k=40)
            print(f"\nPrompt: '{prompt}'")
            print("Generated:", result)
            print("-" * 30)

    except FileNotFoundError:
        print(f"Checkpoint '{checkpoint_path}' not found. Please train the model first.")
    except Exception as e:
        print(f"Error during inference: {e}")