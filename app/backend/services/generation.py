"""Text-generation backends for the chat service.

The chat service used to call this project's DeepSeekV3 directly - reading
`model.config.block_size`, using its custom `generate()` signature and the
GPT-2 tokenizer. That made the app unable to serve anything else, including
Qwen2.5-1.5B-Instruct, which scores 38/45 on the project's own evaluation
where the from-scratch model scores 1-5/45.

A backend is anything with `.generate(prompt, **options) -> str` and a
`.describe()` for /api/model/status:

* DeepSeekBackend - this project's own checkpoints (unchanged behaviour).
* HFBackend       - any Hugging Face causal LM, optionally with a LoRA
                    adapter, prompted with the model's own chat template.
"""

import os


class DeepSeekBackend:
    """This project's DeepSeekV3 checkpoints."""

    kind = "deepseek"

    def __init__(self, model, device="cpu", checkpoint=None):
        from data_sources.tokenizer import get_encoding

        self.model = model
        self.device = device
        self.checkpoint = checkpoint
        self.enc = get_encoding()

    def generate(self, prompt, max_new_tokens=64, temperature=0.7, top_k=40,
                 top_p=0.9, repetition_penalty=1.3):
        import torch

        ids = self.enc.encode_ordinary(prompt)
        max_ctx = self.model.config.block_size - max_new_tokens
        if len(ids) > max_ctx:
            ids = ids[-max_ctx:]
        context = torch.tensor(ids, dtype=torch.long, device=self.device).unsqueeze(0)
        with torch.no_grad():
            out = self.model.generate(
                context, max_new_tokens, temperature=temperature, top_k=top_k,
                top_p=top_p, repetition_penalty=repetition_penalty,
                stop_on_repetition=True,
            )
        return self.enc.decode(out[0, len(ids):].tolist()).strip()

    def describe(self):
        return {
            "kind": self.kind,
            "checkpoint": self.checkpoint,
            "device": self.device,
            "parameters": sum(p.numel() for p in self.model.parameters()),
            "context_length": getattr(self.model.config, "block_size", None),
        }


class HFBackend:
    """A Hugging Face causal LM, optionally with a LoRA adapter on top.

    `prompt` is sent as the user message through the model's chat template, so
    the model is prompted the way it was trained - the same reason the
    project's own evaluation had to switch prompt styles for tuned
    checkpoints."""

    kind = "huggingface"

    SYSTEM_PROMPT = ("You are a financial analyst assistant. Answer concisely and "
                     "correctly. If a question needs a calculation, show the formula.")

    def __init__(self, model_name, adapter_dir=None, device=None, dtype=None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_name = model_name
        self.adapter_dir = adapter_dir
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        if dtype is None:
            dtype = torch.float16 if device == "cuda" else torch.float32

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
        if adapter_dir:
            from peft import PeftModel

            if not os.path.isdir(adapter_dir):
                raise ValueError(f"Adapter directory not found: {adapter_dir}")
            self.model = PeftModel.from_pretrained(self.model, adapter_dir)
        self.model.to(device)
        self.model.eval()

    def generate(self, prompt, max_new_tokens=128, temperature=0.0, top_k=None,
                 top_p=None, repetition_penalty=None):
        import torch

        messages = [{"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False,
                                                  add_generation_prompt=True)
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        kwargs = {"max_new_tokens": max_new_tokens,
                  "pad_token_id": self.tokenizer.pad_token_id}
        # Greedy by default: sampling made the project's own scores swing
        # between 1 and 5 of 45 on identical weights.
        if temperature and temperature > 0:
            kwargs.update(do_sample=True, temperature=temperature)
            if top_k:
                kwargs["top_k"] = top_k
            if top_p:
                kwargs["top_p"] = top_p
        else:
            kwargs["do_sample"] = False
        if repetition_penalty:
            kwargs["repetition_penalty"] = repetition_penalty
        with torch.no_grad():
            out = self.model.generate(**inputs, **kwargs)
        return self.tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                                     skip_special_tokens=True).strip()

    def describe(self):
        return {
            "kind": self.kind,
            "model": self.model_name,
            "adapter": self.adapter_dir,
            "device": self.device,
            "parameters": sum(p.numel() for p in self.model.parameters()),
            "context_length": getattr(self.model.config, "max_position_embeddings", None),
        }
