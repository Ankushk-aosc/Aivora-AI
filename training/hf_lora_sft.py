"""LoRA fine-tuning of an open pretrained model on the project's finance data.

Why this exists: the from-scratch 101M model plateaued - val loss gained 0.04
over a whole 46k-step session and instruction tuning overfitted at two
different learning rates, leaving the evaluation score inside its own noise
band (1-5 of 45). A 1.5B model pretrained on trillions of tokens starts from a
far better place; LoRA adapts it on the same finance data on the same free
Kaggle T4.

Only small adapter matrices are trained (~1% of parameters), so the base
weights stay frozen and the result is a ~50 MB adapter rather than a 3 GB
model.

The prompt format is the base model's own chat template, not this project's
"### Instruction:" format - an instruct-tuned model answers best in the format
it was trained on, and the same template is used again at evaluation time.
"""

import os

import torch

DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

SYSTEM_PROMPT = ("You are a financial analyst assistant. Answer concisely and "
                 "correctly. If a question needs a calculation, show the formula.")


def build_chat_text(tokenizer, question: str, answer: str = None, system: str = SYSTEM_PROMPT):
    """Render one example with the model's own chat template.

    With answer=None the result ends at the point the assistant should start
    writing, which is what generation needs."""
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": question}]
    if answer is None:
        return tokenizer.apply_chat_template(messages, tokenize=False,
                                             add_generation_prompt=True)
    prompt = tokenizer.apply_chat_template(messages, tokenize=False,
                                           add_generation_prompt=True)
    return prompt, prompt + answer + tokenizer.eos_token


def tokenize_records(tokenizer, records, max_len: int = 1024):
    """Tokenize {instruction,input,output} rows into input_ids + labels.

    Prompt tokens are labelled -100 so the loss is taken on the answer only -
    the model learns to answer, not to repeat the question."""
    rows = []
    for rec in records:
        question = (rec.get("instruction") or "").strip()
        context = (rec.get("input") or "").strip()
        if context:
            question = f"{question}\n\n{context}"
        answer = (rec.get("output") or "").strip()
        if not question or not answer:
            continue

        prompt, full = build_chat_text(tokenizer, question, answer)
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        full_ids = tokenizer(full, add_special_tokens=False)["input_ids"][:max_len]
        if len(full_ids) <= len(prompt_ids):  # answer was truncated away
            continue
        labels = list(full_ids)
        for i in range(min(len(prompt_ids), len(labels))):
            labels[i] = -100
        rows.append({"input_ids": full_ids, "labels": labels})
    return rows


class PadCollator:
    """Pad a batch to its longest row; -100 pads labels so padding is ignored."""

    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, features):
        width = max(len(f["input_ids"]) for f in features)
        input_ids, labels, attention = [], [], []
        for f in features:
            pad = width - len(f["input_ids"])
            input_ids.append(f["input_ids"] + [self.pad_token_id] * pad)
            labels.append(f["labels"] + [-100] * pad)
            attention.append([1] * len(f["input_ids"]) + [0] * pad)
        return {"input_ids": torch.tensor(input_ids), "labels": torch.tensor(labels),
                "attention_mask": torch.tensor(attention)}


def load_base_model(model_name: str = DEFAULT_MODEL, device_map="auto"):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map=device_map,
    )
    return model, tokenizer


def make_answerer(model, tokenizer, max_new_tokens: int = 96, temperature: float = 0.0):
    """A generate_fn(question) -> answer for evaluation.generic.

    Greedy by default: the project's own evaluation swung between 1 and 5 of 45
    on identical weights because sampling was on, which made before/after
    comparisons meaningless."""

    def answer(question: str) -> str:
        text = build_chat_text(tokenizer, question)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else None,
                pad_token_id=tokenizer.pad_token_id,
            )
        return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                                skip_special_tokens=True).strip()

    return answer


def attach_lora(model, r: int = 16, alpha: int = 32, dropout: float = 0.05, targets=None):
    from peft import LoraConfig, get_peft_model

    targets = targets or ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"]
    config = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout, bias="none",
                        task_type="CAUSAL_LM", target_modules=targets)
    model = get_peft_model(model, config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"LoRA: training {trainable:,} of {total:,} parameters "
          f"({100 * trainable / total:.2f}%)")
    return model


def train_lora(model, tokenizer, train_rows, val_rows, output_dir,
               epochs: float = 1.0, batch_size: int = 2, grad_accum: int = 8,
               learning_rate: float = 2e-4, eval_steps: int = 100,
               max_train_seconds: float = None, seed: int = 42):
    from transformers import Trainer, TrainingArguments

    os.makedirs(output_dir, exist_ok=True)
    wanted = {
        "output_dir": output_dir,
        "num_train_epochs": epochs,
        "per_device_train_batch_size": batch_size,
        "per_device_eval_batch_size": batch_size,
        "gradient_accumulation_steps": grad_accum,
        "learning_rate": learning_rate,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.03,
        "logging_steps": 25,
        "eval_strategy": "steps",
        "evaluation_strategy": "steps",
        "eval_steps": eval_steps,
        "save_strategy": "steps",
        "save_steps": eval_steps,
        "save_total_limit": 2,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "fp16": torch.cuda.is_available(),
        "max_grad_norm": 1.0,
        "gradient_checkpointing": True,
        "report_to": [],
        "seed": seed,
    }
    # Kaggle's transformers version is not this machine's: "eval_strategy" was
    # "evaluation_strategy" before 4.41, and "warmup_ratio" is gone in 5.x.
    # Pass only the names this installed version actually defines.
    import dataclasses
    supported = {f.name for f in dataclasses.fields(TrainingArguments)}
    dropped = sorted(k for k in wanted if k not in supported)
    args_kwargs = {k: v for k, v in wanted.items() if k in supported}
    if "warmup_ratio" in dropped and "warmup_steps" in supported:
        args_kwargs["warmup_steps"] = 50
    if dropped:
        print(f"TrainingArguments: this transformers ({__import__('transformers').__version__}) "
              f"does not accept {dropped}; using the supported equivalents.")
    args = TrainingArguments(**args_kwargs)
    callbacks = []
    if max_train_seconds:
        from transformers import TrainerCallback

        class TimeBudget(TrainerCallback):
            """Kaggle kills a session at 12 h and a killed run keeps no
            outputs, so stop early and let the notebook finish normally."""

            def __init__(self, budget):
                import time
                self.budget, self.start, self.time = budget, time.time(), time
            def on_step_end(self, args, state, control, **kwargs):
                if self.time.time() - self.start > self.budget:
                    print(f"Time budget of {self.budget:.0f}s reached at step "
                          f"{state.global_step}; stopping.")
                    control.should_training_stop = True
                return control

        callbacks.append(TimeBudget(max_train_seconds))

    trainer = Trainer(model=model, args=args, train_dataset=train_rows,
                      eval_dataset=val_rows, data_collator=PadCollator(tokenizer.pad_token_id),
                      callbacks=callbacks)
    result = trainer.train()
    metrics = trainer.evaluate()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    return trainer, result, metrics
