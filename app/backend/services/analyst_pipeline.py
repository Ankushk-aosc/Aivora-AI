"""The Financial Analyst, served by the Phase 4 tool pipeline.

Why. The Analyst's weakest component is the from-scratch model. Measured on this
project's frozen held-out set, that family of checkpoints scores 13.0% on
extraction and invents a value on 36.8% of answerable items; the Phase 4
pipeline behind Qwen2.5-1.5B-Instruct scores 87.0% extraction, 100% calculation
and invents a value on 0 of 261. See reports/phase4/PHASE4_V2_REPORT.md.

What the pipeline guarantees, whatever model backs it:

  * a value reaches the user only with a source span validated against the
    document it came from
  * every piece of arithmetic is done in Python, never by the model
  * anything that fails validation becomes an abstention, not a guess

Those guarantees are the reason to switch. They are not, however, enough on
their own: a pipeline behind a model that cannot follow the extraction format
abstains on everything, which is safe and useless.

Measured here, on the from-scratch checkpoint, 2026-10-05:

    Q: What is revenue?   (context states Revenue: $10,000,000)
    model emits:  "Revenue = $5,200,000 - the EBIT entry."
    pipeline:     abstains - output was not parseable JSON

The model invented $5.2M and the span rule stopped it, which is the system
working. But coverage was 0 of 3 questions. So this module REFUSES to serve the
Analyst unless a model that can actually drive the pipeline is available, and
says why. An Analyst that refuses every question is a regression, and silently
becoming one would be worse than not switching at all.

    readiness()                  -> can the pipeline serve, and if not why
    answer(question, context)    -> a pipeline answer, or None if not serving
"""

import os
import platform

DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

# Approximate resident memory for CPU inference, in GB. Keyed to the models
# this project has actually measured (reports/phase3/PHASE3_REPORT.md).
MODEL_MEMORY_GB = {
    "Qwen/Qwen2.5-1.5B-Instruct": 7.0,
    "Qwen/Qwen2.5-0.5B-Instruct": 2.5,
    "HuggingFaceTB/SmolLM2-1.7B-Instruct": 8.0,
    "HuggingFaceTB/SmolLM2-360M-Instruct": 2.0,
    "HuggingFaceTB/SmolLM2-135M-Instruct": 1.1,
}

SYSTEM_PROMPT = (
    "You extract values from financial text. You reply with a single JSON "
    "object and nothing else - no explanation, no markdown fence, no commentary. "
    "Copy values exactly as they appear. Never calculate. Never invent a number."
)

_STATE = {"pipeline": None, "model_name": None, "error": None}


def configured_model():
    return os.environ.get("AIVORA_ANALYST_MODEL", DEFAULT_MODEL)


def enabled():
    """Opt-in. Unset or '0' leaves the existing Analyst untouched."""
    return os.environ.get("AIVORA_ANALYST_PIPELINE", "0").lower() in (
        "1", "true", "yes", "on")


def free_memory_gb():
    if platform.system() == "Windows":
        import ctypes

        class Status(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong),
                        ("dwMemoryLoad", ctypes.c_ulong),
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
    except Exception:                                       # noqa: BLE001
        return float("inf")


def readiness():
    """Whether the pipeline can serve the Analyst, in terms a reader can act on.

    Reports the shortfall rather than a bare False, because "not enough memory
    on this machine" and "transformers is not installed" need different
    responses from whoever is running it.
    """
    model = configured_model()
    needed = MODEL_MEMORY_GB.get(model)
    free = free_memory_gb()
    state = {
        "enabled": enabled(),
        "model": model,
        "free_memory_gb": round(free, 1),
        "needed_memory_gb": needed,
        "serving": False,
        "engine": "existing analyst",
    }

    if not enabled():
        state["reason"] = (
            "Not enabled. Set AIVORA_ANALYST_PIPELINE=1 to route the Analyst "
            "through the tool pipeline.")
        return state

    try:
        import transformers                                 # noqa: F401
    except ImportError:
        state["reason"] = ("transformers is not installed, so no pipeline model "
                           "can be loaded.")
        return state

    if needed is None:
        state["reason"] = (
            f"{model} has no measured memory requirement in this project. It may "
            f"work; it has not been verified here.")
        state["serving"] = _STATE["pipeline"] is not None
        return state

    if free < needed:
        state["reason"] = (
            f"{model} needs about {needed} GB and {free:.1f} GB is free. The "
            f"Analyst keeps its existing engine rather than becoming one that "
            f"abstains on every question.")
        return state

    state["serving"] = True
    state["engine"] = f"tool pipeline + {model}"
    state["reason"] = "Ready."
    return state


def _build():
    """Load the model and bind the pipeline. Raises if it cannot."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from pipeline.tool_pipeline import ToolPipeline

    model_name = configured_model()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype)
    model.to(device)
    model.eval()

    def llm(prompt):
        messages = [{"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False,
                                             add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=160, do_sample=False,
                                 temperature=None, top_p=None,
                                 pad_token_id=tokenizer.eos_token_id)
        generated = out[0, inputs["input_ids"].shape[1]:]
        return tokenizer.decode(generated, skip_special_tokens=True).strip()

    _STATE["pipeline"] = ToolPipeline(llm)
    _STATE["model_name"] = model_name
    return _STATE["pipeline"]


def get_pipeline(llm=None):
    """The bound pipeline, or None when it cannot serve.

    `llm` injects a callable directly, which is how the tests drive the real
    pipeline without loading a model.
    """
    if llm is not None:
        from pipeline.tool_pipeline import ToolPipeline

        _STATE["pipeline"] = ToolPipeline(llm)
        _STATE["model_name"] = "injected"
        return _STATE["pipeline"]

    if _STATE["pipeline"] is not None:
        return _STATE["pipeline"]
    if not readiness()["serving"]:
        return None
    try:
        return _build()
    except Exception as error:                              # noqa: BLE001
        _STATE["error"] = f"{type(error).__name__}: {error}"
        return None


def reset():
    """Drop the loaded pipeline - used by tests."""
    _STATE.update({"pipeline": None, "model_name": None, "error": None})


def answer(question, context, llm=None):
    """A pipeline answer for a question about a document, or None.

    None means "this engine is not serving"; the caller keeps its existing
    behaviour. An abstention is NOT None - it is a real answer, and the caller
    should present it as a refusal rather than falling through to a path that
    might guess.
    """
    pipeline = get_pipeline(llm=llm)
    if pipeline is None:
        return None
    result = pipeline.answer(question, context or "")
    return {
        "answer": result.answer,
        "abstained": result.abstained,
        "component": result.component,
        "value": result.value,
        "operation": result.operation,
        "source_spans": result.source_spans,
        "reason": result.reason,
        "engine": f"tool pipeline + {_STATE['model_name']}",
        "arithmetic": "python",
        "guarantee": ("every value carries a source span validated against the "
                      "document; all arithmetic is computed in Python"),
    }
