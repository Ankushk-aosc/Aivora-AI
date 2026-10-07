# Task 7 — the "no external service dependency" claim

**Verdict: true for the shipped application in its default configuration;
false as an unqualified statement about the repository.**

## Network calls and model download paths found in the code

| File | Call | Reached in a default run? |
|---|---|---|
| `app/backend/services/generation.py` | `AutoModel.from_pretrained` | **No** — only on the `HFBackend` path, which a default install does not select. The default backend loads a local checkpoint from disk. |
| `app/backend/services/analyst_pipeline.py` | `AutoModel.from_pretrained` | **No** — gated behind `AIVORA_ANALYST_PIPELINE`, which defaults to off, and behind a memory check. |
| `scripts/eval_base_models.py` | `from_pretrained` | No — evaluation tooling, not the application. |
| `scripts/run_tool_pipeline_eval.py` | `from_pretrained` | No — evaluation tooling. |
| `data_sources/huggingface_loader.py` | `load_dataset` | No — corpus preparation. |
| `data_sources/build_instruction_dataset.py` | `load_dataset` | No — corpus preparation. |
| `scripts/download_kernel_file.py` | `requests` | No — Kaggle result retrieval. |
| `scripts/build_routing_pilot_notebook.py` | `load_dataset` | No — generates a notebook that runs elsewhere. |

## What this means

Serving the application — `python -m app.backend.server` — makes **no outbound
network call**. The HTTP layer is the standard library, the model is read from
`checkpoints/`, retrieval is a local TF-IDF index over local files, and storage
is local SQLite and JSON. There is no API key anywhere in the runtime path, and
the frontend calls only its own origin.

The qualification the documentation should carry: **the moment the Analyst
pipeline is enabled, the claim stops holding.** Setting
`AIVORA_ANALYST_PIPELINE=1` causes `from_pretrained` to download a model from
the Hugging Face Hub on first use — roughly 3 GB for Qwen2.5-1.5B — unless it is
already cached. That is an external service dependency, introduced by a
configuration flag the documentation recommends setting on adequate hardware.

Suggested wording: "No external service dependency in the default
configuration. Enabling the Analyst pipeline downloads a model from the Hugging
Face Hub on first use."
