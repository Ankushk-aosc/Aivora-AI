"""Single registry of every dataset the pipeline is allowed to use.

Every entry records exactly what Part 7/37 of the project spec requires:
the Hugging Face id, config/subset, split, license, source URL, revision
(when pinned), the fields actually consumed, and a verification_status.

VERIFIED   -> the dataset id was confirmed to exist on Hugging Face and its
              license was read from the dataset's own card metadata.
UNVERIFIED -> the dataset id looks right but the license could not be
              confirmed from the card metadata. These are excluded from
              default dataset mixes and must be opted into explicitly.
"""

from dataclasses import dataclass, field
from typing import Optional

VERIFIED = "VERIFIED"
UNVERIFIED = "REQUIRES DATASET VERIFICATION"


@dataclass
class DatasetEntry:
    name: str
    hf_id: str
    category: str
    license: str
    source_url: str
    verification_status: str
    subset: Optional[str] = None
    split: str = "train"
    revision: Optional[str] = None
    fields_used: list = field(default_factory=list)
    notes: str = ""


REGISTRY = {
    # ------------------------------------------------------------------
    # General-purpose text
    # ------------------------------------------------------------------
    "fineweb_edu": DatasetEntry(
        name="fineweb_edu",
        hf_id="HuggingFaceFW/fineweb-edu",
        subset="CC-MAIN-2024-51",
        split="train",
        category="general",
        license="odc-by",
        source_url="https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu",
        verification_status=VERIFIED,
        fields_used=["text"],
        notes="General web-text pretraining data, filtered for educational quality.",
    ),
    "tinystories": DatasetEntry(
        name="tinystories",
        hf_id="roneneldan/TinyStories",
        split="train",
        category="general",
        license="cdla-sharing-1.0",
        source_url="https://huggingface.co/datasets/roneneldan/TinyStories",
        verification_status=VERIFIED,
        fields_used=["text"],
        notes="Architecture validation / debugging / small experiments only, not the "
        "primary financial training dataset.",
    ),
    # ------------------------------------------------------------------
    # Financial educational text / terminology
    # ------------------------------------------------------------------
    "financial_text_investopedia": DatasetEntry(
        name="financial_text_investopedia",
        hf_id="FinLang/investopedia-instruction-tuning-dataset",
        split="train",
        category="financial_text",
        license="cc-by-nc-4.0",
        source_url="https://huggingface.co/datasets/FinLang/investopedia-instruction-tuning-dataset",
        verification_status=VERIFIED,
        fields_used=["context", "question", "answer"],
        notes="Non-commercial license (CC-BY-NC-4.0) - research/education use only.",
    ),
    # ------------------------------------------------------------------
    # Financial QA
    # ------------------------------------------------------------------
    "financial_qa_sujet": DatasetEntry(
        name="financial_qa_sujet",
        hf_id="sujet-ai/Sujet-Finance-Instruct-177k",
        split="train",
        category="financial_qa",
        license="apache-2.0",
        source_url="https://huggingface.co/datasets/sujet-ai/Sujet-Finance-Instruct-177k",
        verification_status=VERIFIED,
        fields_used=["user_prompt", "answer"],
        notes="Broad finance QA/instruction dataset built from multiple public sources.",
    ),
    # ------------------------------------------------------------------
    # Financial reports (real SEC filings)
    # ------------------------------------------------------------------
    "financial_reports_sec": DatasetEntry(
        name="financial_reports_sec",
        hf_id="JanosAudran/financial-reports-sec",
        split="train",
        category="financial_reports",
        license="apache-2.0",
        source_url="https://huggingface.co/datasets/JanosAudran/financial-reports-sec",
        verification_status=VERIFIED,
        fields_used=["sentence"],
        notes="Sentences extracted from real SEC EDGAR annual/quarterly filings.",
    ),
    # ------------------------------------------------------------------
    # Financial reasoning / numerical finance
    # ------------------------------------------------------------------
    "financial_reasoning_finqa": DatasetEntry(
        name="financial_reasoning_finqa",
        hf_id="ibm-research/finqa",
        split="train",
        category="financial_reasoning",
        license="cc-by-4.0",
        source_url="https://huggingface.co/datasets/ibm-research/finqa",
        verification_status=VERIFIED,
        fields_used=["question", "answer", "gold_evidence"],
        notes="Multi-step numerical reasoning questions over financial reports (FinQA).",
    ),
    # ------------------------------------------------------------------
    # Financial instruction / chat
    # ------------------------------------------------------------------
    "financial_instruction_alpaca": DatasetEntry(
        name="financial_instruction_alpaca",
        hf_id="gbharti/finance-alpaca",
        split="train",
        category="financial_instruction",
        license="mit",
        source_url="https://huggingface.co/datasets/gbharti/finance-alpaca",
        verification_status=VERIFIED,
        fields_used=["instruction", "input", "output"],
        notes="Alpaca-style instruction/response pairs for finance.",
    ),
    # ------------------------------------------------------------------
    # Financial news / sentiment (kept deliberately low-weight, see Part 8)
    # ------------------------------------------------------------------
    "financial_sentiment_phrasebank": DatasetEntry(
        name="financial_sentiment_phrasebank",
        hf_id="takala/financial_phrasebank",
        subset="sentences_allagree",
        split="train",
        category="financial_news_sentiment",
        license="cc-by-nc-sa-3.0",
        source_url="https://huggingface.co/datasets/takala/financial_phrasebank",
        verification_status=VERIFIED,
        fields_used=["sentence", "label"],
        notes="Non-commercial license (CC-BY-NC-SA-3.0). Sentiment language only - "
        "must never dominate the training mixture.",
    ),
}


def get_entry(name: str) -> DatasetEntry:
    if name not in REGISTRY:
        raise KeyError(
            f"Unknown dataset '{name}'. Known datasets: {sorted(REGISTRY)}"
        )
    return REGISTRY[name]


def list_entries(category: Optional[str] = None, verified_only: bool = False):
    entries = list(REGISTRY.values())
    if category is not None:
        entries = [e for e in entries if e.category == category]
    if verified_only:
        entries = [e for e in entries if e.verification_status == VERIFIED]
    return entries
