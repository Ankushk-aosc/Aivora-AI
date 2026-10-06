"""Build Aivora_Model_Technical_Documentation.docx.

The companion to Aivora_Technical_Documentation.docx: that one documents the
application, this one documents the language model trained in this repository -
its architecture, how it was trained, what it can and cannot do, and what was
measured rather than assumed.

Written to the same structure as the owner's reference document.

    python scripts/build_model_diagrams.py
    python scripts/build_model_documentation.py
"""

import os

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

OUT = "Aivora_Model_Technical_Documentation.docx"
DIAGRAMS = os.path.join("docs", "diagrams")

INK = RGBColor(0x11, 0x18, 0x27)
MUTED = RGBColor(0x6B, 0x72, 0x80)
ACCENT = RGBColor(0x1D, 0x4E, 0xD8)
WARN = RGBColor(0xB9, 0x1C, 0x1C)


def shade(cell, hex_colour):
    element = OxmlElement("w:shd")
    element.set(qn("w:val"), "clear")
    element.set(qn("w:fill"), hex_colour)
    cell._tc.get_or_add_tcPr().append(element)


def table(document, headers, rows, widths=None):
    t = document.add_table(rows=1, cols=len(headers))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, header in enumerate(headers):
        cell = t.rows[0].cells[i]
        cell.text = ""
        run = cell.paragraphs[0].add_run(header)
        run.bold = True
        run.font.size = Pt(9)
        shade(cell, "E8EDF5")
    for row in rows:
        cells = t.add_row().cells
        for i, value in enumerate(row):
            cells[i].text = ""
            run = cells[i].paragraphs[0].add_run(str(value))
            run.font.size = Pt(9)
            if i == 0 and len(headers) > 2:
                run.bold = True
    if widths:
        for row in t.rows:
            for i, width in enumerate(widths):
                row.cells[i].width = Inches(width)
    document.add_paragraph()
    return t


def bullets(document, items):
    for item in items:
        paragraph = document.add_paragraph(item, style="List Bullet")
        paragraph.paragraph_format.space_after = Pt(3)
        for run in paragraph.runs:
            run.font.size = Pt(10)


def numbered(document, items):
    for item in items:
        paragraph = document.add_paragraph(item, style="List Number")
        paragraph.paragraph_format.space_after = Pt(3)
        for run in paragraph.runs:
            run.font.size = Pt(10)


def body(document, text, italic=False, colour=None, size=10):
    paragraph = document.add_paragraph()
    run = paragraph.add_run(text)
    run.font.size = Pt(size)
    run.italic = italic
    if colour is not None:
        run.font.color.rgb = colour
    paragraph.paragraph_format.space_after = Pt(8)


def code(document, text):
    paragraph = document.add_paragraph()
    run = paragraph.add_run(text)
    run.font.name = "Consolas"
    run.font.size = Pt(8.5)
    paragraph.paragraph_format.left_indent = Inches(0.25)
    paragraph.paragraph_format.space_after = Pt(8)


def callout(document, text):
    t = document.add_table(rows=1, cols=1)
    t.style = "Table Grid"
    cell = t.rows[0].cells[0]
    cell.text = ""
    run = cell.paragraphs[0].add_run(text)
    run.font.size = Pt(9.5)
    run.bold = True
    run.font.color.rgb = WARN
    shade(cell, "FDF2F2")
    document.add_paragraph()


def figure(document, filename, caption, width=6.5):
    path = os.path.join(DIAGRAMS, filename)
    if not os.path.exists(path):
        body(document, f"[figure missing: {filename}]", italic=True, colour=WARN)
        return
    document.add_picture(path, width=Inches(width))
    document.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph = document.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = paragraph.add_run(caption)
    run.font.size = Pt(8.5)
    run.italic = True
    run.font.color.rgb = MUTED
    paragraph.paragraph_format.space_after = Pt(12)


def build():
    document = Document()
    document.styles["Normal"].font.name = "Calibri"
    document.styles["Normal"].font.size = Pt(10)
    for section in document.sections:
        section.top_margin = Inches(0.8)
        section.bottom_margin = Inches(0.8)
        section.left_margin = Inches(0.85)
        section.right_margin = Inches(0.85)

    # ------------------------------------------------------------- title
    title = document.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = title.add_run("AIVORA")
    run.bold = True
    run.font.size = Pt(34)
    run.font.color.rgb = ACCENT
    for text, size, colour, bold in (
            ("Language Model", 15, INK, False),
            ("A 101.7M-parameter DeepSeek-V3-architecture model trained from "
             "scratch for financial text", 10.5, MUTED, False),
            ("Model Technical Documentation", 13, INK, True)):
        paragraph = document.add_paragraph()
        paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = paragraph.add_run(text)
        r.font.size = Pt(size)
        r.font.color.rgb = colour
        r.bold = bold
    paragraph = document.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = paragraph.add_run("Version 1.0  ·  5 October 2026  ·  Companion to "
                            "Aivora_Technical_Documentation.docx")
    run.font.size = Pt(9)
    run.font.color.rgb = MUTED
    document.add_paragraph()

    callout(document,
            "This document reports what the model does, including where it "
            "fails. Every number is measured on a frozen held-out set the model "
            "was never trained on and never selected against. Capability is "
            "never reported from training loss.")

    # ---------------------------------------------------- 1 overview
    document.add_heading("1. Overview", level=1)
    body(document,
         "The Aivora model is a decoder-only transformer of 101.7M parameters, "
         "written from scratch in this repository and following the DeepSeek-V3 "
         "architecture: a Mixture-of-Experts feed-forward network, Multi-head "
         "Latent Attention, and a Multi-Token Prediction auxiliary head. It was "
         "pre-trained on financial text and then supervised fine-tuned for "
         "extraction and question answering.")
    body(document,
         "It is a research model. Within the Aivora application it is used for "
         "generation only; it is never trusted with arithmetic, and no figure "
         "it produces is shown to a user without validation against the source "
         "document. Section 7 reports the measurements that led to that "
         "decision.")

    document.add_heading("1.1 Specification", level=2)
    table(document, ["Property", "Value"], [
        ("Architecture", "DeepSeek-V3 style decoder-only transformer"),
        ("Total parameters", "101.72M"),
        ("Layers", "8"),
        ("Hidden size", "512"),
        ("Attention heads", "8"),
        ("Context length", "1,024 tokens"),
        ("Vocabulary", "50,257 (GPT-2 BPE via tiktoken)"),
        ("Experts", "8 routed, top-2 per token, plus 1 always-active shared expert"),
        ("Expert intermediate size", "512 (routed), 768 (shared)"),
        ("KV LoRA rank", "128"),
        ("Query LoRA rank", "192"),
        ("RoPE dimensions", "32"),
        ("MTP heads", "1, auxiliary loss weight 0.3"),
        ("Dropout", "0.1"),
        ("Precision", "float32, CPU and CUDA"),
    ], widths=[2.2, 4.4])

    document.add_heading("1.2 Parameter Distribution", level=2)
    table(document, ["Component", "Parameters", "Share"], [
        ("Transformer blocks (8)", "66.61M", "65.5%"),
        ("Token embeddings", "25.73M", "25.3%"),
        ("Multi-Token Prediction head", "8.85M", "8.7%"),
        ("Position embeddings", "0.52M", "0.5%"),
        ("Final layer norm", "< 0.01M", "~0%"),
        ("Total", "101.72M", "100%"),
    ], widths=[2.6, 1.8, 1.4])
    body(document,
         "A quarter of the model is its vocabulary embedding. At this scale the "
         "50,257-token GPT-2 vocabulary is a substantial fixed cost, and the "
         "MTP head is a further 8.7% that serves training rather than "
         "inference. (measured)")

    # -------------------------------------------------- 2 architecture
    document.add_heading("2. Architecture", level=1)
    figure(document, "05_model_architecture.png",
           "Figure 1. Model architecture. Attention and feed-forward paths "
           "within one of eight transformer blocks.")

    document.add_heading("2.1 Multi-head Latent Attention", level=2)
    body(document,
         "Standard attention caches a key and value vector per head per token. "
         "MLA instead projects keys and values into a shared low-rank latent "
         "space - rank 128 here - and caches that, reconstructing per-head "
         "keys and values on use. Queries carry their own low-rank projection "
         "at rank 192, and 32 dimensions carry rotary position encoding "
         "separately from the compressed path. The purpose is a smaller KV "
         "cache at long context for a small accuracy cost.")

    document.add_heading("2.2 Mixture of Experts", level=2)
    body(document,
         "Each block's feed-forward network is eight expert networks plus one "
         "shared expert. A router scores the experts per token and the top two "
         "run; the shared expert always runs, giving every token a common "
         "pathway so the routed experts can specialise. Total parameters "
         "therefore exceed activated parameters: most of the feed-forward "
         "capacity is idle for any given token.")
    callout(document,
            "MEASURED DEFECT. The auxiliary load-balancing loss weight is 0.0, "
            "so nothing penalised an unbalanced router - and the router "
            "collapsed. 28 of the 64 routed expert slots receive under 1% of "
            "traffic. In blocks 5, 6 and 7 the top-2 router sends 100% of "
            "traffic to the same two experts regardless of input.")
    body(document,
         "This was verified across five diverse inputs - a financial filing, "
         "technology news, narrative prose, source code and general questions - "
         "so it is collapse rather than domain specialisation: the routing does "
         "not change when the input does. The effect worsens with depth, which "
         "is the signature of a router that lost its gradient signal early and "
         "never recovered. (measured, 6 October 2026)")
    table(document, ["Block", "Experts receiving >= 1% of traffic",
                     "Share taken by the top 2"], [
        ("h.0", "7 of 8", "71.7%"),
        ("h.1", "7 of 8", "52.3%"),
        ("h.2", "5 of 8", "67.9%"),
        ("h.3", "5 of 8", "73.8%"),
        ("h.4", "6 of 8", "66.6%"),
        ("h.5", "2 of 8", "100.0%"),
        ("h.6", "2 of 8", "99.2%"),
        ("h.7", "2 of 8", "100.0%"),
    ], widths=[1.1, 2.9, 2.6])
    body(document,
         "Cost: at 0.788M parameters per routed expert, 28 dead slots are "
         "22.06M parameters that never activate - 21.7% of the model. Its "
         "effective size is approximately 79.7M, not 101.7M. This is "
         "independent of, and additional to, the under-training described in "
         "section 3. (measured)")

    document.add_heading("2.3 Multi-Token Prediction", level=2)
    body(document,
         "One additional head predicts a second future token, contributing to "
         "the loss at weight 0.3. The intent is a denser training signal per "
         "position. The head is 8.85M parameters and is used in training only.")

    document.add_heading("2.4 Objective and Prompt Masking", level=2)
    body(document,
         "Loss is cross-entropy over answer tokens only. Prompt tokens are set "
         "to the ignore index so the model is not rewarded for reproducing the "
         "question. The ignore index is -1, matching the value passed to the "
         "loss function; a mismatch here silently trains on prompt tokens and "
         "is not visible in the loss curve.")
    code(document,
         "IGNORE_INDEX = -1\n"
         "PROMPT_TEMPLATE = \"Question: {prompt}\\nAnswer:\"\n"
         "loss = F.cross_entropy(logits, targets, ignore_index=-1)")

    # ------------------------------------------------------ 3 data
    document.add_heading("3. Training Data", level=1)
    body(document,
         "Pre-training used financial text - filings, financial QA sets, "
         "instruction data and general web text - tokenised with the GPT-2 BPE "
         "vocabulary and packed into 1,024-token sequences with document "
         "separators so unrelated documents do not bleed across a boundary.")
    table(document, ["Property", "Value"], [
        ("Unique training tokens", "120,857,129"),
        ("Epochs", "16.3"),
        ("Total tokens processed", "~1.97B (unique tokens x epochs)"),
        ("Token / parameter ratio", "1.19"),
        ("Compute-optimal ratio (Chinchilla)", "~20"),
        ("Fraction of compute-optimal data", "~6%"),
    ], widths=[3.0, 3.6])
    callout(document,
            "This is the single most important fact about the model. It saw "
            "about 6% of the data a 101.7M-parameter model needs, repeated 16 "
            "times. Repetition is not a substitute for breadth, and no amount "
            "of fine-tuning recovers pre-training the model never had.")

    # ------------------------------------------------- 4 training
    document.add_heading("4. Training Pipeline", level=1)
    figure(document, "06_training_pipeline.png",
           "Figure 2. Training pipeline, from corpus to an accepted checkpoint.",
           width=6.9)

    document.add_heading("4.1 Checkpoint Selection by Capability Gate", level=2)
    body(document,
         "Checkpoints are accepted on measured capability, never on training "
         "loss. Five gates run against probes the model has not been trained "
         "on:")
    table(document, ["Gate", "Tests"], [
        ("A - Copy", "Reproducing a literal value present in the prompt"),
        ("B - Extraction", "Returning the value for a named field"),
        ("C - Calculation", "Producing a correct arithmetic result"),
        ("D - Abstention", "Refusing when the information is absent"),
        ("E - Unseen wording", "Extraction under labels not seen in training"),
    ], widths=[1.8, 4.8])

    document.add_heading("4.2 SFT_001 - A Documented Failure", level=2)
    body(document,
         "The first supervised fine-tuning run improved training loss "
         "monotonically and destroyed the model's ability to copy: gate A fell "
         "from 5/6 to 0/6. Loss would have accepted this checkpoint. The causes "
         "were measured, not guessed:")
    bullets(document, [
        "Extraction examples were 30% of rows but only 10% of answer tokens, so "
        "the objective barely saw them.",
        "956 rows shared just 13 distinct answer strings, teaching the model to "
        "emit a small set of fixed replies.",
        "Early stopping was driven by loss rather than by capability.",
    ])
    body(document,
         "SFT_002 was built to correct exactly these defects - answer-token "
         "balancing, a distinct-answer entropy floor, and gate-based stopping - "
         "and passed all five gates. The dataset builder now refuses to write a "
         "dataset that repeats the SFT_001 defect.")
    callout(document,
            "A loss curve is not a capability measurement. This was the most "
            "expensive lesson in the project and the reason gates exist.")

    # ---------------------------------------------- 5 evaluation method
    document.add_heading("5. Evaluation Methodology", level=1)
    body(document,
         "The model is measured against a frozen held-out set that it was never "
         "trained on and never selected against. The rules were pre-registered "
         "before any system was scored.")
    table(document, ["Property", "Value"], [
        ("Frozen set size", "313 items"),
        ("Hand-written items", "125"),
        ("Generated items", "188"),
        ("Copy / Extraction / Wording", "69 / 69 / 75"),
        ("Calculation / Abstention", "48 / 52"),
        ("Selection split", "61 items, used only to choose a configuration"),
        ("Contamination control", "Exact match plus 8-gram overlap removal "
                                  "against 5 training files"),
        ("Seed", "424242"),
        ("Integrity check", "Content hash over parsed items in canonical order, "
                            "verified on every scored run"),
    ], widths=[2.4, 4.2])
    bullets(document, [
        "One shared scorer for every system, so no system is judged by its own "
        "standard.",
        "Identical prompts and decoding across systems; the only variable is "
        "the system.",
        "The frozen set is scored once per configuration, and raw outputs are "
        "preserved so a scorer correction can be applied without re-running a "
        "model.",
        "Wilson intervals for single scores; the exact McNemar test for two "
        "systems on identical items.",
        "Over-refusal is reported separately from abstention accuracy, because "
        "abstention alone would reward a system that refuses everything.",
    ])

    # ------------------------------------------------- 6 results
    document.add_heading("6. Measured Results", level=1)
    figure(document, "07_evaluation_results.png",
           "Figure 3. Accuracy by capability gate, and the rate of invented "
           "values.", width=6.9)

    table(document, ["Gate", "Base model", "SFT_002", "Tool pipeline + Qwen2.5-1.5B"], [
        ("Copy", "13.0%", "34.8%", "98.6%"),
        ("Extraction", "1.4%", "13.0%", "87.0%"),
        ("Wording", "2.7%", "13.3%", "85.3%"),
        ("Calculation", "0.0%", "2.1%", "100%"),
        ("Abstention", "0.0%", "16.7%", "97.2%"),
        ("Invented values", "42.9%", "36.8%", "0.0%"),
    ], widths=[1.4, 1.3, 1.3, 2.6])

    document.add_heading("6.1 Paired Comparisons", level=2)
    body(document, "Exact McNemar tests on identical items. (measured)")
    table(document, ["Comparison", "Better", "Worse", "p"], [
        ("SFT_002 vs base model", "10", "0", "0.0020"),
        ("SmolLM2-135M vs SFT_002", "14", "4", "0.0309"),
        ("Tool pipeline vs SFT_002", "44", "0", "< 0.0001"),
    ], widths=[2.8, 1.1, 1.1, 1.6])

    document.add_heading("6.2 The Parameter-Matched Result", level=2)
    body(document,
         "SmolLM2-135M-Instruct has 135M parameters against this model's "
         "101.7M. On identical items the pre-trained model wins (p = 0.0309). "
         "With size approximately held constant, the gap does not come from "
         "capacity - it comes from pre-training data. The model is "
         "under-trained, not undersized, and that distinction determines what "
         "would actually fix it. (measured)")

    # --------------------------------------------- 7 capabilities
    document.add_heading("7. Capabilities and Limitations", level=1)
    document.add_heading("7.1 What the model does adequately", level=2)
    bullets(document, [
        "Produces fluent, well-formed financial prose.",
        "Follows the chat template it was fine-tuned on.",
        "Copies a short literal value from a prompt in about a third of cases "
        "after SFT_002.",
    ])

    document.add_heading("7.2 What it does not do", level=2)
    bullets(document, [
        "Reliable extraction. 13.0% on the frozen set after fine-tuning.",
        "Arithmetic of any kind. 2.1% on calculation. It is never asked to "
        "compute in production.",
        "Structured output. Asked for a JSON object, it returns prose: measured "
        "on 5 October 2026 it answered \"Revenue = $5,200,000 - the EBIT "
        "entry.\" for a context stating $10,000,000 - an invented figure in an "
        "unparseable format.",
        "Abstention. Left to itself it invents a value on 36.8% of answerable "
        "items rather than declining.",
    ])

    callout(document,
            "Deployment rule. This model must not be used to state a financial "
            "figure without external validation. In the Aivora application the "
            "span-validation layer makes that structurally impossible: the "
            "invented $5,200,000 above was blocked and became an abstention.")

    document.add_heading("7.3 Defects found and corrected during development",
                         level=2)
    table(document, ["Defect", "Effect", "Resolution"], [
        ("Repetition detector counted prompt tokens",
         "Killed 76.7% of greedy answers after 2-3 tokens",
         "Window restricted to generated tokens"),
        ("Thousands separators split on the comma",
         "\"4,532.00\" parsed as 4 and 532.0; large figures understated",
         "Whole-token numeric parsing"),
        ("Abstention markers missed real refusal wording",
         "Honest refusals scored as hallucinations",
         "Markers extended; SFT data aligned to production wording"),
        ("Loss-based early stopping",
         "Accepted SFT_001, which had lost the ability to copy",
         "Capability gates A-E"),
        ("MoE router collapse (open)",
         "28 of 64 expert slots dead; 21.7% of parameters never activate",
         "Not yet fixed - requires aux_loss_weight > 0 and retraining"),
        ("Byte-level eval-set hash",
         "Platform-dependent; raised a false alarm and could not detect real "
         "change",
         "Content hash over parsed items"),
    ], widths=[1.9, 2.5, 2.2])

    # ---------------------------------------------- 8 checkpoints
    document.add_heading("8. Checkpoints", level=1)
    table(document, ["Checkpoint", "Stage", "Status"], [
        ("checkpoints/final/checkpoint_247850.pt", "Pre-trained base",
         "Frozen baseline, tagged AIVORA_PRE_TRAINING_BASELINE"),
        ("checkpoints/sft_001/", "First SFT attempt",
         "Rejected - gate A fell from 5/6 to 0/6"),
        ("checkpoints/sft_002/", "Corrected SFT",
         "Accepted - passed all five gates"),
        ("checkpoints/sft_003/", "Third SFT run",
         "Loaded by the application by default"),
    ], widths=[2.4, 1.6, 2.6])
    body(document,
         "All checkpoints and their evaluation outputs are preserved, including "
         "the failed run. A failed experiment that is deleted has to be "
         "repeated.")

    # ----------------------------------------------- 9 usage
    document.add_heading("9. Loading and Running the Model", level=1)
    code(document,
         "from models import DeepSeekConfig, DeepSeekV3\n"
         "import torch\n\n"
         "model = DeepSeekV3(DeepSeekConfig.default())\n"
         "state = torch.load('checkpoints/sft_002/sft_002_best.pt',\n"
         "                   map_location='cpu')\n"
         "model.load_state_dict(state.get('model_state_dict', state))\n"
         "model.eval()")
    body(document, "Greedy decoding, as used for every evaluation in this "
                   "document:")
    code(document,
         "out = model.generate(ids, max_new_tokens=160, temperature=1e-5,\n"
         "                     top_k=1, stop_on_repetition=True,\n"
         "                     eos_token_id=enc.eot_token)")
    body(document,
         "Hardware: runs on CPU in roughly 4 GB of RAM. No GPU required for "
         "inference.")

    # -------------------------------------------- 10 recommendations
    document.add_heading("10. Recommendations", level=1)
    numbered(document, [
        "Do not continue fine-tuning this checkpoint for extraction accuracy. "
        "The parameter-matched comparison in section 6.2 shows the limit is "
        "pre-training data, and fine-tuning does not add pre-training.",
        "Keep the model for generation and the tool pipeline for anything "
        "numeric. This is what the application already does.",
        "If a from-scratch model remains a goal, the binding constraint is the "
        "token budget: approximately 2B unique tokens would be needed to reach "
        "a compute-optimal ratio at this parameter count, against 120.9M today.",
        "Set the auxiliary load-balancing loss weight above zero before any "
        "retraining. At 0.0 the router collapsed and 21.7% of the model is "
        "dead weight; this is the cheapest single correction available and it "
        "costs one configuration value.",
        "Retain gate-based checkpoint selection. It caught a regression that "
        "loss endorsed.",
        "Commission a fresh held-out set before any further capability claim. "
        "Four defects have been found in the current one, and three of them "
        "were authored by the same process that measures against it.",
    ])

    # ---------------------------------------------- 11 glossary
    document.add_heading("11. Glossary", level=1)
    table(document, ["Term", "Definition"], [
        ("MoE", "Mixture of Experts. Several feed-forward networks per layer "
                "with a router selecting a subset per token."),
        ("MLA", "Multi-head Latent Attention. Keys and values compressed into a "
                "shared low-rank space to shrink the cache."),
        ("MTP", "Multi-Token Prediction. An auxiliary head predicting a further "
                "future token during training."),
        ("RoPE", "Rotary Position Embedding, applied to a subset of dimensions."),
        ("BPE", "Byte Pair Encoding. The GPT-2 subword vocabulary, 50,257 "
                "tokens."),
        ("Prompt masking", "Setting prompt token targets to the ignore index so "
                           "loss is computed only on answer tokens."),
        ("Token / parameter ratio", "Unique training tokens divided by "
                                    "parameters. Approximately 20 is "
                                    "compute-optimal; this model is at 1.19."),
        ("Capability gate", "A pass/fail behavioural test used to accept or "
                            "reject a checkpoint, replacing loss-based "
                            "selection."),
        ("SFT", "Supervised Fine-Tuning on prompt and answer pairs."),
        ("Frozen set", "Held-out evaluation items, never trained on and never "
                       "used for selection."),
        ("Invented value", "A number in an answer that does not appear in the "
                           "source. The headline safety metric."),
        ("Over-refusal", "Declining a question the source does answer."),
        ("McNemar test", "An exact paired test for two systems measured on "
                         "identical items."),
        ("Wilson interval", "A confidence interval for a proportion, reliable "
                            "at small n and near 0% or 100%."),
    ], widths=[1.9, 4.7])

    document.save(OUT)
    print(f"wrote {OUT} ({os.path.getsize(OUT) / 1024:.0f} KB)")


if __name__ == "__main__":
    build()
