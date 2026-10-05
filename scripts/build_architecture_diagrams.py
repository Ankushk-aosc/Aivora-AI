"""Architecture diagrams for the Aivora technical documentation.

Drawn with matplotlib so the figures are generated from this file rather than
pasted from a drawing tool: when the architecture changes, the diagram is
regenerated instead of redrawn by hand.

    python scripts/build_architecture_diagrams.py

Writes PNGs to docs/diagrams/.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                   # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch    # noqa: E402

OUT_DIR = os.path.join("docs", "diagrams")

INK = "#111827"
MUTED = "#6b7280"
LINE = "#9ca3af"
PANEL = "#f3f4f6"
PANEL_EDGE = "#d1d5db"

# One accent per responsibility, used consistently across every figure.
CLIENT = "#dbeafe", "#1d4ed8"
API = "#e0e7ff", "#4338ca"
LOGIC = "#dcfce7", "#15803d"
MODEL = "#fef3c7", "#b45309"
DATA = "#f3e8ff", "#7e22ce"
GUARD = "#fee2e2", "#b91c1c"


def box(ax, x, y, w, h, title, subtitle=None, palette=LOGIC, fontsize=9.5):
    face, edge = palette
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02",
        facecolor=face, edgecolor=edge, linewidth=1.3, zorder=3))
    # Text is placed as a FRACTION of the box height. A fixed offset put the
    # subtitle on the bottom border of any box shorter than about 0.08, which
    # read as a strikethrough.
    if subtitle:
        ax.text(x + w / 2, y + h * 0.63, title, ha="center", va="center",
                fontsize=fontsize, fontweight="600", color=INK, zorder=4)
        ax.text(x + w / 2, y + h * 0.27, subtitle, ha="center", va="center",
                fontsize=min(7.6, fontsize - 0.9), color=MUTED, zorder=4,
                linespacing=1.45)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center",
                fontsize=fontsize, fontweight="600", color=INK, zorder=4)


def panel(ax, x, y, w, h, label):
    """A grouping band. The caption sits ABOVE the band, never inside it, so a
    box placed in the band can never be drawn over its own label."""
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.006,rounding_size=0.012",
        facecolor=PANEL, edgecolor=PANEL_EDGE, linewidth=1, zorder=1))
    ax.text(x + 0.004, y + h + 0.010, label, ha="left", va="bottom", fontsize=7.8,
            color=MUTED, fontweight="600", zorder=2)


def arrow(ax, start, end, label=None, style="-|>", colour=LINE, dashed=False,
          rad=0.0, label_offset=(0, 0.018), fontsize=7.6):
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle=style, mutation_scale=11, color=colour,
        linewidth=1.2, zorder=5, shrinkA=2, shrinkB=2,
        linestyle="--" if dashed else "-",
        connectionstyle=f"arc3,rad={rad}"))
    if label:
        mx = (start[0] + end[0]) / 2 + label_offset[0]
        my = (start[1] + end[1]) / 2 + label_offset[1]
        ax.text(mx, my, label, ha="center", va="center", fontsize=fontsize,
                color=MUTED, zorder=6,
                bbox=dict(boxstyle="round,pad=0.18", facecolor="white",
                          edgecolor="none", alpha=0.9))


def canvas(width, height, title, subtitle):
    figure, ax = plt.subplots(figsize=(width, height))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.5, 0.975, title, ha="center", va="top", fontsize=13,
            fontweight="700", color=INK)
    ax.text(0.5, 0.935, subtitle, ha="center", va="top", fontsize=8.6,
            color=MUTED)
    return figure, ax


def save(figure, name):
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, name)
    figure.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    print(f"wrote {path}")
    return path


# --------------------------------------------------------------- figure 1
def system_architecture():
    figure, ax = canvas(11.5, 9.0, "Aivora - System Architecture",
                        "Layered request path from the browser to the verified "
                        "financial record")

    # Rows are laid out from the top with a fixed gap, so adding a component
    # never silently overlaps the band below it.
    panel(ax, 0.04, 0.800, 0.92, 0.085, "CLIENT")
    for i, (name, sub) in enumerate([
            ("Overview", "dashboard"), ("Analysis", "metrics"),
            ("Documents", "upload & Q&A"), ("Financial Analyst", "conversation"),
            ("Comparisons", "period vs period"), ("Insights", "+ monitoring"),
            ("Reports", "generate & export")]):
        box(ax, 0.052 + i * 0.1283, 0.812, 0.118, 0.061, name, sub, CLIENT, 8.0)

    panel(ax, 0.04, 0.655, 0.92, 0.085, "HTTP API  -  app/backend/server.py")
    box(ax, 0.052, 0.667, 0.205, 0.061, "ThreadingHTTPServer",
        "stdlib only, no framework", API, 8.4)
    box(ax, 0.272, 0.667, 0.185, 0.061, "Route table", "~60 endpoints", API, 8.4)
    box(ax, 0.472, 0.667, 0.215, 0.061, "security.py",
        "CORS, body limit, path confine", GUARD, 8.4)
    box(ax, 0.702, 0.667, 0.246, 0.061, "workspace.py",
        "company -> document scope", GUARD, 8.4)

    panel(ax, 0.04, 0.330, 0.92, 0.245, "SERVICE LAYER")
    row = [("financial_router", "intent classification", LOGIC),
           ("chat_service", "Analyst conversation", LOGIC),
           ("knowledge_retrieval", "glossary & definitions", LOGIC),
           ("analyst_pipeline", "tool-pipeline bridge (gated)", LOGIC)]
    for i, (name, sub, palette) in enumerate(row):
        box(ax, 0.052 + i * 0.2245, 0.498, 0.213, 0.062, name, sub, palette, 8.3)
    row = [("financial_values", "label-to-value binding", LOGIC),
           ("product_views", "comparison, insights, report", LOGIC),
           ("alerts", "13 metric conditions", LOGIC),
           ("saved_analyses", "named snapshots", LOGIC)]
    for i, (name, sub, palette) in enumerate(row):
        box(ax, 0.052 + i * 0.2245, 0.423, 0.213, 0.062, name, sub, palette, 8.3)
    row = [("grounding / abstention", "refuse rather than guess", GUARD),
           ("quality / question_focus", "does the answer address the ask", GUARD),
           ("financial_calculator", "deterministic arithmetic", MODEL)]
    for i, (name, sub, palette) in enumerate(row):
        box(ax, 0.052 + i * 0.3005, 0.345, 0.288, 0.062, name, sub, palette, 8.3)

    panel(ax, 0.04, 0.150, 0.445, 0.155, "REASONING")
    box(ax, 0.052, 0.232, 0.421, 0.060, "pipeline/tool_pipeline.py",
        "span validation + Python arithmetic", MODEL, 8.4)
    box(ax, 0.052, 0.162, 0.202, 0.060, "DeepSeekV3", "101.7M, local", MODEL, 8.2)
    box(ax, 0.271, 0.162, 0.202, 0.060, "HF backend",
        "Qwen2.5 etc. (optional)", MODEL, 8.2)

    panel(ax, 0.515, 0.150, 0.445, 0.155, "RETRIEVAL & STORAGE")
    box(ax, 0.527, 0.232, 0.203, 0.060, "DocumentStore",
        "TF-IDF chunk index", DATA, 8.2)
    box(ax, 0.745, 0.232, 0.203, 0.060, "persistent_store",
        "SQLite chunk text", DATA, 8.2)
    box(ax, 0.527, 0.162, 0.203, 0.060, "data/documents/",
        "source filings", DATA, 8.2)
    box(ax, 0.745, 0.162, 0.203, 0.060, "monitoring.db",
        "rules & saved analyses", DATA, 8.2)

    panel(ax, 0.04, 0.030, 0.92, 0.078, "EVIDENCE RETURNED TO THE CLIENT")
    for i, (name, sub) in enumerate([
            ("Value", "as stated"), ("Source span", "validated in document"),
            ("Formula", "how it was derived"), ("Working", "the operands used"),
            ("Document", "which filing"), ("Abstention", "when unverifiable")]):
        box(ax, 0.052 + i * 0.1497, 0.041, 0.138, 0.056, name, sub, GUARD, 8.0)

    arrow(ax, (0.5, 0.800), (0.5, 0.742), "HTTP / JSON")
    arrow(ax, (0.5, 0.655), (0.5, 0.578), "routed by intent")
    arrow(ax, (0.26, 0.330), (0.26, 0.307), "extract / compute",
          label_offset=(0, 0.014))
    arrow(ax, (0.74, 0.330), (0.74, 0.307), "retrieve", label_offset=(0, 0.014))
    arrow(ax, (0.5, 0.150), (0.5, 0.110), "verified result")

    ax.text(0.5, 0.008,
            "No value reaches the client without a source span validated "
            "against the document it came from.",
            ha="center", va="bottom", fontsize=8.6, color=GUARD[1],
            fontweight="600")
    return save(figure, "01_system_architecture.png")


# --------------------------------------------------------------- figure 2
def request_routing():
    figure, ax = canvas(11, 6.4, "Aivora - Question Routing",
                        "Each question is answered by the component that can "
                        "answer it correctly")

    box(ax, 0.40, 0.845, 0.20, 0.060, "User question", "natural language",
        CLIENT, 9.5)
    box(ax, 0.40, 0.735, 0.20, 0.060, "Intent classification",
        "financial_router.py", API, 9.5)
    arrow(ax, (0.50, 0.845), (0.50, 0.795))

    lanes = [
        (0.035, "Asks for a figure", "Extraction", "model copies the value\nwith its source span", MODEL),
        (0.235, "Asks for a ratio", "Calculation", "operands extracted,\nPython computes", MODEL),
        (0.435, "Asks what a term means", "Knowledge", "glossary lookup,\nno document needed", LOGIC),
        (0.635, "Asks about a document", "Retrieval", "workspace-scoped\nchunk search", DATA),
        (0.835, "Asks for the future", "Refusal rule", "no forecast exists\nin the filing", GUARD),
    ]
    for x, question, title, detail, palette in lanes:
        ax.text(x + 0.065, 0.695, question, ha="center", va="center",
                fontsize=7.8, color=MUTED, style="italic")
        # The detail is the box's subtitle, not a separate label drawn over it.
        box(ax, x, 0.555, 0.13, 0.115, title, detail, palette, 9.0)
        arrow(ax, (0.50, 0.735), (x + 0.065, 0.672), rad=0.0)

    box(ax, 0.22, 0.410, 0.56, 0.072, "Validation gate",
        "is the value present in the document, inside the span it cites?",
        GUARD, 9.5)
    for x, _, _, _, _ in lanes[:4]:
        arrow(ax, (x + 0.065, 0.555), (0.50, 0.484), rad=0.05)
    arrow(ax, (0.90, 0.555), (0.90, 0.300), "refuses directly", dashed=True,
          label_offset=(0.085, 0))

    box(ax, 0.12, 0.225, 0.33, 0.072, "Answer with evidence",
        "value + span + formula + document", LOGIC, 9.5)
    box(ax, 0.55, 0.225, 0.33, 0.072, "Abstention",
        "states what is missing and why", GUARD, 9.5)
    arrow(ax, (0.40, 0.410), (0.28, 0.297), "validated")
    arrow(ax, (0.60, 0.410), (0.715, 0.297), "fails validation")

    ax.text(0.5, 0.105,
            "An abstention is a correct answer. It is never replaced by a guess "
            "from another path.",
            ha="center", va="center", fontsize=8.6, color=GUARD[1],
            fontweight="600")
    ax.text(0.5, 0.045,
            "Measured on the frozen held-out set (n=313): 0 invented values in "
            "261 answerable items; 261 of 261 stated values span-validated.",
            ha="center", va="center", fontsize=7.8, color=MUTED)
    return save(figure, "02_question_routing.png")


# --------------------------------------------------------------- figure 3
def tool_pipeline():
    figure, ax = canvas(11, 6.8, "Aivora - Tool Pipeline",
                        "The model reads; Python decides and calculates")

    panel(ax, 0.04, 0.70, 0.92, 0.185, "1.  ROUTE  -  rules, not the model")
    box(ax, 0.06, 0.785, 0.21, 0.058, "Question", "\"What is the gross margin?\"",
        CLIENT, 8.8)
    box(ax, 0.30, 0.785, 0.21, 0.058, "Formula lookup", "21 formulas", LOGIC, 8.8)
    box(ax, 0.54, 0.785, 0.19, 0.058, "Operands", "revenue, cost of sales",
        LOGIC, 8.8)
    box(ax, 0.76, 0.785, 0.19, 0.058, "Caption resolution",
        "24 fields, 147 captions", LOGIC, 8.8)
    arrow(ax, (0.27, 0.814), (0.30, 0.814))
    arrow(ax, (0.51, 0.814), (0.54, 0.814))
    arrow(ax, (0.73, 0.814), (0.76, 0.814))
    ax.text(0.50, 0.722,
            "Python finds the caption the document actually uses -\n"
            "\"Turnover\" for revenue, \"Borrowings\" for total debt - so the "
            "model is only ever asked for a label that is in front of it.",
            ha="center", va="center", fontsize=7.6, color=MUTED, linespacing=1.5)

    panel(ax, 0.04, 0.495, 0.92, 0.175, "2.  READ  -  the model's only job")
    box(ax, 0.06, 0.560, 0.26, 0.070, "Extraction prompt",
        "per operand, JSON only", API, 8.8)
    box(ax, 0.36, 0.560, 0.26, 0.070, "Model",
        "returns value + source_span", MODEL, 8.8)
    box(ax, 0.66, 0.560, 0.29, 0.070, "Structured output",
        '{"value": "...", "source_span": "..."}', MODEL, 8.8)
    arrow(ax, (0.32, 0.595), (0.36, 0.595))
    arrow(ax, (0.62, 0.595), (0.66, 0.595))
    ax.text(0.50, 0.516, "The model never performs arithmetic. Not once, not "
            "for \"simple\" cases.", ha="center", va="center", fontsize=7.8,
            color=MUTED, style="italic")

    panel(ax, 0.04, 0.245, 0.92, 0.220, "3.  VALIDATE  -  three conditions, all "
                                        "required")
    box(ax, 0.06, 0.330, 0.27, 0.075, "Was it found?",
        "found = true", GUARD, 8.8)
    box(ax, 0.365, 0.330, 0.27, 0.075, "Is the span real?",
        "span appears in the context", GUARD, 8.8)
    box(ax, 0.67, 0.330, 0.28, 0.075, "Is the value in its span?",
        "value occurs inside the span", GUARD, 8.8)
    ax.text(0.50, 0.283,
            "Any condition failing produces an abstention. A fabricated span "
            "cannot pass,\nso a value the document does not contain cannot "
            "reach the user.",
            ha="center", va="center", fontsize=7.6, color=GUARD[1],
            linespacing=1.5)

    panel(ax, 0.04, 0.035, 0.92, 0.180, "4.  COMPUTE  -  Python only")
    box(ax, 0.06, 0.098, 0.26, 0.070, "Validated operands",
        "revenue 12,000  cost 7,400", DATA, 8.8)
    box(ax, 0.36, 0.098, 0.24, 0.070, "Python operation",
        "8 operations", MODEL, 8.8)
    box(ax, 0.64, 0.098, 0.31, 0.070, "Answer with working",
        "38.33% (revenue = 12,000.00, ...)", LOGIC, 8.8)
    arrow(ax, (0.32, 0.133), (0.36, 0.133))
    arrow(ax, (0.60, 0.133), (0.64, 0.133))
    ax.text(0.50, 0.055,
            "A model that returns a wrong total cannot make the pipeline wrong, "
            "because it is never asked for one.",
            ha="center", va="center", fontsize=8.0, color=INK, fontweight="600")

    arrow(ax, (0.50, 0.700), (0.50, 0.672))
    arrow(ax, (0.50, 0.495), (0.50, 0.467))
    arrow(ax, (0.50, 0.245), (0.50, 0.217))
    return save(figure, "03_tool_pipeline.png")


# --------------------------------------------------------------- figure 4
def document_lifecycle():
    figure, ax = canvas(11, 4.6, "Aivora - Document Lifecycle",
                        "From an uploaded filing to an answerable, scoped "
                        "record")

    # Captions are wrapped to two short lines: a single line of this length
    # overflows a box this narrow.
    stages = [
        ("Upload", "POST\n/api/document/upload", CLIENT),
        ("Validate", "extension, size,\npath confine", GUARD),
        ("Store", "data/\ndocuments/", DATA),
        ("Chunk", "document_\nloader.py", LOGIC),
        ("Index", "TF-IDF +\nSQLite", DATA),
        ("Register", "joins the\nworkspace", GUARD),
        ("Answerable", "scoped Q&A\n+ extraction", MODEL),
    ]
    width, gap = 0.117, 0.019
    for i, (title, sub, palette) in enumerate(stages):
        x = 0.035 + i * (width + gap)
        box(ax, x, 0.50, width, 0.19, title, sub, palette, 9.2)
        if i < len(stages) - 1:
            arrow(ax, (x + width, 0.595), (x + width + gap, 0.595))

    box(ax, 0.26, 0.235, 0.48, 0.095, "Workspace scope",
        "a document answers only for the company it belongs to", GUARD, 9.5)
    arrow(ax, (0.80, 0.50), (0.64, 0.332), rad=-0.18)

    ax.text(0.5, 0.135,
            "Before this rule existed, \"What was revenue?\" returned another "
            "company's figure - correctly cited, and completely wrong.",
            ha="center", va="center", fontsize=8.2, color=GUARD[1],
            fontweight="600")
    ax.text(0.5, 0.065,
            "Statuses surfaced to the interface: Uploading - Processing - Ready "
            "- Failed. Progress is reported, never simulated.",
            ha="center", va="center", fontsize=7.8, color=MUTED)
    return save(figure, "04_document_lifecycle.png")


def main():
    paths = [system_architecture(), request_routing(), tool_pipeline(),
             document_lifecycle()]
    print(f"\n{len(paths)} diagrams written to {OUT_DIR}/")


if __name__ == "__main__":
    main()
