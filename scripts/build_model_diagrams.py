"""Diagrams for the Aivora model documentation.

Figures are generated from this file, and the evaluation chart is plotted from
the measured numbers in reports/, so a figure cannot drift from the result it
depicts.

    python scripts/build_model_diagrams.py
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

ATTN = "#dbeafe", "#1d4ed8"
MOE = "#dcfce7", "#15803d"
MTP = "#fef3c7", "#b45309"
NORM = "#ede9fe", "#6d28d9"
DATA = "#f3e8ff", "#7e22ce"
WARN = "#fee2e2", "#b91c1c"


def box(ax, x, y, w, h, title, subtitle=None, palette=MOE, fontsize=9.0):
    face, edge = palette
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.010,rounding_size=0.018",
        facecolor=face, edgecolor=edge, linewidth=1.3, zorder=3))
    if subtitle:
        ax.text(x + w / 2, y + h * 0.63, title, ha="center", va="center",
                fontsize=fontsize, fontweight="600", color=INK, zorder=4)
        ax.text(x + w / 2, y + h * 0.27, subtitle, ha="center", va="center",
                fontsize=min(7.5, fontsize - 0.9), color=MUTED, zorder=4,
                linespacing=1.45)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center",
                fontsize=fontsize, fontweight="600", color=INK, zorder=4)


def panel(ax, x, y, w, h, label):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.006,rounding_size=0.012",
        facecolor=PANEL, edgecolor=PANEL_EDGE, linewidth=1, zorder=1))
    ax.text(x + 0.004, y + h + 0.009, label, ha="left", va="bottom",
            fontsize=7.8, color=MUTED, fontweight="600", zorder=2)


def arrow(ax, start, end, label=None, colour=LINE, dashed=False, rad=0.0,
          offset=(0, 0.016)):
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=11, color=colour,
        linewidth=1.2, zorder=5, shrinkA=2, shrinkB=2,
        linestyle="--" if dashed else "-",
        connectionstyle=f"arc3,rad={rad}"))
    if label:
        ax.text((start[0] + end[0]) / 2 + offset[0],
                (start[1] + end[1]) / 2 + offset[1], label, ha="center",
                va="center", fontsize=7.4, color=MUTED, zorder=6,
                bbox=dict(boxstyle="round,pad=0.18", facecolor="white",
                          edgecolor="none", alpha=0.9))


def canvas(width, height, title, subtitle):
    figure, ax = plt.subplots(figsize=(width, height))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.5, 0.982, title, ha="center", va="top", fontsize=13,
            fontweight="700", color=INK)
    ax.text(0.5, 0.943, subtitle, ha="center", va="top", fontsize=8.6,
            color=MUTED)
    return figure, ax


def save(figure, name):
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, name)
    figure.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    print(f"wrote {path}")
    return path


# --------------------------------------------------------------- figure 5
def model_architecture():
    figure, ax = canvas(10.5, 8.6, "Aivora Model - Architecture",
                        "DeepSeek-V3 inspired: 101.7M parameters, 8 layers, "
                        "512 hidden, 1,024 context")

    box(ax, 0.30, 0.862, 0.40, 0.048, "Token IDs", "GPT-2 BPE, 50,257 vocab",
        DATA, 9.0)
    box(ax, 0.30, 0.796, 0.40, 0.048, "Token + position embeddings",
        "25.73M + 0.52M parameters", DATA, 9.0)
    arrow(ax, (0.50, 0.862), (0.50, 0.846))

    panel(ax, 0.06, 0.300, 0.88, 0.468, "TRANSFORMER BLOCK  x8   (66.61M "
                                        "parameters, 65.5% of the model)")
    arrow(ax, (0.50, 0.796), (0.50, 0.772))

    box(ax, 0.10, 0.700, 0.80, 0.042, "RMSNorm", None, NORM, 8.6)

    # --- Multi-head Latent Attention
    panel(ax, 0.10, 0.470, 0.37, 0.195, "MULTI-HEAD LATENT ATTENTION")
    box(ax, 0.12, 0.590, 0.33, 0.055, "KV compression",
        "down-project to rank 128", ATTN, 8.6)
    box(ax, 0.12, 0.525, 0.155, 0.055, "Query LoRA", "rank 192", ATTN, 8.4)
    box(ax, 0.295, 0.525, 0.155, 0.055, "RoPE", "32 dims", ATTN, 8.4)
    box(ax, 0.12, 0.482, 0.33, 0.036, "8 heads -> output projection", None,
        ATTN, 8.2)
    arrow(ax, (0.285, 0.590), (0.285, 0.580), colour=LINE)

    # --- Mixture of Experts
    panel(ax, 0.53, 0.470, 0.37, 0.195, "MIXTURE OF EXPERTS")
    box(ax, 0.55, 0.590, 0.33, 0.055, "Router", "top-2 of 8 experts", MOE, 8.6)
    for i in range(4):
        box(ax, 0.552 + i * 0.0835, 0.527, 0.077, 0.050,
            f"E{i + 1}", None, MOE, 7.8)
    box(ax, 0.55, 0.482, 0.33, 0.036, "Shared expert  (always active, 768 dim)",
        None, MOE, 8.0)
    arrow(ax, (0.715, 0.590), (0.715, 0.580), colour=LINE)

    ax.text(0.285, 0.447, "attention", ha="center", fontsize=8, color=MUTED)
    ax.text(0.715, 0.447, "feed-forward", ha="center", fontsize=8, color=MUTED)

    box(ax, 0.10, 0.372, 0.80, 0.042, "Residual + RMSNorm", None, NORM, 8.6)
    arrow(ax, (0.285, 0.470), (0.40, 0.416), rad=0.0)
    arrow(ax, (0.715, 0.470), (0.60, 0.416), rad=0.0)

    ax.text(0.50, 0.330,
            "Only 2 of 8 experts run per token, so activated parameters are a "
            "fraction of the total.",
            ha="center", va="center", fontsize=7.8, color=MUTED, style="italic")

    arrow(ax, (0.50, 0.300), (0.50, 0.268))
    box(ax, 0.10, 0.205, 0.37, 0.058, "Next-token head",
        "tied to token embeddings", DATA, 8.8)
    box(ax, 0.53, 0.205, 0.37, 0.058, "Multi-Token Prediction",
        "1 extra head, 8.85M params, loss weight 0.3", MTP, 8.8)
    arrow(ax, (0.40, 0.268), (0.285, 0.263))
    arrow(ax, (0.60, 0.268), (0.715, 0.263))

    box(ax, 0.26, 0.108, 0.48, 0.052, "Cross-entropy loss",
        "ignore_index = -1 masks prompt tokens", WARN, 8.8)
    arrow(ax, (0.285, 0.205), (0.42, 0.160))
    arrow(ax, (0.715, 0.205), (0.58, 0.160))

    ax.text(0.5, 0.062,
            "Prompt tokens are masked to -1 so loss is computed only on answer "
            "tokens.",
            ha="center", va="center", fontsize=8.0, color=INK, fontweight="600")
    ax.text(0.5, 0.022,
            "MoE reduces compute per token; MLA reduces the KV cache; MTP adds "
            "a lookahead training signal.",
            ha="center", va="center", fontsize=7.8, color=MUTED)
    return save(figure, "05_model_architecture.png")


# --------------------------------------------------------------- figure 6
def training_pipeline():
    figure, ax = canvas(11.0, 5.4, "Aivora Model - Training Pipeline",
                        "From raw corpus to an evaluated checkpoint")

    stages = [
        ("Corpora", "financial text,\ninstructions", DATA),
        ("Tokenise", "GPT-2 BPE\n50,257", DATA),
        ("Pack", "1,024-token\nsequences", DATA),
        ("Mask", "prompt tokens\nset to -1", WARN),
        ("Pre-train", "120.9M tokens\n16.3 epochs", MOE),
        ("SFT", "supervised\nfine-tuning", MOE),
        ("Gate", "capability gates\nA-E", WARN),
    ]
    width, gap = 0.117, 0.019
    for i, (title, sub, palette) in enumerate(stages):
        x = 0.035 + i * (width + gap)
        box(ax, x, 0.545, width, 0.185, title, sub, palette, 9.0)
        if i < len(stages) - 1:
            arrow(ax, (x + width, 0.638), (x + width + gap, 0.638))

    box(ax, 0.20, 0.330, 0.26, 0.095, "Checkpoint accepted",
        "all five gates pass", MOE, 9.0)
    box(ax, 0.54, 0.330, 0.26, 0.095, "Checkpoint rejected",
        "a gate regressed", WARN, 9.0)
    arrow(ax, (0.83, 0.545), (0.40, 0.428), rad=0.12, label="pass")
    arrow(ax, (0.89, 0.545), (0.70, 0.428), rad=-0.12, label="fail")

    ax.text(0.5, 0.235,
            "Checkpoints are selected by capability gates, never by training "
            "loss.",
            ha="center", va="center", fontsize=9.0, color=INK, fontweight="600")
    ax.text(0.5, 0.165,
            "SFT_001 improved loss monotonically while destroying the model's "
            "ability to copy a value (5/6 -> 0/6).",
            ha="center", va="center", fontsize=8.2, color=WARN[1])
    ax.text(0.5, 0.100,
            "Gates: A copy  ·  B extraction  ·  C calculation  ·  D abstention  "
            "·  E unseen wording",
            ha="center", va="center", fontsize=7.8, color=MUTED)
    ax.text(0.5, 0.040,
            "Token / parameter ratio 1.19 against a compute-optimal 20: the "
            "model is under-trained by about 17x.",
            ha="center", va="center", fontsize=7.8, color=MUTED, style="italic")
    return save(figure, "06_training_pipeline.png")


# --------------------------------------------------------------- figure 7
def evaluation_results():
    """Measured results. Numbers come from reports/phase4 and reports/heldout."""
    figure, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.6))

    gates = ["Copy", "Extraction", "Wording", "Calculation", "Abstention"]
    baseline = [13.0, 1.4, 2.7, 0.0, 0.0]
    sft002 = [34.8, 13.0, 13.3, 2.1, 16.7]
    pipeline = [98.6, 87.0, 85.3, 100.0, 97.2]

    x = range(len(gates))
    offset = 0.26
    ax1.bar([i - offset for i in x], baseline, offset, label="Base model",
            color="#fca5a5", edgecolor="#b91c1c", linewidth=0.8)
    ax1.bar(list(x), sft002, offset, label="After fine-tuning (SFT_002)",
            color="#fcd34d", edgecolor="#b45309", linewidth=0.8)
    ax1.bar([i + offset for i in x], pipeline, offset,
            label="Tool pipeline + Qwen2.5-1.5B", color="#86efac",
            edgecolor="#15803d", linewidth=0.8)
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(gates, fontsize=8.5)
    ax1.set_ylabel("Accuracy (%)", fontsize=9)
    ax1.set_ylim(0, 128)
    ax1.set_title("Accuracy by capability gate", fontsize=10.5,
                  fontweight="600", color=INK)
    # Above the bars, not over them: the tallest bar reaches 100.
    ax1.legend(fontsize=7.4, loc="upper center", ncol=3,
               bbox_to_anchor=(0.5, 1.02), framealpha=0.95,
               borderpad=0.5)
    ax1.grid(axis="y", alpha=0.25, linewidth=0.7)
    ax1.set_axisbelow(True)
    for spine in ("top", "right"):
        ax1.spines[spine].set_visible(False)

    systems = ["Base\nmodel", "SFT_002", "Tool\npipeline"]
    invented = [42.9, 36.8, 0.0]
    bars = ax2.bar(systems, invented,
                   color=["#fca5a5", "#fcd34d", "#86efac"],
                   edgecolor=["#b91c1c", "#b45309", "#15803d"], linewidth=0.9,
                   width=0.55)
    for bar, value in zip(bars, invented):
        ax2.text(bar.get_x() + bar.get_width() / 2, value + 1.6,
                 f"{value:.1f}%", ha="center", fontsize=9.5, fontweight="600",
                 color=INK)
    ax2.set_ylabel("Answers containing an invented value (%)", fontsize=9)
    ax2.set_ylim(0, 52)
    ax2.set_title("Invented values - lower is better", fontsize=10.5,
                  fontweight="600", color=INK)
    ax2.grid(axis="y", alpha=0.25, linewidth=0.7)
    ax2.set_axisbelow(True)
    for spine in ("top", "right"):
        ax2.spines[spine].set_visible(False)

    figure.suptitle("Aivora Model - Measured Results on the Frozen Held-Out Set "
                    "(n=313)", fontsize=12, fontweight="700", color=INK,
                    y=1.02)
    figure.text(0.5, -0.04,
                "Scored once per configuration by one shared scorer, with "
                "identical prompts and decoding. Base-model and SFT_002 figures "
                "are from the same frozen set.",
                ha="center", fontsize=7.8, color=MUTED)
    figure.tight_layout()
    return save(figure, "07_evaluation_results.png")


def main():
    model_architecture()
    training_pipeline()
    evaluation_results()
    print(f"\n3 model diagrams written to {OUT_DIR}/")


if __name__ == "__main__":
    main()
