#!/usr/bin/env python3
"""Generate publication-quality plots for the CS 7643 final paper.

Creates 11 figures:
  Existing:
    1. technique_impact.png — horizontal bar chart of delta BPB per technique
    2. slm_sweep.png — line plot of BPB vs SLM retention ratio k
    3. transfer.png — grouped bar chart comparing 2xH100 vs 8xH100 deltas
  New (Part A — compilation blockers):
    4. attention_variants.png — bar chart of attention variant BPB
    5. training_curves.png — BPB vs wall-clock time
    6. paper_overview.png — taxonomy diagram (score/budget layers)
  New (Part B — supplementary):
    7. ema_sweep.png — EMA decay sensitivity line plot
    8. ttt_grid.png — TTT fine-tuning heatmap
    9. v2_factorial.png — V2 3x3 factorial heatmap
   10. gptq_gap.png — GPTQ compression gap bar chart
   11. resformer_sweep.png — ResFormer alpha sweep

Usage:
    python tools/plot_paper.py
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np


OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "docs", "plots")


def setup_academic_theme():
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#333333",
        "axes.labelcolor": "#111111",
        "axes.grid": True,
        "grid.color": "#cccccc",
        "grid.alpha": 0.5,
        "grid.linewidth": 0.5,
        "text.color": "#111111",
        "xtick.color": "#333333",
        "ytick.color": "#333333",
        "font.family": "serif",
        "font.size": 10,
        "figure.dpi": 300,
        "savefig.facecolor": "white",
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.15,
    })


# ═══════════════════════════════════════════════════════════════════════
# Existing plots (unchanged)
# ═══════════════════════════════════════════════════════════════════════

def plot_technique_impact():
    """Horizontal bar chart: delta BPB vs control at 2xH100."""
    techniques = [
        ("Small Batch",        -0.0153),
        ("EMA tuning (0.990)", -0.0117),
        ("Headwise gate*",     -0.0005),
        ("SLM k=0.95",        +0.002),
        ("LR Warmup (2%)",    +0.0024),
        ("ResFormer \u03b1=0.5", +0.0025),
        ("HybridNorm",        +0.011),
        ("Diff. Attention",   +0.0138),
        ("Structured FFN",    +0.0425),
    ]
    peri_ln_x = 0.055  # display position for NaN technique

    names = [t[0] for t in techniques] + ["Peri-LN"]
    deltas = [t[1] for t in techniques] + [peri_ln_x]

    colors = ["#2e7d32" if d < 0 else "#c62828" for d in deltas]
    colors[-1] = "#888888"  # Peri-LN is gray (NaN)

    setup_academic_theme()
    fig, ax = plt.subplots(figsize=(6, 4))

    y_pos = np.arange(len(names))
    bars = ax.barh(y_pos, deltas, color=colors, edgecolor="white", linewidth=0.5, height=0.65)

    # Peri-LN: hatch pattern to indicate NaN
    bars[-1].set_hatch("///")
    bars[-1].set_edgecolor("#555555")
    ax.annotate("NaN", xy=(peri_ln_x, y_pos[-1]), xytext=(5, 0),
                textcoords="offset points", fontsize=8, va="center",
                fontstyle="italic", color="#555555")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=9)
    ax.invert_yaxis()
    ax.axvline(x=0, color="#111111", linewidth=0.8)
    ax.set_xlabel("\u0394 BPB vs. control (negative = better)", fontsize=10)
    ax.set_title("Technique Impact on BPB (2\u00d7H100)", fontsize=11, fontweight="bold")

    # Annotate novel technique
    ax.annotate("* novel contribution", xy=(0, 0), xycoords="axes fraction",
                xytext=(0.98, 0.02), textcoords="axes fraction",
                fontsize=7, ha="right", va="bottom", fontstyle="italic", color="#555555")

    fig.tight_layout()
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, "technique_impact.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_slm_sweep():
    """Line plot: BPB vs SLM retention ratio k."""
    # SP8192 combo slim + TTT, 2xH100
    sp8192_k =   [1.0,    0.8,    0.7,    0.6]
    sp8192_bpb = [1.2411, 1.2652, 1.3183, 1.4002]

    # SP1024 GQA baseline, 2xH100
    sp1024_k =   [1.0,    0.95,   0.6]
    sp1024_bpb = [1.2649, 1.2668, 1.4204]

    setup_academic_theme()
    fig, ax = plt.subplots(figsize=(6, 4))

    ax.plot(sp8192_k, sp8192_bpb, "o-", color="#1565c0", linewidth=1.8,
            markersize=6, label="SP8192 (combo slim + TTT)", zorder=3)
    ax.plot(sp1024_k, sp1024_bpb, "s--", color="#e65100", linewidth=1.5,
            markersize=5, label="SP1024 (GQA baseline)", zorder=3)

    # Baseline reference lines
    ax.axhline(y=1.2411, color="#1565c0", linestyle=":", linewidth=0.8, alpha=0.5)
    ax.axhline(y=1.2649, color="#e65100", linestyle=":", linewidth=0.8, alpha=0.5)

    ax.set_xlabel("Retention ratio k (1.0 = no filtering)", fontsize=10)
    ax.set_ylabel("BPB (lower is better)", fontsize=10)
    ax.set_title("Rho-1 SLM: BPB vs. Retention Ratio", fontsize=11, fontweight="bold")
    ax.set_xlim(0.55, 1.05)
    ax.invert_xaxis()

    ax.annotate("optimal: k = 1.0\n(no filtering)", xy=(1.0, 1.2411),
                xytext=(-60, 30), textcoords="offset points", fontsize=8,
                arrowprops=dict(arrowstyle="->", color="#333333", lw=0.8),
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc", lw=0.5))

    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "slm_sweep.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_transfer():
    """Grouped bar chart: 2xH100 vs 8xH100 delta BPB."""
    techniques = ["Headwise gate", "ResFormer \u03b1=0.5", "EMA=0.990", "Small Batch\n+ EMA"]
    delta_2gpu = [-0.0005, +0.0025, -0.0117, -0.0254]
    delta_8gpu = [-0.0005, +0.0022, +0.0025, +0.0121]

    setup_academic_theme()
    fig, ax = plt.subplots(figsize=(6, 4))

    x = np.arange(len(techniques))
    w = 0.32

    bars_2 = ax.bar(x - w/2, delta_2gpu, w, label="2\u00d7H100", color="#1565c0",
                    edgecolor="white", linewidth=0.5)
    bars_8 = ax.bar(x + w/2, delta_8gpu, w, label="8\u00d7H100", color="#e65100",
                    edgecolor="white", linewidth=0.5)

    ax.axhline(y=0, color="#111111", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(techniques, fontsize=8.5)
    ax.set_ylabel("\u0394 BPB vs. control", fontsize=10)
    ax.set_title("Transfer: 2\u00d7H100 \u2192 8\u00d7H100", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9, loc="upper left", framealpha=0.9)

    # Annotate sign flips
    for i in [2, 3]:  # EMA and Small Batch
        mid_x = x[i]
        top = max(delta_2gpu[i], delta_8gpu[i])
        ax.annotate("sign\nflipped", xy=(mid_x, top + 0.002),
                    fontsize=7, ha="center", va="bottom", color="#c62828",
                    fontweight="bold")

    fig.tight_layout()
    out = os.path.join(OUT_DIR, "transfer.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved: {out}")


# ═══════════════════════════════════════════════════════════════════════
# Part A: Fix 3 missing figures (compilation blockers)
# ═══════════════════════════════════════════════════════════════════════

def plot_attention_variants():
    """A1: Bar chart of attention variants at 2xH100, SP1024."""
    variants = ["Elementwise\ngate", "Headwise\ngate", "GQA\n(baseline)", "MQA"]
    bpb =      [1.2602,           1.2653,          1.2667,            1.2761]
    sizes =    ["17.87 MB",       "15.75 MB",      "15.75 MB",        "16.84 MB"]
    under_budget = [False,        True,            True,               False]

    setup_academic_theme()
    fig, ax = plt.subplots(figsize=(5.5, 4))

    x = np.arange(len(variants))
    colors = ["#2e7d32" if ub else "#c62828" for ub in under_budget]

    bars = ax.bar(x, bpb, color=colors, edgecolor="white", linewidth=0.5, width=0.6)

    # Hatch over-budget bars
    for i, bar in enumerate(bars):
        if not under_budget[i]:
            bar.set_hatch("///")
            bar.set_edgecolor("#888888")

    # Annotate sizes
    for i, (bar, sz) in enumerate(zip(bars, sizes)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0005,
                sz, ha="center", va="bottom", fontsize=8, color="#555555")

    # GQA baseline reference
    ax.axhline(y=1.2667, color="#333333", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.text(3.4, 1.2667 + 0.0003, "GQA baseline", fontsize=7, color="#555555", ha="right")

    ax.set_xticks(x)
    ax.set_xticklabels(variants, fontsize=9)
    ax.set_ylabel("BPB (lower is better)", fontsize=10)
    ax.set_title("Attention Variants (2\u00d7H100, SP1024)", fontsize=11, fontweight="bold")
    ax.set_ylim(1.255, 1.282)

    # Legend
    green_patch = mpatches.Patch(facecolor="#2e7d32", label="Under 16 MB")
    red_patch = mpatches.Patch(facecolor="#c62828", hatch="///", edgecolor="#888888",
                               label="Over 16 MB")
    ax.legend(handles=[green_patch, red_patch], fontsize=8, loc="upper right", framealpha=0.9)

    fig.tight_layout()
    out = os.path.join(OUT_DIR, "attention_variants.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_training_curves():
    """A2: BPB vs wall-clock time for baseline and V2 stack."""
    # Baseline: SP1024 GQA, 2xH100, 171ms/step, ~3500 steps in 600s
    # Interpolated from findings.md
    baseline_time = [0, 171, 342, 513, 600]  # seconds (approximate)
    baseline_bpb =  [4.11, 1.65, 1.38, 1.33, 1.27]

    # V2 stack: SP8192 combo, 2xH100, ~140ms/step, ~4200 steps in 600s
    # Interpolated from findings.md
    v2_time = [0, 140, 280, 420, 600]
    v2_bpb =  [4.11, 1.50, 1.30, 1.26, 1.17]

    setup_academic_theme()
    fig, ax = plt.subplots(figsize=(6, 4))

    ax.plot(baseline_time, baseline_bpb, "-", color="#1565c0", linewidth=2.0,
            label="Baseline (SP1024 GQA)", zorder=3)
    ax.plot(v2_time, v2_bpb, "-", color="#2e7d32", linewidth=2.0,
            label="V2 Stack (SP8192)", zorder=3)

    # Mark 120s divergence point
    ax.axvline(x=120, color="#888888", linestyle=":", linewidth=0.8, alpha=0.6)
    ax.annotate("V2 diverges", xy=(120, 2.0), xytext=(150, 2.4),
                fontsize=8, color="#555555",
                arrowprops=dict(arrowstyle="->", color="#888888", lw=0.8))

    # Mark end still dropping
    ax.annotate("still dropping\nat 600s", xy=(590, 1.17), xytext=(460, 1.5),
                fontsize=8, color="#2e7d32",
                arrowprops=dict(arrowstyle="->", color="#2e7d32", lw=0.8))

    ax.set_xlabel("Wall-clock time (seconds)", fontsize=10)
    ax.set_ylabel("Validation BPB", fontsize=10)
    ax.set_title("Validation BPB vs. Wall-Clock Time", fontsize=11, fontweight="bold")
    ax.set_xlim(-10, 630)
    ax.set_ylim(1.0, 4.3)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)

    fig.tight_layout()
    out = os.path.join(OUT_DIR, "training_curves.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_paper_overview():
    """A3: Conceptual taxonomy diagram — score layer + budget layer."""
    setup_academic_theme()
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.axis("off")
    ax.set_aspect("equal")

    # Colors
    c_green = "#c8e6c9"   # tested, helped
    c_red = "#ffcdd2"     # tested, hurt
    c_gray = "#e0e0e0"    # not tested
    c_header_score = "#1565c0"
    c_header_budget = "#e65100"

    def draw_box(x, y, w, h, color, label, items, fontsize=7):
        rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05",
                              facecolor=color, edgecolor="#555555", linewidth=0.8)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h - 0.18, label, ha="center", va="top",
                fontsize=8, fontweight="bold", color="#222222")
        for i, item in enumerate(items):
            ax.text(x + 0.1, y + h - 0.42 - i*0.22, item, ha="left", va="top",
                    fontsize=fontsize, color="#333333")

    # Title
    ax.text(5, 6.8, "Parameter Golf Technique Space", ha="center", va="top",
            fontsize=13, fontweight="bold")

    # Score Layer header
    ax.text(5, 6.35, "SCORE LAYER", ha="center", va="top",
            fontsize=10, fontweight="bold", color=c_header_score)
    ax.text(5, 6.1, "(directly improves BPB)", ha="center", va="top",
            fontsize=8, color="#555555")

    # Score layer boxes (y=3.8 to 5.8)
    score_y = 3.9
    score_h = 1.9
    score_w = 2.2
    gap = 0.27

    draw_box(0.2, score_y, score_w, score_h, c_red,
             "Architecture",
             ["x Structured FFN", "x Peri-LN", "x HybridNorm",
              "+ Depth recurrence*"])

    draw_box(0.2 + score_w + gap, score_y, score_w, score_h, c_green,
             "Attention",
             ["+ Headwise gate", "x Diff. Attention",
              "+ XSA*", "+ QK-Gain*"])

    draw_box(0.2 + 2*(score_w + gap), score_y, score_w, score_h, c_red,
             "Optimization",
             ["+ Small Batch", "+ EMA tuning",
              "x LR Warmup", "x ResFormer"])

    draw_box(0.2 + 3*(score_w + gap), score_y, score_w, score_h, c_red,
             "Data Selection",
             ["x SLM (Rho-1)", "", "", ""])

    # Divider
    ax.axhline(y=3.6, xmin=0.03, xmax=0.97, color="#aaaaaa", linewidth=1.0, linestyle="-")

    # Budget Layer header
    ax.text(5, 3.4, "BUDGET LAYER", ha="center", va="top",
            fontsize=10, fontweight="bold", color=c_header_budget)
    ax.text(5, 3.15, "(frees bytes/steps for score-layer capacity)", ha="center", va="top",
            fontsize=8, color="#555555")

    # Budget layer boxes (y=0.8 to 2.7)
    budget_y = 1.0
    budget_h = 1.9
    budget_w = 2.2

    draw_box(0.2, budget_y, budget_w, budget_h, c_green,
             "Compression",
             ["+ GPTQ int6", "+ Brotli/LZMA",
              "+ Embed clip", ""])

    draw_box(0.2 + budget_w + gap, budget_y, budget_w, budget_h, c_green,
             "Tokenizer",
             ["+ SP8192", "+ CaseOps*",
              "", ""])

    draw_box(0.2 + 2*(budget_w + gap), budget_y, budget_w, budget_h, c_green,
             "Hardware/Framework",
             ["+ FlashAttention-3", "+ PyTorch 2.11",
              "+ 8xH100 SXM", ""])

    draw_box(0.2 + 3*(budget_w + gap), budget_y, budget_w, budget_h, c_green,
             "Training Efficiency",
             ["+ Muon optimizer*", "+ EMA*",
              "+ Warmdown*", ""])

    # Legend
    leg_y = 0.3
    for i, (color, label) in enumerate([
        (c_green, "Tested & helped"),
        (c_red, "Tested & hurt"),
        (c_gray, "Not tested"),
    ]):
        rect = FancyBboxPatch((0.3 + i*3.2, leg_y), 0.3, 0.25,
                              boxstyle="round,pad=0.02", facecolor=color,
                              edgecolor="#555555", linewidth=0.5)
        ax.add_patch(rect)
        ax.text(0.7 + i*3.2, leg_y + 0.12, label, ha="left", va="center",
                fontsize=8, color="#333333")

    ax.text(9.5, 0.42, "* adopted from\nleaderboard", ha="right", va="center",
            fontsize=7, color="#555555", fontstyle="italic")

    fig.tight_layout()
    out = os.path.join(OUT_DIR, "paper_overview.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved: {out}")


# ═══════════════════════════════════════════════════════════════════════
# Part B: High-value supplementary plots
# ═══════════════════════════════════════════════════════════════════════

def plot_ema_sweep():
    """B1: EMA decay sensitivity — line plot with cliff."""
    decay =  [0.990,  0.993,  0.995,  0.9965, 0.997,  0.999]
    bpb =    [1.1505, 1.1526, 1.1562, 1.1622, 1.1690, 1.3475]

    setup_academic_theme()
    fig, ax = plt.subplots(figsize=(6, 4))

    ax.plot(decay, bpb, "o-", color="#1565c0", linewidth=2.0, markersize=7, zorder=3)

    # Highlight optimal
    ax.scatter([0.990], [1.1505], color="#2e7d32", s=100, zorder=4, marker="*")
    ax.annotate("optimal\n(0.990)", xy=(0.990, 1.1505), xytext=(-50, 20),
                textcoords="offset points", fontsize=8, color="#2e7d32",
                fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#2e7d32", lw=0.8))

    # Highlight catastrophic
    ax.annotate("catastrophic\n(+0.185 BPB)", xy=(0.999, 1.3475), xytext=(-70, -20),
                textcoords="offset points", fontsize=8, color="#c62828",
                fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#c62828", lw=0.8))

    # Default marker
    ax.axvline(x=0.9965, color="#888888", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.text(0.9965, 1.14, "default", fontsize=7, ha="center", color="#888888")

    ax.set_xlabel("EMA Decay", fontsize=10)
    ax.set_ylabel("TTT BPB (lower is better)", fontsize=10)
    ax.set_title("EMA Decay Sensitivity (2\u00d7H100)", fontsize=11, fontweight="bold")

    fig.tight_layout()
    out = os.path.join(OUT_DIR, "ema_sweep.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_ttt_grid():
    """B2: TTT fine-tuning heatmap (3x3)."""
    epochs = [3, 5, 7]
    lrs = [0.003, 0.005, 0.01]
    # Data from Session 15 (rows=epochs, cols=LRs)
    data = np.array([
        [1.1665, 1.1650, 1.1633],
        [1.1643, 1.1634, 1.1636],
        [1.1631, 1.1624, 1.1629],
    ])

    setup_academic_theme()
    fig, ax = plt.subplots(figsize=(5, 4))

    im = ax.imshow(data, cmap="RdYlGn_r", aspect="auto",
                   vmin=data.min() - 0.001, vmax=data.max() + 0.001)

    # Annotate cells
    for i in range(3):
        for j in range(3):
            color = "white" if data[i, j] > 1.165 else "black"
            weight = "bold" if (i, j) == (2, 1) else "normal"
            ax.text(j, i, f"{data[i, j]:.4f}", ha="center", va="center",
                    fontsize=9, color=color, fontweight=weight)

    # Mark default (0,1) and best (2,1)
    ax.text(1, 0, "\n\n(default)", ha="center", va="top", fontsize=6, color="white")
    ax.text(1, 2, "\n\n(best)", ha="center", va="top", fontsize=6, color="black")

    ax.set_xticks(range(3))
    ax.set_xticklabels([f"{lr}" for lr in lrs])
    ax.set_yticks(range(3))
    ax.set_yticklabels([f"{ep} ep" for ep in epochs])
    ax.set_xlabel("Learning Rate", fontsize=10)
    ax.set_ylabel("Epochs", fontsize=10)
    ax.set_title("TTT Fine-Tuning Grid (BPB)", fontsize=11, fontweight="bold")

    plt.colorbar(im, ax=ax, shrink=0.8, label="BPB")
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "ttt_grid.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_v2_factorial():
    """B3: V2 3x3 factorial heatmap (residual x gate type)."""
    residuals = ["PR only", "RF only", "PR + RF"]
    gates = ["No Gate", "Headwise", "Elementwise"]
    # Data from Session 12
    data = np.array([
        [1.1641, 1.1636, 1.1665],
        [1.1666, 1.1661, 1.1700],
        [1.1636, 1.1650, 1.1686],
    ])
    over_budget = np.array([
        [False, False, True],
        [False, False, True],
        [False, False, True],
    ])

    setup_academic_theme()
    fig, ax = plt.subplots(figsize=(5.5, 4))

    im = ax.imshow(data, cmap="RdYlGn_r", aspect="auto",
                   vmin=data.min() - 0.001, vmax=data.max() + 0.001)

    # Annotate cells
    for i in range(3):
        for j in range(3):
            txt = f"{data[i, j]:.4f}"
            if over_budget[i, j]:
                txt += "*"
            color = "white" if data[i, j] > 1.167 else "black"
            ax.text(j, i, txt, ha="center", va="center", fontsize=9,
                    color=color, fontweight="bold" if data[i, j] == data.min() else "normal")

    # Hatch over-budget column
    for i in range(3):
        ax.add_patch(plt.Rectangle((2 - 0.5, i - 0.5), 1, 1,
                                   fill=False, hatch="//", edgecolor="#c62828",
                                   linewidth=0, alpha=0.3))

    ax.set_xticks(range(3))
    ax.set_xticklabels(gates)
    ax.set_yticks(range(3))
    ax.set_yticklabels(residuals)
    ax.set_xlabel("Gate Type", fontsize=10)
    ax.set_ylabel("Residual Connection", fontsize=10)
    ax.set_title("V2 Factorial: Residual \u00d7 Gate Type (BPB)", fontsize=11, fontweight="bold")

    ax.text(0.98, -0.12, "* over 16 MB budget", transform=ax.transAxes,
            fontsize=7, ha="right", color="#c62828", fontstyle="italic")

    plt.colorbar(im, ax=ax, shrink=0.8, label="BPB")
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "v2_factorial.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_gptq_gap():
    """B4: GPTQ compression gap — horizontal bar chart."""
    methods = [
        "Kevin Clark (rank 5)",
        "dexhunter (rank 7)",
        "Us \u2014 baseline",
        "Us \u2014 sequential",
        "Us \u2014 embed GPTQ",
        "Us \u2014 all combined",
    ]
    gaps = [0.012, 0.010, 0.054, 0.188, 0.487, 0.664]

    setup_academic_theme()
    fig, ax = plt.subplots(figsize=(6, 4))

    y_pos = np.arange(len(methods))
    colors = ["#2e7d32", "#2e7d32", "#1565c0", "#c62828", "#c62828", "#c62828"]

    bars = ax.barh(y_pos, gaps, color=colors, edgecolor="white", linewidth=0.5, height=0.6)

    # Leaderboard best reference
    ax.axvline(x=0.012, color="#2e7d32", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.text(0.012, -0.7, "leaderboard\nbest", fontsize=7, ha="center",
            color="#2e7d32", va="bottom")

    # Annotate values
    for bar, gap in zip(bars, gaps):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f"+{gap:.3f}", va="center", fontsize=8, color="#555555")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(methods, fontsize=9)
    ax.set_xlabel("GPTQ Gap (BPB degradation from quantization)", fontsize=10)
    ax.set_title("GPTQ Compression Gap: Us vs. Leaderboard", fontsize=11, fontweight="bold")
    ax.set_xlim(0, 0.75)

    fig.tight_layout()
    out = os.path.join(OUT_DIR, "gptq_gap.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_resformer_sweep():
    """B5: ResFormer alpha sweep — line plot."""
    alpha =   [0.0,    0.1,    0.5,    0.7]
    pre_q =   [1.2040, 1.2020, 1.2004, 1.2025]
    ttt_bpb = [1.2584, 1.2545, 1.2536, 1.2551]

    setup_academic_theme()
    fig, ax = plt.subplots(figsize=(6, 4))

    ax.plot(alpha, pre_q, "o-", color="#1565c0", linewidth=2.0, markersize=7,
            label="Pre-Quantization BPB", zorder=3)
    ax.plot(alpha, ttt_bpb, "s-", color="#e65100", linewidth=2.0, markersize=7,
            label="Post-TTT BPB", zorder=3)

    # Mark optimal
    ax.scatter([0.5], [1.2004], color="#1565c0", s=120, zorder=4, marker="*")
    ax.scatter([0.5], [1.2536], color="#e65100", s=120, zorder=4, marker="*")
    ax.annotate("optimal (\u03b1=0.5)", xy=(0.5, 1.2004), xytext=(20, -20),
                textcoords="offset points", fontsize=8, color="#1565c0",
                arrowprops=dict(arrowstyle="->", color="#1565c0", lw=0.8))

    ax.set_xlabel("ResFormer \u03b1 (value residual blend)", fontsize=10)
    ax.set_ylabel("BPB (lower is better)", fontsize=10)
    ax.set_title("ResFormer Value Residual: \u03b1 Sweep", fontsize=11, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)

    fig.tight_layout()
    out = os.path.join(OUT_DIR, "resformer_sweep.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved: {out}")


# ═══════════════════════════════════════════════════════════════════════
# Part C: Compact multi-panel figures & table renders
# ═══════════════════════════════════════════════════════════════════════

def plot_hyperparameter_sensitivity():
    """C1: 3-panel — (a) EMA decay, (b) TTT grid, (c) ResFormer alpha."""
    setup_academic_theme()
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 3.5))

    # --- (a) EMA Decay ---
    decay = [0.990, 0.993, 0.995, 0.9965, 0.997, 0.999]
    bpb_ema = [1.1505, 1.1526, 1.1562, 1.1622, 1.1690, 1.3475]

    ax1.plot(decay, bpb_ema, "o-", color="#1565c0", linewidth=1.8, markersize=5, zorder=3)
    ax1.scatter([0.990], [1.1505], color="#2e7d32", s=80, zorder=4, marker="*")
    ax1.axvline(x=0.9965, color="#888888", linestyle="--", linewidth=0.7, alpha=0.6)
    ax1.annotate("optimal", xy=(0.990, 1.1505), xytext=(15, 15),
                 textcoords="offset points", fontsize=7, color="#2e7d32")
    ax1.annotate("catastrophic", xy=(0.999, 1.3475), xytext=(-55, -15),
                 textcoords="offset points", fontsize=7, color="#c62828")
    ax1.set_xlabel("EMA Decay", fontsize=9)
    ax1.set_ylabel("BPB", fontsize=9)
    ax1.set_title("(a) EMA Decay Sensitivity", fontsize=10, fontweight="bold")

    # --- (b) TTT Grid ---
    ttt_data = np.array([
        [1.1665, 1.1650, 1.1633],
        [1.1643, 1.1634, 1.1636],
        [1.1631, 1.1624, 1.1629],
    ])
    im = ax2.imshow(ttt_data, cmap="RdYlGn_r", aspect="auto",
                    vmin=ttt_data.min() - 0.001, vmax=ttt_data.max() + 0.001)
    for i in range(3):
        for j in range(3):
            c = "white" if ttt_data[i, j] > 1.165 else "black"
            w = "bold" if (i, j) == (2, 1) else "normal"
            ax2.text(j, i, f"{ttt_data[i, j]:.4f}", ha="center", va="center",
                     fontsize=8, color=c, fontweight=w)
    ax2.set_xticks(range(3))
    ax2.set_xticklabels(["0.003", "0.005", "0.01"], fontsize=8)
    ax2.set_yticks(range(3))
    ax2.set_yticklabels(["3 ep", "5 ep", "7 ep"], fontsize=8)
    ax2.set_xlabel("Learning Rate", fontsize=9)
    ax2.set_ylabel("Epochs", fontsize=9)
    ax2.set_title("(b) TTT Fine-Tuning Grid", fontsize=10, fontweight="bold")

    # --- (c) ResFormer ---
    alpha = [0.0, 0.1, 0.5, 0.7]
    pre_q = [1.2040, 1.2020, 1.2004, 1.2025]
    ttt = [1.2584, 1.2545, 1.2536, 1.2551]

    ax3.plot(alpha, pre_q, "o-", color="#1565c0", linewidth=1.8, markersize=5, label="Pre-Q")
    ax3.plot(alpha, ttt, "s-", color="#e65100", linewidth=1.8, markersize=5, label="Post-TTT")
    ax3.scatter([0.5], [1.2004], color="#1565c0", s=80, zorder=4, marker="*")
    ax3.annotate("\u03b1=0.5", xy=(0.5, 1.2004), xytext=(10, -15),
                 textcoords="offset points", fontsize=7, color="#1565c0")
    ax3.set_xlabel("ResFormer \u03b1", fontsize=9)
    ax3.set_ylabel("BPB", fontsize=9)
    ax3.set_title("(c) ResFormer \u03b1 Sweep", fontsize=10, fontweight="bold")
    ax3.legend(fontsize=7, loc="upper right", framealpha=0.9)

    fig.tight_layout()
    out = os.path.join(OUT_DIR, "hyperparameter_sensitivity.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_architecture_compression():
    """C2: 2-panel — (a) V2 factorial heatmap, (b) GPTQ gap bars."""
    setup_academic_theme()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # --- (a) V2 Factorial ---
    data = np.array([
        [1.1641, 1.1636, 1.1665],
        [1.1666, 1.1661, 1.1700],
        [1.1636, 1.1650, 1.1686],
    ])
    over_budget = np.array([
        [False, False, True],
        [False, False, True],
        [False, False, True],
    ])
    im = ax1.imshow(data, cmap="RdYlGn_r", aspect="auto",
                    vmin=data.min() - 0.001, vmax=data.max() + 0.001)
    for i in range(3):
        for j in range(3):
            txt = f"{data[i, j]:.4f}"
            if over_budget[i, j]:
                txt += "*"
            c = "white" if data[i, j] > 1.167 else "black"
            w = "bold" if data[i, j] == data.min() else "normal"
            ax1.text(j, i, txt, ha="center", va="center", fontsize=8,
                     color=c, fontweight=w)
    for i in range(3):
        ax1.add_patch(plt.Rectangle((2 - 0.5, i - 0.5), 1, 1,
                                    fill=False, hatch="//", edgecolor="#c62828",
                                    linewidth=0, alpha=0.3))
    ax1.set_xticks(range(3))
    ax1.set_xticklabels(["No Gate", "Headwise", "Elementwise"], fontsize=8)
    ax1.set_yticks(range(3))
    ax1.set_yticklabels(["PR only", "RF only", "PR + RF"], fontsize=8)
    ax1.set_xlabel("Gate Type", fontsize=9)
    ax1.set_ylabel("Residual", fontsize=9)
    ax1.set_title("(a) V2 Factorial (BPB)", fontsize=10, fontweight="bold")
    ax1.text(0.98, -0.12, "* over 16 MB", transform=ax1.transAxes,
             fontsize=7, ha="right", color="#c62828", fontstyle="italic")

    # --- (b) GPTQ Gap ---
    methods = ["Kevin Clark", "dexhunter", "Us \u2014 baseline",
               "Us \u2014 sequential", "Us \u2014 embed GPTQ", "Us \u2014 all combined"]
    gaps = [0.012, 0.010, 0.054, 0.188, 0.487, 0.664]
    y_pos = np.arange(len(methods))
    colors = ["#2e7d32", "#2e7d32", "#1565c0", "#c62828", "#c62828", "#c62828"]

    ax2.barh(y_pos, gaps, color=colors, edgecolor="white", linewidth=0.5, height=0.6)
    ax2.axvline(x=0.012, color="#2e7d32", linestyle="--", linewidth=0.8, alpha=0.7)
    for y, gap in zip(y_pos, gaps):
        ax2.text(gap + 0.008, y, f"+{gap:.3f}", va="center", fontsize=7, color="#555555")
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(methods, fontsize=8)
    ax2.set_xlabel("GPTQ Gap (BPB)", fontsize=9)
    ax2.set_title("(b) GPTQ Compression Gap", fontsize=10, fontweight="bold")
    ax2.set_xlim(0, 0.75)

    fig.tight_layout()
    out = os.path.join(OUT_DIR, "architecture_compression.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_results_progression():
    """C3: 3-panel — (a) BPB progression, (b) artifact size, (c) technique count."""
    setup_academic_theme()
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 3.5))

    configs = ["Baseline\n(SP1024)", "V1 slim\n(SP8192)", "V2 stack\n(3-seed)",
               "V2+CaseOps\n(1-seed)", "Ext. SOTA"]
    bpb = [1.2244, 1.2073, 1.0805, 1.0621, 1.0611]
    sizes = [15.91, 15.35, 15.70, 15.98, 15.90]
    tech_counts = [3, 6, 12, 15, None]  # None for external

    # Color gradient
    norm = plt.Normalize(vmin=min(bpb), vmax=max(bpb))
    cmap = plt.cm.RdYlGn_r
    colors_bpb = [cmap(norm(b)) for b in bpb]

    y_pos = np.arange(len(configs))

    # --- (a) BPB ---
    ax1.barh(y_pos, bpb, color=colors_bpb, edgecolor="white", linewidth=0.5, height=0.6)
    for y, b in zip(y_pos, bpb):
        ax1.text(b + 0.002, y, f"{b:.4f}", va="center", fontsize=8, color="#333333")
    # Delta annotations
    for i in range(1, len(bpb)):
        delta = bpb[i] - bpb[i-1]
        ax1.annotate(f"{delta:+.4f}", xy=((bpb[i] + bpb[i-1])/2, (y_pos[i] + y_pos[i-1])/2),
                     fontsize=6, ha="center", va="center", color="#555555",
                     bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="#cccccc", lw=0.3))
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(configs, fontsize=8)
    ax1.set_xlabel("BPB (lower is better)", fontsize=9)
    ax1.set_title("(a) BPB Progression", fontsize=10, fontweight="bold")
    ax1.set_xlim(1.04, 1.26)
    ax1.invert_yaxis()

    # --- (b) Artifact Size ---
    colors_size = ["#2e7d32" if s <= 16.0 else "#c62828" for s in sizes]
    ax2.barh(y_pos[:4], sizes[:4], color=colors_size[:4], edgecolor="white",
             linewidth=0.5, height=0.6)
    ax2.axvline(x=16.0, color="#c62828", linestyle="--", linewidth=1.0, alpha=0.7)
    ax2.text(16.0, -0.5, "16 MB cap", fontsize=7, ha="center", color="#c62828")
    for y, s in zip(y_pos[:4], sizes[:4]):
        ax2.text(s + 0.02, y, f"{s:.2f}", va="center", fontsize=8, color="#333333")
    ax2.set_yticks(y_pos[:4])
    ax2.set_yticklabels([c.split("\n")[0] for c in configs[:4]], fontsize=8)
    ax2.set_xlabel("Artifact Size (MB)", fontsize=9)
    ax2.set_title("(b) Artifact Size", fontsize=10, fontweight="bold")
    ax2.set_xlim(14.5, 16.5)
    ax2.invert_yaxis()

    # --- (c) Technique Count ---
    labels_tc = ["Baseline", "V1 slim", "V2 stack", "V2+CaseOps"]
    counts = [3, 6, 12, 15]
    colors_tc = ["#ef9a9a", "#ffcc80", "#81c784", "#2e7d32"]
    ax3.barh(np.arange(4), counts, color=colors_tc, edgecolor="white",
             linewidth=0.5, height=0.6)
    for y, c in zip(range(4), counts):
        ax3.text(c + 0.2, y, str(c), va="center", fontsize=9, fontweight="bold",
                 color="#333333")
    ax3.set_yticks(range(4))
    ax3.set_yticklabels(labels_tc, fontsize=8)
    ax3.set_xlabel("Technique Count", fontsize=9)
    ax3.set_title("(c) Technique Stack", fontsize=10, fontweight="bold")
    ax3.set_xlim(0, 18)
    ax3.invert_yaxis()

    fig.tight_layout()
    out = os.path.join(OUT_DIR, "results_progression.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_negative_results_panel():
    """C4: 2-panel — (a) SLM sweep, (b) technique impact bars."""
    setup_academic_theme()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # --- (a) SLM Sweep ---
    sp8192_k = [1.0, 0.8, 0.7, 0.6]
    sp8192_bpb = [1.2411, 1.2652, 1.3183, 1.4002]
    sp1024_k = [1.0, 0.95, 0.6]
    sp1024_bpb = [1.2649, 1.2668, 1.4204]

    ax1.plot(sp8192_k, sp8192_bpb, "o-", color="#1565c0", linewidth=1.8,
             markersize=5, label="SP8192", zorder=3)
    ax1.plot(sp1024_k, sp1024_bpb, "s--", color="#e65100", linewidth=1.5,
             markersize=4, label="SP1024", zorder=3)
    ax1.axhline(y=1.2411, color="#1565c0", linestyle=":", linewidth=0.6, alpha=0.4)
    ax1.axhline(y=1.2649, color="#e65100", linestyle=":", linewidth=0.6, alpha=0.4)
    ax1.set_xlabel("Retention ratio k", fontsize=9)
    ax1.set_ylabel("BPB", fontsize=9)
    ax1.set_title("(a) SLM Sweep (monotonic degradation)", fontsize=10, fontweight="bold")
    ax1.set_xlim(0.55, 1.05)
    ax1.invert_xaxis()
    ax1.legend(fontsize=7, loc="upper left", framealpha=0.9)

    # --- (b) Technique Impact ---
    techniques = [
        ("Small Batch", -0.0153),
        ("EMA (0.990)", -0.0117),
        ("Headwise gate*", -0.0005),
        ("SLM k=0.95", +0.002),
        ("LR Warmup", +0.0024),
        ("ResFormer", +0.0025),
        ("HybridNorm", +0.011),
        ("Diff. Attn", +0.0138),
        ("Struct. FFN", +0.0425),
    ]
    peri_ln_x = 0.055
    names = [t[0] for t in techniques] + ["Peri-LN"]
    deltas = [t[1] for t in techniques] + [peri_ln_x]
    colors = ["#2e7d32" if d < 0 else "#c62828" for d in deltas]
    colors[-1] = "#888888"

    y_pos = np.arange(len(names))
    bars = ax2.barh(y_pos, deltas, color=colors, edgecolor="white", linewidth=0.5, height=0.6)
    bars[-1].set_hatch("///")
    bars[-1].set_edgecolor("#555555")
    ax2.annotate("NaN", xy=(peri_ln_x, y_pos[-1]), xytext=(3, 0),
                 textcoords="offset points", fontsize=7, va="center",
                 fontstyle="italic", color="#555555")
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(names, fontsize=8)
    ax2.invert_yaxis()
    ax2.axvline(x=0, color="#111111", linewidth=0.8)
    ax2.set_xlabel("\u0394 BPB vs. control", fontsize=9)
    ax2.set_title("(b) Technique Impact", fontsize=10, fontweight="bold")

    fig.tight_layout()
    out = os.path.join(OUT_DIR, "negative_results_panel.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_reportcard_table():
    """C5: Report card rendered as a matplotlib table image."""
    setup_academic_theme()
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    columns = ["Technique", "2\u00d7 \u0394 BPB", "8\u00d7 \u0394 BPB", "Verdict"]
    rows = [
        ["Headwise gate (novel)", "\u22120.0005", "\u22120.0005", "Helped"],
        ["Small batch", "\u22120.0153", "+0.0121", "Scale-dep."],
        ["EMA tuning (0.990)", "\u22120.0117", "+0.0025", "Scale-dep."],
        ["Diff. Attention", "+0.0138", "\u2014", "Throughput"],
        ["ResFormer \u03b1=0.5", "+0.0025", "+0.0022", "Redundant"],
        ["LR warmup", "+0.002\u20130.007", "\u2014", "Monotonic"],
        ["Structured FFN", "+0.04\u20130.05", "\u2014", "Severe"],
        ["HybridNorm", "+0.011", "\u2014", "Norm conflict"],
        ["Peri-LN", "NaN", "\u2014", "Diverged"],
        ["SLM (Rho-1)", "+0.002\u20130.156", "\u2014", "Monotonic"],
    ]

    # Cell colors
    def delta_color(val):
        if val.startswith("\u2212") or val.startswith("-"):
            return "#c8e6c9"  # green
        if val in ("\u2014", "NaN"):
            return "#eeeeee"  # gray
        return "#ffcdd2"  # red

    verdict_colors = {
        "Helped": "#c8e6c9",
        "Scale-dep.": "#fff9c4",
        "Throughput": "#ffcdd2",
        "Redundant": "#ffcdd2",
        "Monotonic": "#ffcdd2",
        "Severe": "#ffcdd2",
        "Norm conflict": "#ffcdd2",
        "Diverged": "#ffcdd2",
    }

    cell_colors = []
    for row in rows:
        row_colors = [
            "#ffffff",  # technique name
            delta_color(row[1]),
            delta_color(row[2]),
            verdict_colors.get(row[3], "#ffffff"),
        ]
        cell_colors.append(row_colors)

    table = ax.table(
        cellText=rows,
        colLabels=columns,
        cellColours=cell_colors,
        colColours=["#37474f"] * 4,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.4)

    # Style header
    for j in range(4):
        cell = table[0, j]
        cell.set_text_props(color="white", fontweight="bold", fontsize=9)
        cell.set_edgecolor("#555555")

    # Style data cells
    for i in range(1, len(rows) + 1):
        for j in range(4):
            cell = table[i, j]
            cell.set_edgecolor("#cccccc")
            if j == 0:
                cell.set_text_props(ha="left")
            # Bold the "Helped" row
            if i == 1:
                cell.set_text_props(fontweight="bold")

    ax.set_title("Technique Report Card", fontsize=12, fontweight="bold",
                 pad=20, color="#111111")

    fig.tight_layout()
    out = os.path.join(OUT_DIR, "reportcard_table.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_main_results_table():
    """C6: Main results rendered as a matplotlib table image."""
    setup_academic_theme()
    fig, ax = plt.subplots(figsize=(11, 3))
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    columns = ["Configuration", "BPB", "Artifact", "Notes"]
    rows = [
        ["Baseline (SP1024, 9L\u00d7512d, 17M)", "1.2244", "15.91 MB", "reference"],
        ["V1 slim (SP8192, 9L\u00d7448d, 17M)", "1.2073", "15.35 MB", "3-seed, std 0.0006"],
        ["V2 stack (11L\u00d7512d, 36M, PR #2005)", "1.0805", "15.70 MB", "3-seed, std 0.0012"],
        ["V2 + CaseOps + SOTA hparams", "1.0621", "15.98 MB", "seed 1337"],
        ["External SOTA (PR #1855)", "1.0611", "15.90 MB", "external ref"],
    ]

    # BPB color gradient
    bpb_vals = [float(r[1]) for r in rows]
    bpb_min, bpb_max = min(bpb_vals), max(bpb_vals)
    cmap = plt.cm.RdYlGn_r

    cell_colors = []
    for i, row in enumerate(rows):
        bpb = float(row[1])
        bpb_norm = (bpb - bpb_min) / (bpb_max - bpb_min)
        bpb_color = cmap(bpb_norm)
        # Make it lighter
        bpb_color = tuple(0.5 + 0.5 * c for c in bpb_color[:3]) + (1.0,)
        row_colors = ["#ffffff", bpb_color, "#f5f5f5", "#f5f5f5"]
        cell_colors.append(row_colors)

    # Bold best row (V2 + CaseOps, index 3)
    cell_colors[3] = ["#e8f5e9", "#c8e6c9", "#e8f5e9", "#e8f5e9"]

    table = ax.table(
        cellText=rows,
        colLabels=columns,
        cellColours=cell_colors,
        colColours=["#37474f"] * 4,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)

    # Style header
    for j in range(4):
        cell = table[0, j]
        cell.set_text_props(color="white", fontweight="bold", fontsize=9)
        cell.set_edgecolor("#555555")

    # Style data cells
    for i in range(1, len(rows) + 1):
        for j in range(4):
            cell = table[i, j]
            cell.set_edgecolor("#cccccc")
            if j == 0:
                cell.set_text_props(ha="left")
            # Bold the best row
            if i == 4:  # V2 + CaseOps (1-indexed including header)
                cell.set_text_props(fontweight="bold")

    ax.set_title("Main Results \u2014 8\u00d7H100, 10-min Wall Clock",
                 fontsize=12, fontweight="bold", pad=20, color="#111111")

    fig.tight_layout()
    out = os.path.join(OUT_DIR, "main_results_table.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved: {out}")


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    print("Generating paper plots...")

    # Existing 3
    plot_technique_impact()
    plot_slm_sweep()
    plot_transfer()

    # Part A: 3 missing figures
    plot_attention_variants()
    plot_training_curves()
    plot_paper_overview()

    # Part B: 5 supplementary
    plot_ema_sweep()
    plot_ttt_grid()
    plot_v2_factorial()
    plot_gptq_gap()
    plot_resformer_sweep()

    # Part C: 6 compact multi-panel + table renders
    plot_hyperparameter_sensitivity()
    plot_architecture_compression()
    plot_results_progression()
    plot_negative_results_panel()
    plot_reportcard_table()
    plot_main_results_table()

    print("\nAll 17 paper plots generated.")
