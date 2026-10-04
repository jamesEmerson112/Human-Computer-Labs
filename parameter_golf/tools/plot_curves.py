#!/usr/bin/env python3
"""Parse Parameter Golf training logs and produce learning curve plots.

Usage:
    python tools/plot_curves.py logs/run1.txt --mode single
    python tools/plot_curves.py logs/run1.txt logs/run2.txt --name "GQA" "MQA" --mode compare
    python tools/plot_curves.py logs/*.txt --mode csv
    python tools/plot_curves.py logs/*.txt --mode all
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless — works on RunPod without X11
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BASELINE_BPB = 1.2244
SOTA_BPB = 1.0810

# Dashboard-matching dark theme colors
BG = "#0d1117"
BG_AXES = "#161b22"
TEXT = "#e6edf3"
TEXT_MUTED = "#8b949e"
GRID = "#30363d"
BORDER = "#30363d"

# Palette for comparison lines (6 distinct colors from dashboard)
PALETTE = ["#58a6ff", "#3fb950", "#bc8cff", "#f0883e", "#f85149", "#d29922"]

# Reference line colors
BASELINE_COLOR = "#d29922"
SOTA_COLOR = "#f85149"

# ---------------------------------------------------------------------------
# Data structure
# ---------------------------------------------------------------------------

@dataclass
class RunData:
    name: str
    filepath: str = ""
    model_params: int = 0
    world_size: int = 0
    train_steps: list[int] = field(default_factory=list)
    train_losses: list[float] = field(default_factory=list)
    val_steps: list[int] = field(default_factory=list)
    val_losses: list[float] = field(default_factory=list)
    val_bpbs: list[float] = field(default_factory=list)
    final_val_bpb: float = 0.0
    final_val_loss: float = 0.0
    total_steps: int = 0

# ---------------------------------------------------------------------------
# Compiled regex patterns
# ---------------------------------------------------------------------------

RE_TRAIN = re.compile(
    r"step:(\d+)/(\d+)\s+train_loss:([\d.]+)\s+train_time:(\d+)ms"
)
RE_VAL = re.compile(
    r"step:(\d+)/(\d+)\s+val_loss:([\d.]+)\s+val_bpb:([\d.]+)\s+train_time:(\d+)ms"
)
RE_FINAL_EXACT = re.compile(
    r"final_int8_zlib_roundtrip_exact\s+val_loss:([\d.]+)\s+val_bpb:([\d.]+)"
)
RE_FINAL = re.compile(
    r"final_int8_zlib_roundtrip\s+val_loss:([\d.]+)\s+val_bpb:([\d.]+)"
)
RE_PARAMS = re.compile(r"model_params:(\d+)")
RE_WORLD = re.compile(r"world_size:(\d+)")

# Noise prefixes to skip before even trying regex
NOISE_PREFIXES = ("[rank", "[W", "[E", "W0", "E0", "NCCL", "Traceback", "[2026")

# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def parse_log(filepath: str, name: str = "") -> RunData:
    """Parse a Parameter Golf training log file into a RunData object."""
    p = Path(filepath)
    if not name:
        # For train.log files, use parent directory name
        if p.name in ("train.log", "training.log"):
            name = p.parent.name
        else:
            name = p.stem

    run = RunData(name=name, filepath=str(p))

    with open(p, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(NOISE_PREFIXES):
                continue

            # Try val line first (it also contains "step:" but has val_bpb)
            m = RE_VAL.match(line)
            if m:
                step, total, val_loss, val_bpb = int(m[1]), int(m[2]), float(m[3]), float(m[4])
                run.val_steps.append(step)
                run.val_losses.append(val_loss)
                run.val_bpbs.append(val_bpb)
                run.total_steps = total
                continue

            m = RE_TRAIN.match(line)
            if m:
                step, total, train_loss = int(m[1]), int(m[2]), float(m[3])
                run.train_steps.append(step)
                run.train_losses.append(train_loss)
                run.total_steps = total
                continue

            m = RE_FINAL_EXACT.match(line)
            if m:
                run.final_val_loss = float(m[1])
                run.final_val_bpb = float(m[2])
                continue

            # Fallback to rounded final if exact not found
            if run.final_val_bpb == 0.0:
                m = RE_FINAL.match(line)
                if m:
                    run.final_val_loss = float(m[1])
                    run.final_val_bpb = float(m[2])
                    continue

            m = RE_PARAMS.match(line)
            if m:
                run.model_params = int(m[1])
                continue

            m = RE_WORLD.match(line)
            if m:
                run.world_size = int(m[1])

    return run

# ---------------------------------------------------------------------------
# Dark theme setup
# ---------------------------------------------------------------------------

def setup_dark_theme():
    """Configure matplotlib for dark theme matching docs/dashboard.html."""
    plt.rcParams.update({
        "figure.facecolor": BG,
        "axes.facecolor": BG_AXES,
        "axes.edgecolor": BORDER,
        "axes.labelcolor": TEXT,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.alpha": 0.5,
        "text.color": TEXT,
        "xtick.color": TEXT_MUTED,
        "ytick.color": TEXT_MUTED,
        "legend.facecolor": BG_AXES,
        "legend.edgecolor": BORDER,
        "legend.labelcolor": TEXT,
        "font.size": 11,
        "figure.dpi": 150,
        "savefig.facecolor": BG,
        "savefig.edgecolor": BG,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.2,
    })

# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_single(run: RunData, out_dir: str):
    """Two stacked subplots: train_loss and val_bpb vs step."""
    if not run.train_steps and not run.val_steps:
        print(f"  SKIP {run.name}: no metric data found", file=sys.stderr)
        return

    setup_dark_theme()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    fig.suptitle(f"{run.name}", fontsize=14, fontweight="bold")

    # Top: train loss
    if run.train_steps:
        ax1.plot(run.train_steps, run.train_losses, color=PALETTE[0], linewidth=1.2, label="train_loss")
        ax1.set_ylabel("Train Loss (CE)")
        ax1.legend(loc="upper right", fontsize=9)

    # Bottom: val BPB
    if run.val_steps:
        ax2.plot(run.val_steps, run.val_bpbs, color=PALETTE[1], linewidth=1.5, label="val_bpb")
        ax2.axhline(y=BASELINE_BPB, color=BASELINE_COLOR, linestyle="--", linewidth=1, alpha=0.8, label=f"PG Baseline ({BASELINE_BPB})")
        ax2.axhline(y=SOTA_BPB, color=SOTA_COLOR, linestyle="--", linewidth=1, alpha=0.8, label=f"PG SOTA ({SOTA_BPB})")

        # Annotate final BPB
        final_bpb = run.final_val_bpb or run.val_bpbs[-1]
        final_step = run.val_steps[-1]
        ax2.annotate(
            f"{final_bpb:.4f}",
            xy=(final_step, final_bpb), xytext=(10, 10),
            textcoords="offset points", fontsize=9, color=PALETTE[1],
            arrowprops=dict(arrowstyle="->", color=PALETTE[1], lw=0.8),
        )

        ax2.set_ylabel("Val BPB")
        ax2.legend(loc="upper right", fontsize=9)

    ax2.set_xlabel("Training Step")
    fig.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{run.name}_curve.png")
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_comparison(runs: list[RunData], out_dir: str):
    """Overlay val_bpb curves from multiple runs on one chart."""
    valid = [r for r in runs if r.val_steps]
    if not valid:
        print("  SKIP comparison: no runs have val_bpb data", file=sys.stderr)
        return

    setup_dark_theme()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title("Parameter Golf — Val BPB Comparison", fontsize=14, fontweight="bold")

    for i, run in enumerate(valid):
        color = PALETTE[i % len(PALETTE)]
        ax.plot(run.val_steps, run.val_bpbs, color=color, linewidth=1.5, label=run.name)

        # Annotate final point
        final_bpb = run.final_val_bpb or run.val_bpbs[-1]
        ax.annotate(
            f"{final_bpb:.4f}",
            xy=(run.val_steps[-1], run.val_bpbs[-1]),
            xytext=(8, 5 + (i * 12)),  # offset to reduce overlap
            textcoords="offset points", fontsize=8, color=color,
        )

    # Reference lines
    ax.axhline(y=BASELINE_BPB, color=BASELINE_COLOR, linestyle="--", linewidth=1, alpha=0.8, label=f"PG Baseline ({BASELINE_BPB})")
    ax.axhline(y=SOTA_BPB, color=SOTA_COLOR, linestyle="--", linewidth=1, alpha=0.8, label=f"PG SOTA ({SOTA_BPB})")

    ax.set_xlabel("Training Step")
    ax.set_ylabel("Val BPB (lower is better)")
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "comparison.png")
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved: {out_path}")

# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

def export_csv(runs: list[RunData], out_dir: str):
    """Export all runs to a long-format CSV."""
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "metrics.csv")

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["run_name", "metric_type", "step", "value"])

        for run in runs:
            for step, loss in zip(run.train_steps, run.train_losses):
                writer.writerow([run.name, "train_loss", step, f"{loss:.6f}"])
            for step, loss, bpb in zip(run.val_steps, run.val_losses, run.val_bpbs):
                writer.writerow([run.name, "val_loss", step, f"{loss:.6f}"])
                writer.writerow([run.name, "val_bpb", step, f"{bpb:.6f}"])
            if run.final_val_bpb > 0:
                writer.writerow([run.name, "final_bpb", run.val_steps[-1] if run.val_steps else 0, f"{run.final_val_bpb:.8f}"])

    print(f"  Saved: {out_path}")

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Parse Parameter Golf logs and produce learning curve plots.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("logs", nargs="+", help="Path(s) to log files")
    parser.add_argument("--mode", choices=["single", "compare", "csv", "all"], default="all",
                        help="Output mode (default: all)")
    parser.add_argument("--out", default="docs/plots", help="Output directory (default: docs/plots)")
    parser.add_argument("--name", nargs="*", help="Display names for each log (order matches logs)")
    args = parser.parse_args()

    # Parse all logs
    runs = []
    for i, log_path in enumerate(args.logs):
        if not os.path.isfile(log_path):
            print(f"  WARNING: {log_path} not found, skipping", file=sys.stderr)
            continue
        name = args.name[i] if args.name and i < len(args.name) else ""
        run = parse_log(log_path, name)
        runs.append(run)
        n_train = len(run.train_steps)
        n_val = len(run.val_steps)
        final = f", final_bpb={run.final_val_bpb:.4f}" if run.final_val_bpb else ""
        print(f"  Parsed: {run.name} ({n_train} train, {n_val} val points{final})")

    if not runs:
        print("ERROR: No valid log files found.", file=sys.stderr)
        sys.exit(1)

    # Dispatch
    if args.mode in ("single", "all"):
        for run in runs:
            plot_single(run, args.out)

    if args.mode in ("compare", "all") and len(runs) > 1:
        plot_comparison(runs, args.out)

    if args.mode in ("csv", "all"):
        export_csv(runs, args.out)

    if args.mode == "compare" and len(runs) == 1:
        print("  NOTE: Only 1 run provided — comparison needs 2+ runs. Use --mode single.", file=sys.stderr)


if __name__ == "__main__":
    main()
