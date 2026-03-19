"""
scale.py — Scaling law mini-experiment.

Trains three model sizes (1M / 5M / 20M parameters) on the same data and
plots validation loss against cumulative training FLOPs.

Even a small Chinchilla-style curve on chess data shows whether loss follows
a power-law in compute — a concrete result that goes beyond benchmarking
individual architectures.

Usage
-----
python scale.py --pgn_path data/games.pgn [--max_games 20000] [--max_steps 3000]

Outputs
-------
checkpoints/scaling/<size>/metrics.json   — per-checkpoint metrics
plots/scaling_laws.png                    — loss vs compute curve
plots/scaling_summary.json                — best loss per model size
"""

import sys
import json
import argparse
import math
from pathlib import Path

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "src"))

from train import train


# ---------------------------------------------------------------------------
# Model size configurations
# Params ≈ vocab_size × d_model  +  12 × n_layers × d_model²
# At vocab_size ≈ 2 000:
#   small  → ~1M   medium → ~5M   large → ~20M
# ---------------------------------------------------------------------------

MODEL_SIZES = {
    "small":  {"d_model": 128, "n_heads": 4, "n_layers": 4},
    "medium": {"d_model": 256, "n_heads": 8, "n_layers": 6},
    "large":  {"d_model": 512, "n_heads": 8, "n_layers": 6},
}

SIZE_LABELS  = {"small": "~1M params", "medium": "~5M params", "large": "~20M params"}
SIZE_COLORS  = {"small": "#4C72B0",    "medium": "#DD8452",     "large": "#55A868"}


# ---------------------------------------------------------------------------
# Power-law fit:  L(C) = a * C^b
# Fit in log space: log L = log a + b * log C
# ---------------------------------------------------------------------------

def fit_power_law(flops: list, losses: list):
    """Returns (a, b) such that L ≈ a * C^b."""
    log_c = np.log(flops)
    log_l = np.log(losses)
    b, log_a = np.polyfit(log_c, log_l, 1)
    return math.exp(log_a), b


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_scaling(results: dict, out_path: str):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Scaling Laws — Chess Transformer", fontsize=13, fontweight="bold")

    for size, metrics in results.items():
        if not metrics:
            continue
        flops  = [m["cumulative_flops"] for m in metrics]
        losses = [m["val_loss"]         for m in metrics]
        color  = SIZE_COLORS[size]
        label  = SIZE_LABELS[size]

        # Left: loss vs FLOPs (log-log)
        axes[0].plot(flops, losses, marker="o", markersize=4,
                     linewidth=2, color=color, label=label)

        # Fit power law if we have enough points
        if len(flops) >= 3:
            a, b = fit_power_law(flops, losses)
            c_range = np.logspace(np.log10(min(flops)), np.log10(max(flops)), 100)
            axes[0].plot(c_range, a * c_range ** b, "--", color=color,
                         alpha=0.5, linewidth=1)

        # Right: loss vs FLOPs on linear-x for readability
        axes[1].plot(flops, losses, marker="o", markersize=4,
                     linewidth=2, color=color, label=label)

    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Cumulative Training FLOPs")
    axes[0].set_ylabel("Validation Loss")
    axes[0].set_title("Log–Log (dashed = power-law fit)")
    axes[0].legend()
    axes[0].grid(True, which="both", alpha=0.3)

    axes[1].set_xscale("log")
    axes[1].set_xlabel("Cumulative Training FLOPs")
    axes[1].set_ylabel("Validation Loss")
    axes[1].set_title("Log-X Linear-Y")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Scaling curve saved to {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_scaling(args: dict):
    results  = {}
    summary  = []

    for size, arch in MODEL_SIZES.items():
        print(f"\n{'='*60}")
        print(f"Training {size} model ({SIZE_LABELS[size]})")
        print(f"{'='*60}")

        config = {
            # architecture
            "variant":   "vanilla",   # fixed variant isolates size as the variable
            "d_model":   arch["d_model"],
            "n_heads":   arch["n_heads"],
            "n_layers":  arch["n_layers"],
            "dropout":   0.1,
            "kv_heads":  arch["n_heads"] // 2,
            "window_size": 32,
            # data
            "pgn_path":    args["pgn_path"],
            "max_games":   args.get("max_games"),
            "seq_len":     128,
            "train_split": 0.9,
            # training
            "batch_size":   args["batch_size"],
            "num_workers":  args["num_workers"],
            "max_steps":    args["max_steps"],
            "max_lr":       3e-4,
            "min_lr":       3e-5,
            "weight_decay": 0.1,
            "grad_clip":    1.0,
            "warmup_steps": max(100, args["max_steps"] // 10),
            "gradient_accumulation_steps": 1,
            # logging
            "log_interval":  args["log_interval"],
            "eval_interval": args["eval_interval"],
            "out_dir":       f"checkpoints/scaling/{size}",
            "wandb":         False,
        }

        metrics = train(config)
        results[size] = metrics

        if metrics:
            best = min(metrics, key=lambda m: m["val_loss"])
            summary.append({
                "size":             size,
                "label":            SIZE_LABELS[size],
                "best_val_loss":    best["val_loss"],
                "best_val_ppl":     best["val_ppl"],
                "best_step":        best["step"],
                "total_flops":      metrics[-1]["cumulative_flops"],
            })
            print(f"\n{size}: best val loss = {best['val_loss']:.4f} (step {best['step']})")

    # Print summary table
    print(f"\n{'='*60}")
    print("SCALING EXPERIMENT RESULTS")
    print(f"{'='*60}")
    print(f"{'Size':<10} {'Params':>12} {'Best Loss':>10} {'Best PPL':>10}")
    print("-" * 60)
    for s in summary:
        print(f"{s['size']:<10} {s['label']:>12} {s['best_val_loss']:>10.4f} {s['best_val_ppl']:>10.2f}")
    print(f"{'='*60}\n")

    Path("plots").mkdir(exist_ok=True)
    plot_scaling(results, "plots/scaling_laws.png")

    with open("plots/scaling_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("Summary saved to plots/scaling_summary.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scaling law mini-experiment")
    parser.add_argument("--pgn_path",     type=str,   default="data/games.pgn")
    parser.add_argument("--max_games",    type=int,   default=None)
    parser.add_argument("--max_steps",    type=int,   default=3000,
                        help="Steps per model size (default 3000 — enough to see scaling)")
    parser.add_argument("--batch_size",   type=int,   default=64)
    parser.add_argument("--num_workers",  type=int,   default=2)
    parser.add_argument("--log_interval", type=int,   default=100)
    parser.add_argument("--eval_interval",type=int,   default=300,
                        help="More frequent evals = more points on the scaling curve")
    args = vars(parser.parse_args())
    run_scaling(args)
