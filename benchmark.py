"""
benchmark.py — Compare all 4 attention variants side-by-side.

Produces:
  - Console table: params, val_loss, perplexity, move_legality, tokens/sec
  - plots/loss_curves.png
  - plots/benchmark.png
  - plots/benchmark_summary.json
"""

import sys
import json
import time
import argparse
from pathlib import Path

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent / "src"))

from model import ChessTransformer
from pgn_data import ChessTokenizer
from train import make_attention_factory


VARIANTS = ["vanilla", "rope", "gqa", "sparse"]

VARIANT_LABELS = {
    "vanilla": "Vanilla MHA",
    "rope": "RoPE MHA",
    "gqa": "GQA (kv=2)",
    "sparse": "Sliding Window",
}

COLORS = {
    "vanilla": "#4C72B0",
    "rope": "#DD8452",
    "gqa": "#55A868",
    "sparse": "#C44E52",
}


### Helpers

def load_metrics(checkpoint_dir: str, variant: str):
    path = Path(checkpoint_dir) / variant / "metrics.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def count_parameters(variant: str, vocab_size: int, config: dict) -> int:
    factory = make_attention_factory({**config, "variant": variant})
    model = ChessTransformer(
        vocab_size=vocab_size,
        attention_factory=factory,
        d_model=config["d_model"],
        n_heads=config["n_heads"],
        n_layers=config["n_layers"],
        use_sinusoidal_pe=(variant != "rope"),
    )
    return model.count_parameters()


def measure_throughput(
    variant: str,
    vocab_size: int,
    config: dict,
    device: torch.device,
    n_iters: int = 50,
) -> float:
    """Tokens processed per second at inference (forward pass only)."""
    factory = make_attention_factory({**config, "variant": variant})
    model = ChessTransformer(
        vocab_size=vocab_size,
        attention_factory=factory,
        d_model=config["d_model"],
        n_heads=config["n_heads"],
        n_layers=config["n_layers"],
        use_sinusoidal_pe=(variant != "rope"),
    ).to(device)
    model.eval()

    seq_len = config["seq_len"]
    batch = torch.randint(0, vocab_size, (4, seq_len), device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(5):
            model(batch)

    if device.type == "cuda":
        torch.cuda.synchronize()

    t0 = time.time()
    with torch.no_grad():
        for _ in range(n_iters):
            model(batch)

    if device.type == "cuda":
        torch.cuda.synchronize()

    elapsed = time.time() - t0
    return 4 * seq_len * n_iters / elapsed


#### Plots

def plot_loss_curves(all_metrics: dict, out_path: str):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Chess Transformer — Attention Variant Comparison", fontsize=14, fontweight="bold")

    for variant, metrics in all_metrics.items():
        if metrics is None:
            continue
        steps = [m["step"] for m in metrics]
        val_loss = [m["val_loss"] for m in metrics]
        val_ppl = [m["val_ppl"] for m in metrics]
        color = COLORS[variant]

        axes[0].plot(steps, val_loss, label=VARIANT_LABELS[variant], color=color, linewidth=2, marker="o", markersize=4)
        axes[1].plot(steps, val_ppl, label=VARIANT_LABELS[variant], color=color, linewidth=2, marker="o", markersize=4)

    axes[0].set_title("Validation Loss")
    axes[0].set_xlabel("Training Step")
    axes[0].set_ylabel("Cross-Entropy Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_title("Validation Perplexity")
    axes[1].set_xlabel("Training Step")
    axes[1].set_ylabel("Perplexity")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Loss curves saved to {out_path}")


def plot_benchmark_bars(summary: list, out_path: str):
    variants   = [s["variant"]                        for s in summary]
    colors     = [COLORS[v]                           for v in variants]
    labels     = [VARIANT_LABELS[v]                   for v in variants]
    ppls       = [s.get("best_val_ppl",   0)          for s in summary]
    legality   = [s.get("move_legality",  0) * 100    for s in summary]
    throughput = [s.get("tokens_per_sec", 0) / 1000   for s in summary]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Architecture Benchmark Summary", fontsize=14, fontweight="bold")

    # Perplexity (lower = better)
    axes[0].bar(labels, ppls, color=colors, edgecolor="white", linewidth=0.5)
    axes[0].set_title("Val Perplexity ↓")
    axes[0].set_ylabel("Perplexity")
    axes[0].tick_params(axis="x", rotation=15)
    for i, v in enumerate(ppls):
        axes[0].text(i, v * 1.01, f"{v:.1f}", ha="center", fontsize=9)

    # Move legality (higher = better)
    axes[1].bar(labels, legality, color=colors, edgecolor="white", linewidth=0.5)
    axes[1].set_title("Move Legality Rate ↑ (%)")
    axes[1].set_ylabel("% Legal Moves")
    axes[1].set_ylim(0, 110)
    axes[1].tick_params(axis="x", rotation=15)
    for i, v in enumerate(legality):
        axes[1].text(i, v + 1, f"{v:.1f}%", ha="center", fontsize=9)

    # Throughput (higher = better)
    axes[2].bar(labels, throughput, color=colors, edgecolor="white", linewidth=0.5)
    axes[2].set_title("Inference Throughput ↑ (K tok/s)")
    axes[2].set_ylabel("Thousands of tokens/sec")
    axes[2].tick_params(axis="x", rotation=15)
    for i, v in enumerate(throughput):
        axes[2].text(i, v * 1.01, f"{v:.1f}K", ha="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Benchmark bars saved to {out_path}")




### Summary table

def print_summary_table(summary: list):
    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS")
    print("=" * 80)
    print(f"{'Variant':<16} {'Params':>8} {'Val Loss':>10} {'PPL':>8} {'Legality':>10} {'Tok/s':>10}")
    print("-" * 80)
    for s in summary:
        print(
            f"{VARIANT_LABELS[s['variant']]:<16} "
            f"{s.get('params',         0):>8,} "
            f"{s.get('best_val_loss',  0):>10.4f} "
            f"{s.get('best_val_ppl',   0):>8.2f} "
            f"{s.get('move_legality',  0):>9.1%} "
            f"{s.get('tokens_per_sec', 0):>10,.0f}"
        )
    print("=" * 80 + "\n")


#### torch.compile() speedup benchmark

def benchmark_compile(variant: str, vocab_size: int, config: dict, device: torch.device, n_iters: int = 100) -> dict:
    """
    Measures forward-pass throughput with and without torch.compile().
    Returns {"eager_tok_per_sec": ..., "compiled_tok_per_sec": ..., "speedup": ...}.
    Only meaningful on PyTorch 2.0+ with CUDA; on MPS/CPU the gain is smaller.
    """
    if not hasattr(torch, "compile"):
        print("torch.compile() not available (requires PyTorch 2.0+) — skipping")
        return {}

    seq_len = config["seq_len"]
    batch   = torch.randint(0, vocab_size, (4, seq_len), device=device)

    def make():
        factory = make_attention_factory({**config, "variant": variant})
        return ChessTransformer(
            vocab_size=vocab_size,
            attention_factory=factory,
            d_model=config["d_model"],
            n_heads=config["n_heads"],
            n_layers=config["n_layers"],
            use_sinusoidal_pe=(variant != "rope"),
        ).to(device).eval()

    def measure(model):
        # Warmup — compile happens on the first call
        with torch.no_grad():
            for _ in range(5):
                model(batch)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        with torch.no_grad():
            for _ in range(n_iters):
                model(batch)
        if device.type == "cuda":
            torch.cuda.synchronize()
        return 4 * seq_len * n_iters / (time.time() - t0)

    eager_model = make()
    try:
        compiled_model = torch.compile(make())
    except RuntimeError as e:
        print(f"torch.compile() failed ({e}) — skipping compile benchmark")
        return {}

    eager_tps    = measure(eager_model)
    compiled_tps = measure(compiled_model)
    speedup      = compiled_tps / eager_tps

    return {
        "eager_tok_per_sec":    eager_tps,
        "compiled_tok_per_sec": compiled_tps,
        "speedup":              speedup,
    }


#### Main

def run_benchmark(checkpoint_dir: str, config: dict):
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Benchmarking on device: {device}")

    # Load tokenizer to get vocab_size (fall back to estimate if not found)
    tok_path = Path(checkpoint_dir) / "vanilla" / "tokenizer.json"
    if tok_path.exists():
        tokenizer  = ChessTokenizer.load(str(tok_path))
        vocab_size = tokenizer.vocab_size
    else:
        print("Warning: tokenizer not found, using vocab_size=2000 estimate")
        vocab_size = 2000

    all_metrics = {}
    summary     = []

    for variant in VARIANTS:
        metrics = load_metrics(checkpoint_dir, variant)
        all_metrics[variant] = metrics

        params = count_parameters(variant, vocab_size, config)
        tps    = measure_throughput(variant, vocab_size, config, device)

        entry = {
            "variant":       variant,
            "params":        params,
            "tokens_per_sec": tps,
        }

        if metrics:
            best = min(metrics, key=lambda m: m["val_loss"])
            last = metrics[-1]
            entry["best_val_loss"] = best["val_loss"]
            entry["best_val_ppl"]  = best["val_ppl"]
            # Move legality from final checkpoint — reflects end-of-training model
            entry["move_legality"] = last.get("move_legality", 0)
        else:
            print(f"  [WARNING] No metrics for '{variant}' — run train.py --variant {variant} first")
            entry["best_val_loss"] = 0
            entry["best_val_ppl"]  = 0
            entry["move_legality"] = 0

        summary.append(entry)
        print(f"  {VARIANT_LABELS[variant]}: {params:,} params | {tps:,.0f} tok/s")

    print_summary_table(summary)

    # torch.compile() speedup — measured once on the vanilla variant as representative
    print("Measuring torch.compile() speedup (vanilla variant)...")
    compile_result = benchmark_compile("vanilla", vocab_size, config, device)
    if compile_result:
        print(
            f"  Eager:    {compile_result['eager_tok_per_sec']:>10,.0f} tok/s\n"
            f"  Compiled: {compile_result['compiled_tok_per_sec']:>10,.0f} tok/s\n"
            f"  Speedup:  {compile_result['speedup']:.2f}×"
        )

    out_dir = Path("plots")
    out_dir.mkdir(exist_ok=True)

    plot_loss_curves(all_metrics,   str(out_dir / "loss_curves.png"))
    plot_benchmark_bars(summary,    str(out_dir / "benchmark.png"))

    with open(out_dir / "benchmark_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary JSON saved to plots/benchmark_summary.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark ChessTransformer attention variants")
    parser.add_argument("--checkpoint_dir", default="checkpoints")
    parser.add_argument("--d_model",     type=int,   default=128)
    parser.add_argument("--n_heads",     type=int,   default=4)
    parser.add_argument("--n_layers",    type=int,   default=4)
    parser.add_argument("--seq_len",     type=int,   default=128)
    parser.add_argument("--kv_heads",    type=int,   default=2)
    parser.add_argument("--window_size", type=int,   default=32)
    args = vars(parser.parse_args())
    checkpoint_dir = args.pop("checkpoint_dir")
    run_benchmark(checkpoint_dir, args)
