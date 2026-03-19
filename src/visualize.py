"""
visualize.py — Attention head visualisation for ChessTransformer.

Usage
-----
from visualize import plot_attention_heads

# Show all heads of layer 0 for a single game
fig = plot_attention_heads(model, token_ids, token_labels, layer=0)
fig.savefig("attn_layer0.png", dpi=150, bbox_inches="tight")

# Show one head per layer (averaged across heads) for all layers
fig = plot_attention_heads(model, token_ids, token_labels, layer=None)

Parameters
----------
model        : ChessTransformer in eval mode
token_ids    : (1, T) int64 tensor — a single tokenised game
token_labels : list[str] of length T — move strings for axis labels
layer        : int  → show all heads for that layer (one subplot per head)
               None → show each layer in its own row, one head per column
               (columns are capped at MAX_HEADS_SHOWN to keep the plot readable)
out_path     : optional file path to save the figure (PNG/PDF/SVG)
"""

import math
from typing import List, Optional, Union

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

MAX_HEADS_SHOWN = 8   # cap columns so the grid stays readable
MAX_LABELS      = 32  # truncate x/y tick labels beyond this many tokens


def _collect_weights(model: "ChessTransformer", token_ids: torch.Tensor):
    """
    Run one forward pass and return a list of (B, heads, T, T) tensors,
    one per transformer layer.
    """
    model.eval()
    with torch.no_grad():
        model(token_ids)
    return [block.attention.attn_weights for block in model.blocks]


def _make_tick_labels(token_labels: List[str]) -> List[str]:
    """Truncate to MAX_LABELS and shorten long individual tokens."""
    labels = token_labels[:MAX_LABELS]
    return [t[:6] for t in labels]   # keep labels short enough to fit


def _plot_layer(ax: plt.Axes, weights: torch.Tensor, title: str, tick_labels: List[str]):
    """
    Draw a single attention heatmap on `ax`.
    weights: (T_q, T_k) float tensor (already sliced to one head / averaged)
    """
    T = weights.shape[-1]
    shown = min(T, MAX_LABELS)
    data  = weights[:shown, :shown].float().cpu().numpy()

    im = ax.imshow(data, aspect="auto", cmap="Blues", vmin=0.0, vmax=1.0)
    ax.set_title(title, fontsize=8, pad=3)

    ticks = list(range(shown))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(tick_labels[:shown], rotation=90, fontsize=5)
    ax.set_yticklabels(tick_labels[:shown], fontsize=5)

    return im


def plot_attention_heads(
    model,
    token_ids:    torch.Tensor,
    token_labels: List[str],
    layer:        Optional[int] = 0,
    out_path:     Optional[str] = None,
) -> plt.Figure:
    """
    Visualise attention weights extracted from a ChessTransformer.

    Returns a matplotlib Figure.  Call fig.savefig(...) or plt.show() as needed.
    """
    all_weights = _collect_weights(model, token_ids)  # list[(B, heads, T, T)]
    tick_labels = _make_tick_labels(token_labels)

    if layer is not None:
        # ── single layer ────────────────────────────────────────────────────
        if layer >= len(all_weights):
            raise ValueError(f"layer={layer} but model only has {len(all_weights)} layers")
        weights = all_weights[layer][0]                   # (heads, T, T)
        n_heads = min(weights.shape[0], MAX_HEADS_SHOWN)

        fig, axes = plt.subplots(1, n_heads, figsize=(3 * n_heads, 3.5), constrained_layout=True)
        if n_heads == 1:
            axes = [axes]

        fig.suptitle(f"Attention Weights — Layer {layer}", fontsize=11, fontweight="bold")
        for h, ax in enumerate(axes[:n_heads]):
            _plot_layer(ax, weights[h], f"Head {h}", tick_labels)

        # Shared colour bar on the right
        fig.colorbar(
            plt.cm.ScalarMappable(cmap="Blues", norm=plt.Normalize(0, 1)),
            ax=axes, shrink=0.8, label="Attention weight",
        )

    else:
        # ── all layers ──────────────────────────────────────────────────────
        n_layers    = len(all_weights)
        n_heads_max = min(all_weights[0].shape[1], MAX_HEADS_SHOWN)

        fig, axes = plt.subplots(
            n_layers, n_heads_max,
            figsize=(2.8 * n_heads_max, 2.8 * n_layers),
            squeeze=False,
            constrained_layout=True,
        )
        fig.suptitle("Attention Weights — All Layers", fontsize=11, fontweight="bold")

        for l_idx, layer_weights in enumerate(all_weights):
            weights = layer_weights[0]                    # (heads, T, T)
            for h in range(n_heads_max):
                ax = axes[l_idx][h]
                _plot_layer(
                    ax, weights[h],
                    f"L{l_idx} H{h}" if l_idx == 0 else f"L{l_idx} H{h}",
                    tick_labels,
                )

        fig.colorbar(
            plt.cm.ScalarMappable(cmap="Blues", norm=plt.Normalize(0, 1)),
            ax=axes, shrink=0.6, label="Attention weight",
        )

    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved attention plot to {out_path}")

    return fig
