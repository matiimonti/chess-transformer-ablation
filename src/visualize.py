"""
visualize.py — Attention head visualization utilities.

Reads the attention weights cached on each attention module during the most
recent forward pass (model.blocks[i].attention.attn_weights) and renders them
as heatmaps — one subplot per head.

Usage:
    from visualize import plot_attention_heads, plot_all_layers

    # Run a forward pass (weights are cached automatically)
    with torch.no_grad():
        model(idx)

    fig = plot_attention_heads(model, tokenizer, idx, layer=0)
    fig.savefig("attn_layer0.png")
"""

from __future__ import annotations

from typing import List, Optional

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_tokens(tokenizer, idx: torch.Tensor) -> List[str]:
    """
    Decode a 1-D token id tensor into a list of string labels.
    Falls back to numeric ids if the tokenizer has no id_to_token mapping.
    """
    ids = idx.squeeze().tolist()
    if isinstance(ids, int):
        ids = [ids]
    if hasattr(tokenizer, "id_to_token"):
        return [tokenizer.id_to_token.get(i, str(i)) for i in ids]
    if hasattr(tokenizer, "decode"):
        return [tokenizer.decode([i])[0] if tokenizer.decode([i]) else str(i) for i in ids]
    return [str(i) for i in ids]


def _fetch_weights(model, layer: int) -> Optional[torch.Tensor]:
    """
    Return the cached attn_weights tensor from layer *layer*, or None.
    Shape: (B, n_heads, T, T)
    """
    try:
        weights = model.blocks[layer].attention.attn_weights
    except (AttributeError, IndexError):
        return None
    return weights


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def plot_attention_heads(
    model,
    tokenizer,
    idx: torch.Tensor,
    layer: int = 0,
    figsize_per_head: tuple = (3.5, 3.0),
    cmap: str = "Blues",
    title_prefix: str = "",
) -> plt.Figure:
    """
    Plot one heatmap per attention head for a single layer.

    Parameters
    ----------
    model       : ChessTransformer (in eval mode, forward pass already run).
    tokenizer   : ChessTokenizer — used to label axes with token strings.
    idx         : (1, T) input token ids used in the last forward pass.
    layer       : Which transformer block to visualise (0-indexed).
    figsize_per_head : (width, height) in inches for each subplot.
    cmap        : Matplotlib colormap name.
    title_prefix: Optional string prepended to the figure title.

    Returns
    -------
    matplotlib Figure.  Call fig.savefig(...) or plt.show() to display.
    """
    weights = _fetch_weights(model, layer)
    if weights is None:
        raise RuntimeError(
            f"No attention weights found for layer {layer}. "
            "Make sure the model ran a forward pass and that the attention "
            "module stores weights in self.attn_weights."
        )

    # weights: (B, n_heads, T, T) — take first item in batch
    w = weights[0].detach().cpu().float()   # (n_heads, T, T)
    n_heads, T_q, T_k = w.shape

    tokens = _get_tokens(tokenizer, idx[0])   # length T

    ncols = min(n_heads, 4)
    nrows = (n_heads + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(figsize_per_head[0] * ncols, figsize_per_head[1] * nrows),
        squeeze=False,
    )

    prefix = f"{title_prefix} | " if title_prefix else ""
    fig.suptitle(
        f"{prefix}Layer {layer} — Attention Weights ({n_heads} heads)",
        fontsize=12, fontweight="bold",
    )

    tick_labels = tokens[:T_k]

    for h in range(n_heads):
        row, col = divmod(h, ncols)
        ax = axes[row][col]
        im = ax.imshow(w[h].numpy(), aspect="auto", cmap=cmap, vmin=0.0, vmax=1.0)
        ax.set_title(f"Head {h}", fontsize=9)

        # Label axes with token strings (skip if seq is long)
        if T_k <= 32:
            ax.set_xticks(range(T_k))
            ax.set_xticklabels(tick_labels, rotation=90, fontsize=6)
            ax.set_yticks(range(T_q))
            ax.set_yticklabels(tokens[:T_q], fontsize=6)
        else:
            ax.xaxis.set_major_locator(ticker.MultipleLocator(max(1, T_k // 8)))
            ax.yaxis.set_major_locator(ticker.MultipleLocator(max(1, T_q // 8)))

        ax.set_xlabel("Key position", fontsize=7)
        ax.set_ylabel("Query position", fontsize=7)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Hide unused subplots
    for h in range(n_heads, nrows * ncols):
        row, col = divmod(h, ncols)
        axes[row][col].set_visible(False)

    plt.tight_layout()
    return fig


def plot_all_layers(
    model,
    tokenizer,
    idx: torch.Tensor,
    out_dir: Optional[str] = None,
    **kwargs,
) -> List[plt.Figure]:
    """
    Call plot_attention_heads for every transformer layer.

    Parameters
    ----------
    model, tokenizer, idx : same as plot_attention_heads.
    out_dir   : If provided, each figure is saved as
                ``<out_dir>/attn_layer<N>.png`` and then closed.
    **kwargs  : Forwarded to plot_attention_heads.

    Returns
    -------
    List of Figures (empty list if out_dir was given and figures were closed).
    """
    n_layers = len(model.blocks)
    figs = []

    for layer in range(n_layers):
        weights = _fetch_weights(model, layer)
        if weights is None:
            continue   # layer has no cached weights — skip silently

        fig = plot_attention_heads(model, tokenizer, idx, layer=layer, **kwargs)
        if out_dir is not None:
            import os
            os.makedirs(out_dir, exist_ok=True)
            path = os.path.join(out_dir, f"attn_layer{layer}.png")
            fig.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved {path}")
        else:
            figs.append(fig)

    return figs
