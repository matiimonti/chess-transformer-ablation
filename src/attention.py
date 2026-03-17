"""
attention.py — Multi-Head Attention variants from scratch.

Implements:
  - Vanilla Multi-Head Self-Attention (MHA)
  - Multi-Head Attention with Rotary Position Embeddings (RoPE)
  - Grouped Query Attention (GQA)
  - Sparse / Sliding-Window Attention

No nn.MultiheadAttention used anywhere — everything is explicit.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# Utility: scaled dot-product attention (shared by all variants)

def scaled_dot_product_attention(
    q: torch.Tensor,          # (B, heads, T, head_dim)
    k: torch.Tensor,          # (B, heads, T, head_dim)
    v: torch.Tensor,          # (B, heads, T, head_dim)
    mask: Optional[torch.Tensor] = None,  # (T, T) or (B, 1, T, T)
    dropout: float = 0.0,
    training: bool = False,
) -> torch.Tensor:
    """
    Core attention operation.
    Dividing by sqrt(head_dim) keeps gradients stable — without this,
    dot products grow large in magnitude, pushing softmax into flat regions.
    """
    head_dim = q.size(-1)
    scale = math.sqrt(head_dim)

    # (B, heads, T, T)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)

    if mask is not None:
        # mask = True/1 where we want to BLOCK attention
        scores = scores.masked_fill(mask == 1, float("-inf"))

    attn_weights = F.softmax(scores, dim=-1)

    if dropout > 0.0 and training:
        attn_weights = F.dropout(attn_weights, p=dropout)

    return torch.matmul(attn_weights, v)

## Casual mask
# Verify by hand that position 0 can only attend to position 0, position 1 to positions 0-1, etc
def causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """Upper-triangular mask — prevents position i from attending to j > i."""
    return torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()


