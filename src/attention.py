"""
attention.py — Multi-Head Attention variants from scratch.

Implements:
  - Vanilla Multi-Head Self-Attention (MHA)
  - Multi-Head Attention with Rotary Position Embeddings (RoPE)
  - Grouped Query Attention (GQA)
  - Sparse / Sliding-Window Attention

No nn.MultiheadAttention used anywhere — everything is explicit.

B — Batch size. How many sequences you're processing at once. e.g. 64 games simultaneously
T — Time steps, i.e. sequence length. How many tokens in each sequence. e.g. 128 moves
C — Channels, i.e. d_model. The embedding dimension of each token. e.g. 128

Q (Query) — what you're looking for. "What information do I need at this position?"
K (Key) — what each position is advertising. "What information do I contain?"
V (Value) — the actual content. "What information do I actually pass along if selected?"
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

    # (B, heads, T, T)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)

    if mask is not None:
        # mask = True/1 where we want to BLOCK attention
        scores = scores.masked_fill(mask == 1, float("-inf"))

    attn_weights = F.softmax(scores, dim=-1)

    if dropout > 0.0 and training:
        attn_weights = F.dropout(attn_weights, p=dropout)

    return torch.matmul(attn_weights, v)

## Causal mask
# Verify by hand that position 0 can only attend to position 0, position 1 to positions 0-1, etc
def causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """Upper-triangular mask — prevents position i from attending to j > i."""
    return torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()

######################################################
#### Variant 1: Vanilla Multi-Head Self-Attention ####
######################################################

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model  = d_model
        self.n_heads  = n_heads
        self.head_dim = d_model // n_heads
        self.dropout  = dropout

        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        B, T, C = x.shape

        # Project and split into Q, K, V
        qkv = self.qkv_proj(x)                          # (B, T, 3*d_model)
        q, k, v = qkv.split(self.d_model, dim=-1)       # each (B, T, d_model) 

        # Reshape to (B, heads, T, head_dim)
        def reshape(t):
            return t.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        q, k, v = reshape(q), reshape(k), reshape(v)  

        out = scaled_dot_product_attention(
            q, k, v, mask=mask, dropout=self.dropout, training=self.training
        )

        # Merge heads: (B, heads, T, head_dim) -> (B, T, d_model)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)


##############################
#### RoPE helper functions ###
##############################

def precompute_rope_freqs(
    head_dim: int,
    max_seq_len: int,
    device: torch.device,
    theta: float = 10000.0,
):
    """
    Precomputes cos/sin tables for RoPE.

    RoPE encodes position by *rotating* query and key vectors rather than
    adding a fixed vector. This gives better length generalisation and is
    used in LLaMA, Mistral, Gemma, and most modern open models.

    The rotation matrix for position m and dimension pair (2i, 2i+1) is:
        R(m, i) = [[cos(m * theta_i), -sin(m * theta_i)],
                   [sin(m * theta_i),  cos(m * theta_i)]]
    where theta_i = 10000^(-2i/d).

    cos/sin are duplicated across the full head_dim so they can be applied
    with a single elementwise multiply — no slicing needed at runtime.
    """
    assert head_dim % 2 == 0
    # θ_i = 1 / 10000^(2i/head_dim),  shape: (head_dim/2,)
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    # positions × frequencies,  shape: (max_seq_len, head_dim/2)
    positions = torch.arange(max_seq_len, device=device).float()
    freqs = torch.outer(positions, freqs)
    # duplicate so final shape is (max_seq_len, head_dim)
    cos = torch.cat([freqs.cos(), freqs.cos()], dim=-1)
    sin = torch.cat([freqs.sin(), freqs.sin()], dim=-1)
    return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    Rotates the second half of the last dimension into the first half.
    [-x2, x1] is the 90-degree rotation needed to implement the RoPE formula.
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    Apply rotary embeddings to a (B, heads, T, head_dim) tensor.

    x * cos + rotate_half(x) * sin expands to:
        first half:  x1*cos - x2*sin
        second half: x2*cos + x1*sin
    which is exactly the 2D rotation matrix applied to each dimension pair.
    """
    T = x.size(2)
    cos = cos[:T].unsqueeze(0).unsqueeze(0)  # (1, 1, T, head_dim)
    sin = sin[:T].unsqueeze(0).unsqueeze(0)
    return x * cos + rotate_half(x) * sin


################################################################
#### Variant 2: Multi-Head Attention with Rotary Embeddings ####
################################################################

class RoPEMultiHeadAttention(nn.Module):
    """
    MHA where position is encoded via rotation of Q and K vectors (RoPE).
    """

    def __init__(self, d_model: int, n_heads: int, max_seq_len: int = 512, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0

        self.d_model  = d_model
        self.n_heads  = n_heads
        self.head_dim = d_model // n_heads
        self.dropout  = dropout

        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        # Precompute and register as buffer — moves to GPU automatically with .to(device)
        cos, sin = precompute_rope_freqs(self.head_dim, max_seq_len, device=torch.device("cpu"))
        self.register_buffer("rope_cos", cos)
        self.register_buffer("rope_sin", sin)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        B, T, C = x.shape

        qkv = self.qkv_proj(x)
        q, k, v = qkv.split(self.d_model, dim=-1)

        def reshape(t):
            return t.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        q, k, v = reshape(q), reshape(k), reshape(v)

        # Apply rotary embeddings to Q and K only — V is not rotated
        q = apply_rope(q, self.rope_cos, self.rope_sin)
        k = apply_rope(k, self.rope_cos, self.rope_sin)

        out = scaled_dot_product_attention(
            q, k, v, mask=mask, dropout=self.dropout, training=self.training
        )

        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)


##################################################
#### Variant 3: Grouped Query Attention (GQA) ####
##################################################

class GroupedQueryAttention(nn.Module):
    """
    GQA: fewer K/V heads than Q heads, reducing KV-cache memory at inference.

    Used in: LLaMA-2 70B, Mistral 7B, Gemma, Falcon.

    If kv_heads == n_heads  -> standard MHA
    If kv_heads == 1        -> Multi-Query Attention (MQA)
    Otherwise               -> GQA

    Memory saving at inference: KV cache shrinks by factor (n_heads / kv_heads).
    Quality: minimal degradation with kv_heads >= n_heads // 4.

    Why it suits the ablation framing:
    Same model quality, measurably less memory — a concrete benchmark story.
    """

    def __init__(self, d_model: int, n_heads: int, kv_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        assert n_heads % kv_heads == 0, "n_heads must be divisible by kv_heads"

        self.d_model  = d_model
        self.n_heads  = n_heads
        self.kv_heads = kv_heads
        self.head_dim = d_model // n_heads
        self.groups   = n_heads // kv_heads  # how many Q heads share each K/V head
        self.dropout  = dropout

        self.q_proj   = nn.Linear(d_model, d_model, bias=False)
        self.k_proj   = nn.Linear(d_model, kv_heads * self.head_dim, bias=False)
        self.v_proj   = nn.Linear(d_model, kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        B, T, C = x.shape

        q = self.q_proj(x).view(B, T, self.n_heads,  self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.kv_heads, self.head_dim).transpose(1, 2)

        # Expand K and V so each Q head has a corresponding K/V head
        # (B, kv_heads, T, head_dim) -> (B, n_heads, T, head_dim)
        k = k.repeat_interleave(self.groups, dim=1)
        v = v.repeat_interleave(self.groups, dim=1)

        out = scaled_dot_product_attention(
            q, k, v, mask=mask, dropout=self.dropout, training=self.training
        )

        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)


######################################################
#### Variant 4: Sparse / Sliding-Window Attention ####
######################################################

class SlidingWindowAttention(nn.Module):
    """
    Each token attends only to the previous `window_size` tokens (local attention).

    Reduces complexity from O(T^2) to O(T * window_size).
    Used in: Longformer, BigBird, Mistral's sliding window layers.

    For chess sequences (length ~60-80), this is a mild constraint but
    illustrates the principle clearly and benchmarks well.
    """

    def __init__(self, d_model: int, n_heads: int, window_size: int = 32, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0

        self.d_model     = d_model
        self.n_heads     = n_heads
        self.head_dim    = d_model // n_heads
        self.window_size = window_size
        self.dropout     = dropout

        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj  = nn.Linear(d_model, d_model, bias=False)

    def _sparse_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Combines causal mask with local window mask.
        True = blocked. Position i can attend to positions max(0, i-window_size) ... i.

        distance[i, j] = j - i
        out_of_window: j < i - window_size  (too far in the past)
        causal:        j > i                (future — always blocked)
        Result: a causal band of width window_size.
        """
        causal = causal_mask(seq_len, device)
        positions = torch.arange(seq_len, device=device)
        distance = positions.unsqueeze(0) - positions.unsqueeze(1)  # (T, T)
        out_of_window = distance < -self.window_size
        return causal | out_of_window

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        B, T, C = x.shape

        qkv = self.qkv_proj(x)
        q, k, v = qkv.split(self.d_model, dim=-1)

        def reshape(t):
            return t.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        q, k, v = reshape(q), reshape(k), reshape(v)

        # Always use the sliding window mask — ignore any passed-in mask
        sparse_mask = self._sparse_mask(T, x.device)

        out = scaled_dot_product_attention(
            q, k, v, mask=sparse_mask, dropout=self.dropout, training=self.training
        )

        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)

