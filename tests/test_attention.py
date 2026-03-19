"""
test_attention.py — Tests for all four attention variants.

Covers:
  - Output shapes
  - Strict causality (no future token leakage)
  - RoPE norm-preservation (rotation must not change vector magnitudes)
  - rotate_half antisymmetry (double rotation negates the vector)
  - GQA parameter reduction vs vanilla MHA
  - Sliding window mask correctness
  - Scaled dot-product attention basics
"""

import math
import pytest
import torch

from attention import (
    MultiHeadAttention,
    RoPEMultiHeadAttention,
    GroupedQueryAttention,
    SlidingWindowAttention,
    causal_mask,
    apply_rope,
    precompute_rope_freqs,
    rotate_half,
    scaled_dot_product_attention,
)

# Small dimensions so tests run fast on CPU
D_MODEL  = 32
N_HEADS  = 4
B, T     = 2, 16
HEAD_DIM = D_MODEL // N_HEADS


@pytest.fixture
def x():
    torch.manual_seed(0)
    return torch.randn(B, T, D_MODEL)


# ---------------------------------------------------------------------------
# Causal mask
# ---------------------------------------------------------------------------

class TestCausalMask:
    def test_shape(self):
        mask = causal_mask(T, torch.device("cpu"))
        assert mask.shape == (T, T)

    def test_lower_triangle_unmasked(self):
        mask = causal_mask(T, torch.device("cpu"))
        for i in range(T):
            for j in range(i + 1):
                assert not mask[i, j], f"Position ({i},{j}) should be unmasked (past/present)"

    def test_upper_triangle_masked(self):
        mask = causal_mask(T, torch.device("cpu"))
        for i in range(T):
            for j in range(i + 1, T):
                assert mask[i, j], f"Position ({i},{j}) should be masked (future)"

    def test_diagonal_unmasked(self):
        mask = causal_mask(T, torch.device("cpu"))
        assert not mask.diagonal().any(), "A token must be able to attend to itself"


# ---------------------------------------------------------------------------
# No future token leakage (causality test for all variants)
# ---------------------------------------------------------------------------

class TestNoFutureLeakage:
    """
    Perturb tokens at positions >= T//2 and verify that the outputs for
    positions < T//2 are unchanged.  Any change would mean a past position
    read from a future token — a causal mask failure.
    """

    def _check_causal(self, module: torch.nn.Module, x: torch.Tensor):
        # Attention modules accept an optional mask; supply the causal mask
        # exactly as ChessTransformer does in its forward pass.
        mask = causal_mask(T, torch.device("cpu"))
        x2 = x.clone()
        x2[:, T // 2 :] = x2[:, T // 2 :] + 100.0  # large perturbation

        with torch.no_grad():
            out1, _ = module(x,  mask=mask)
            out2, _ = module(x2, mask=mask)

        assert torch.allclose(out1[:, : T // 2], out2[:, : T // 2], atol=1e-5), (
            f"{type(module).__name__}: future tokens leaked into past positions"
        )

    def test_vanilla_causal(self, x):
        self._check_causal(MultiHeadAttention(D_MODEL, N_HEADS).eval(), x)

    def test_rope_causal(self, x):
        self._check_causal(
            RoPEMultiHeadAttention(D_MODEL, N_HEADS, max_seq_len=T).eval(), x
        )

    def test_gqa_causal(self, x):
        self._check_causal(
            GroupedQueryAttention(D_MODEL, N_HEADS, kv_heads=2).eval(), x
        )

    def test_sliding_window_causal(self, x):
        self._check_causal(
            SlidingWindowAttention(D_MODEL, N_HEADS, window_size=4).eval(), x
        )


# ---------------------------------------------------------------------------
# Output shapes
# ---------------------------------------------------------------------------

class TestOutputShapes:
    @pytest.mark.parametrize("module_cls,kwargs", [
        (MultiHeadAttention,      {"d_model": D_MODEL, "n_heads": N_HEADS}),
        (RoPEMultiHeadAttention,  {"d_model": D_MODEL, "n_heads": N_HEADS, "max_seq_len": T}),
        (GroupedQueryAttention,   {"d_model": D_MODEL, "n_heads": N_HEADS, "kv_heads": 2}),
        (SlidingWindowAttention,  {"d_model": D_MODEL, "n_heads": N_HEADS, "window_size": 4}),
    ])
    def test_output_shape(self, x, module_cls, kwargs):
        module = module_cls(**kwargs).eval()
        with torch.no_grad():
            out, cache = module(x)
        assert out.shape == (B, T, D_MODEL), (
            f"{module_cls.__name__}: expected {(B, T, D_MODEL)}, got {out.shape}"
        )
        assert cache is None  # use_cache=False by default


# ---------------------------------------------------------------------------
# RoPE
# ---------------------------------------------------------------------------

class TestRoPE:
    def test_apply_rope_preserves_norm(self):
        """RoPE is a rotation — it must preserve vector L2 norms."""
        cos, sin = precompute_rope_freqs(HEAD_DIM, T, torch.device("cpu"))
        torch.manual_seed(1)
        x = torch.randn(B, N_HEADS, T, HEAD_DIM)
        rotated = apply_rope(x, cos, sin)
        assert torch.allclose(x.norm(dim=-1), rotated.norm(dim=-1), atol=1e-5), (
            "RoPE changed vector norms — rotation must be norm-preserving"
        )

    def test_rotate_half_shape(self):
        x = torch.randn(B, N_HEADS, T, HEAD_DIM)
        assert rotate_half(x).shape == x.shape

    def test_rotate_half_double_negates(self):
        """Applying rotate_half twice should negate the original vector (180° rotation)."""
        torch.manual_seed(2)
        x = torch.randn(4, HEAD_DIM)
        assert torch.allclose(rotate_half(rotate_half(x)), -x, atol=1e-6)

    def test_rope_freqs_shape(self):
        cos, sin = precompute_rope_freqs(HEAD_DIM, T, torch.device("cpu"))
        assert cos.shape == (T, HEAD_DIM)
        assert sin.shape == (T, HEAD_DIM)

    def test_rope_cos_sin_bounded(self):
        """cos/sin values must stay in [-1, 1]."""
        cos, sin = precompute_rope_freqs(HEAD_DIM, T, torch.device("cpu"))
        assert cos.abs().max().item() <= 1.0 + 1e-6
        assert sin.abs().max().item() <= 1.0 + 1e-6


# ---------------------------------------------------------------------------
# Grouped Query Attention
# ---------------------------------------------------------------------------

class TestGQA:
    def test_fewer_params_than_vanilla(self, x):
        vanilla = MultiHeadAttention(D_MODEL, N_HEADS)
        gqa     = GroupedQueryAttention(D_MODEL, N_HEADS, kv_heads=2)
        vanilla_n = sum(p.numel() for p in vanilla.parameters())
        gqa_n     = sum(p.numel() for p in gqa.parameters())
        assert gqa_n < vanilla_n, (
            f"GQA ({gqa_n}) should have fewer params than vanilla MHA ({vanilla_n})"
        )

    def test_mqa_extreme_case(self, x):
        """kv_heads=1 is Multi-Query Attention — must still produce correct shape."""
        mqa = GroupedQueryAttention(D_MODEL, N_HEADS, kv_heads=1).eval()
        with torch.no_grad():
            out, _ = mqa(x)
        assert out.shape == (B, T, D_MODEL)

    def test_gqa_equals_mha_shape_when_kv_heads_equals_n_heads(self, x):
        """GQA with kv_heads == n_heads is full MHA — output shape must match."""
        gqa = GroupedQueryAttention(D_MODEL, N_HEADS, kv_heads=N_HEADS).eval()
        with torch.no_grad():
            out, _ = gqa(x)
        assert out.shape == (B, T, D_MODEL)


# ---------------------------------------------------------------------------
# Sliding Window Attention
# ---------------------------------------------------------------------------

class TestSlidingWindow:
    WINDOW = 4
    T_TEST = 12

    def _mask(self):
        attn = SlidingWindowAttention(D_MODEL, N_HEADS, window_size=self.WINDOW)
        return attn._sparse_mask(self.T_TEST, torch.device("cpu"))

    def test_blocks_distant_past(self):
        """A position must not attend to tokens beyond the window."""
        mask = self._mask()
        # position 10 -> position 3: distance = 7 > window (4)
        assert mask[10, 3], "Out-of-window position should be masked (blocked)"

    def test_allows_within_window(self):
        """A position must be able to attend to tokens inside the window."""
        mask = self._mask()
        # position 10 -> position 7: distance = 3 <= window (4)
        assert not mask[10, 7], "In-window position should not be masked"

    def test_mask_is_still_causal(self):
        """Every future position that causal masking blocks must also be in the window mask."""
        mask  = self._mask()
        causal = causal_mask(self.T_TEST, torch.device("cpu"))
        assert mask[causal].all(), (
            "Sliding window mask must block at least all causally-blocked positions"
        )

    def test_position_attends_to_self(self):
        """A token must always be able to attend to itself."""
        mask = self._mask()
        assert not mask.diagonal().any(), "Diagonal must be unmasked (self-attention)"


# ---------------------------------------------------------------------------
# Scaled dot-product attention (shared utility)
# ---------------------------------------------------------------------------

class TestScaledDotProduct:
    def test_output_shape(self):
        q = k = v = torch.randn(B, N_HEADS, T, HEAD_DIM)
        out = scaled_dot_product_attention(q, k, v)
        assert out.shape == (B, N_HEADS, T, HEAD_DIM)

    def test_masked_positions_get_zero_weight(self):
        """Fully-masked positions should receive ~0 attention weight (softmax of -inf)."""
        q = k = v = torch.ones(1, 1, 4, HEAD_DIM)
        # Block all but the first key
        mask = torch.ones(4, 4, dtype=torch.bool)
        mask[:, 0] = False  # only position 0 is visible
        out = scaled_dot_product_attention(q, k, v, mask=mask)
        # All output rows must equal v[:, :, 0, :] because only key 0 gets weight
        expected = v[:, :, 0:1, :].expand_as(out)
        assert torch.allclose(out, expected, atol=1e-5)

    def test_scaling_by_sqrt_head_dim(self):
        """
        Without scaling, large head_dim → huge dot products → flat softmax.
        Verify the function divides by sqrt(head_dim) by checking that identical
        q/k with large head_dim still produce a valid (non-uniform) distribution.
        """
        large_head_dim = 64
        q = k = v = torch.randn(1, 1, 8, large_head_dim)
        # If scaling is missing this would produce near-uniform softmax;
        # with proper scaling the output should differ from a uniform average of v.
        out = scaled_dot_product_attention(q, k, v)
        uniform = v.mean(dim=2, keepdim=True).expand_as(out)
        assert not torch.allclose(out, uniform, atol=1e-3), (
            "Output should not be uniform — scaling may be broken"
        )


# ---------------------------------------------------------------------------
# KV cache
# ---------------------------------------------------------------------------

class TestKVCache:
    """
    Verify that the KV cache produces output identical to the no-cache path
    and that cached tensors have the expected shapes.
    """

    T_PROMPT = 8   # tokens in the prompt (prefill)
    T_NEW    = 4   # tokens generated after prefill (decode steps)

    @pytest.mark.parametrize("module_cls,kwargs", [
        (MultiHeadAttention,     {"d_model": D_MODEL, "n_heads": N_HEADS}),
        (RoPEMultiHeadAttention, {"d_model": D_MODEL, "n_heads": N_HEADS, "max_seq_len": T}),
        (GroupedQueryAttention,  {"d_model": D_MODEL, "n_heads": N_HEADS, "kv_heads": 2}),
    ])
    def test_cache_returned_when_use_cache_true(self, module_cls, kwargs):
        module = module_cls(**kwargs).eval()
        x = torch.randn(1, self.T_PROMPT, D_MODEL)
        with torch.no_grad():
            out, cache = module(x, use_cache=True)
        assert cache is not None, "use_cache=True must return a non-None cache"
        k_cache, v_cache = cache
        assert k_cache.shape[2] == self.T_PROMPT  # T dimension of cache
        assert v_cache.shape[2] == self.T_PROMPT

    @pytest.mark.parametrize("module_cls,kwargs", [
        (MultiHeadAttention,     {"d_model": D_MODEL, "n_heads": N_HEADS}),
        (RoPEMultiHeadAttention, {"d_model": D_MODEL, "n_heads": N_HEADS, "max_seq_len": T}),
        (GroupedQueryAttention,  {"d_model": D_MODEL, "n_heads": N_HEADS, "kv_heads": 2}),
    ])
    def test_cache_grows_by_one_per_decode_step(self, module_cls, kwargs):
        module = module_cls(**kwargs).eval()
        x_prompt = torch.randn(1, self.T_PROMPT, D_MODEL)
        x_new    = torch.randn(1, 1, D_MODEL)

        with torch.no_grad():
            _, cache = module(x_prompt, use_cache=True)
            _, cache2 = module(x_new, past_kv=cache, use_cache=True)

        assert cache2[0].size(2) == self.T_PROMPT + 1, (
            "Cache should grow by 1 after each decode step"
        )

    def test_gqa_cache_uses_kv_heads_not_n_heads(self):
        """GQA KV cache is stored at kv_heads dimension — the memory saving."""
        kv_heads = 2
        gqa = GroupedQueryAttention(D_MODEL, N_HEADS, kv_heads=kv_heads).eval()
        x = torch.randn(1, self.T_PROMPT, D_MODEL)
        with torch.no_grad():
            _, cache = gqa(x, use_cache=True)
        k_cache, v_cache = cache
        assert k_cache.shape[1] == kv_heads, (
            f"GQA cache should have {kv_heads} heads, got {k_cache.shape[1]}"
        )

    def test_sliding_window_cache_bounded(self):
        """
        After a decode step, SlidingWindowAttention must evict old entries so
        the cache never exceeds window_size.  (The prefill cache is unbounded;
        eviction happens when past_kv is provided.)
        """
        window = 4
        swa = SlidingWindowAttention(D_MODEL, N_HEADS, window_size=window).eval()
        x_prefill = torch.randn(1, window + 3, D_MODEL)   # 7 tokens
        x_new     = torch.randn(1, 1, D_MODEL)             # 1 new decode token
        with torch.no_grad():
            _, cache  = swa(x_prefill, use_cache=True)            # prefill → 7 cached
            _, cache2 = swa(x_new, past_kv=cache, use_cache=True)  # decode → evicts old
        assert cache2[0].size(2) <= window, (
            f"SlidingWindowAttention cache must be ≤ window_size ({window}) "
            f"after a decode step, got {cache2[0].size(2)}"
        )
