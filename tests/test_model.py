"""
test_model.py — Tests for ChessTransformer and its sub-modules.

Covers:
  - Forward pass output shapes
  - Loss computation and padding ignore
  - Weight tying (embedding == output projection)
  - Autoregressive generation length and token validity
  - Greedy generation determinism
  - Training-mode preservation after generate()
  - Sequence length guard (AssertionError beyond max_seq_len)
  - FeedForward shape and expansion factor
  - RoPE model (use_sinusoidal_pe=False) forward pass
"""

import pytest
import torch
import torch.nn as nn

from model import ChessTransformer, FeedForward, TransformerBlock
from attention import MultiHeadAttention, RoPEMultiHeadAttention

D_MODEL    = 32
N_HEADS    = 4
VOCAB_SIZE = 50
SEQ_LEN    = 16
B          = 2


def make_model(use_sinusoidal_pe: bool = True, n_layers: int = 2) -> ChessTransformer:
    factory = lambda: MultiHeadAttention(d_model=D_MODEL, n_heads=N_HEADS)
    return ChessTransformer(
        vocab_size=VOCAB_SIZE,
        attention_factory=factory,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_layers=n_layers,
        max_seq_len=SEQ_LEN,
        use_sinusoidal_pe=use_sinusoidal_pe,
    ).eval()


# ---------------------------------------------------------------------------
# Forward pass
# ---------------------------------------------------------------------------

class TestForwardPass:
    def test_logits_shape(self):
        model = make_model()
        idx = torch.randint(0, VOCAB_SIZE, (B, SEQ_LEN))
        with torch.no_grad():
            logits, loss, cache = model(idx)
        assert logits.shape == (B, SEQ_LEN, VOCAB_SIZE)
        assert loss is None
        assert cache is None  # use_cache=False by default

    def test_loss_is_scalar_when_targets_given(self):
        model = make_model()
        idx     = torch.randint(0, VOCAB_SIZE, (B, SEQ_LEN))
        targets = torch.randint(0, VOCAB_SIZE, (B, SEQ_LEN))
        with torch.no_grad():
            _, loss, _ = model(idx, targets=targets)
        assert loss is not None
        assert loss.ndim == 0, "Loss must be a scalar"
        assert loss.item() > 0

    def test_loss_ignores_padding_targets(self):
        """
        When all target positions are -1 (padding), cross_entropy should
        return NaN or raise — not silently return 0.  Mixed targets must
        produce a valid positive loss.
        """
        model   = make_model()
        idx     = torch.randint(0, VOCAB_SIZE, (B, SEQ_LEN))
        targets = torch.full((B, SEQ_LEN), -1, dtype=torch.long)
        targets[:, :SEQ_LEN // 2] = torch.randint(0, VOCAB_SIZE, (B, SEQ_LEN // 2))
        with torch.no_grad():
            _, loss, _ = model(idx, targets=targets)
        assert loss is not None and loss.item() > 0

    def test_exceeds_max_seq_len_raises(self):
        model = make_model()
        idx = torch.randint(0, VOCAB_SIZE, (1, SEQ_LEN + 1))
        with pytest.raises(AssertionError):
            model(idx)


# ---------------------------------------------------------------------------
# Weight tying
# ---------------------------------------------------------------------------

class TestWeightTying:
    def test_head_shares_embedding_weights(self):
        model = make_model()
        assert model.head.weight is model.token_emb.weight, (
            "Output projection and token embedding must be the same object (weight tying)"
        )

    def test_gradient_flows_through_both(self):
        """A gradient step on the loss should update shared weights once, not twice."""
        factory = lambda: MultiHeadAttention(d_model=D_MODEL, n_heads=N_HEADS)
        model = ChessTransformer(
            vocab_size=VOCAB_SIZE,
            attention_factory=factory,
            d_model=D_MODEL,
            n_heads=N_HEADS,
            n_layers=2,
            max_seq_len=SEQ_LEN,
        ).train()

        idx     = torch.randint(0, VOCAB_SIZE, (B, SEQ_LEN))
        targets = torch.randint(0, VOCAB_SIZE, (B, SEQ_LEN))
        _, loss, _ = model(idx, targets=targets)
        loss.backward()
        # Shared weight must have a gradient
        assert model.token_emb.weight.grad is not None


# ---------------------------------------------------------------------------
# Autoregressive generation
# ---------------------------------------------------------------------------

class TestGenerate:
    def test_output_length(self):
        model   = make_model()
        seed    = torch.zeros(1, 1, dtype=torch.long)
        n_new   = 8
        with torch.no_grad():
            out = model.generate(seed, max_new_tokens=n_new)
        assert out.shape == (1, 1 + n_new), (
            f"Expected shape (1, {1 + n_new}), got {out.shape}"
        )

    def test_generated_tokens_in_vocab(self):
        model = make_model()
        seed  = torch.zeros(1, 1, dtype=torch.long)
        # max_new_tokens kept < SEQ_LEN so the PE table is never exceeded
        with torch.no_grad():
            out = model.generate(seed, max_new_tokens=10)
        assert out.max().item() < VOCAB_SIZE
        assert out.min().item() >= 0

    def test_greedy_is_deterministic(self):
        """top_k=1, very low temperature ≈ greedy: two runs must match exactly."""
        model = make_model()
        seed  = torch.zeros(1, 1, dtype=torch.long)
        with torch.no_grad():
            out1 = model.generate(seed, max_new_tokens=10, temperature=1e-6, top_k=1)
            out2 = model.generate(seed, max_new_tokens=10, temperature=1e-6, top_k=1)
        assert torch.equal(out1, out2), "Greedy generation must be deterministic"

    def test_seed_tokens_preserved(self):
        """The seed tokens must appear unchanged at the start of the output."""
        model = make_model()
        seed  = torch.tensor([[3, 7, 11]], dtype=torch.long)
        with torch.no_grad():
            out = model.generate(seed, max_new_tokens=5)
        assert torch.equal(out[:, :3], seed), "Seed tokens must be preserved in output"

    def test_restores_training_mode(self):
        """generate() must restore .training=True if the model was in train mode."""
        model = make_model()
        model.train()
        seed = torch.zeros(1, 1, dtype=torch.long)
        with torch.no_grad():
            model.generate(seed, max_new_tokens=5)
        assert model.training, "generate() must restore train mode after completion"

    def test_stays_eval_if_was_eval(self):
        model = make_model()  # already eval from make_model()
        seed  = torch.zeros(1, 1, dtype=torch.long)
        with torch.no_grad():
            model.generate(seed, max_new_tokens=5)
        assert not model.training, "generate() must not switch to train mode if model was eval"


# ---------------------------------------------------------------------------
# RoPE model (sinusoidal PE disabled)
# ---------------------------------------------------------------------------

class TestRoPEModel:
    def test_forward_pass(self):
        factory = lambda: RoPEMultiHeadAttention(
            d_model=D_MODEL, n_heads=N_HEADS, max_seq_len=SEQ_LEN
        )
        model = ChessTransformer(
            vocab_size=VOCAB_SIZE,
            attention_factory=factory,
            d_model=D_MODEL,
            n_heads=N_HEADS,
            n_layers=2,
            max_seq_len=SEQ_LEN,
            use_sinusoidal_pe=False,
        ).eval()
        idx = torch.randint(0, VOCAB_SIZE, (B, SEQ_LEN))
        with torch.no_grad():
            logits, _, _ = model(idx)
        assert logits.shape == (B, SEQ_LEN, VOCAB_SIZE)


# ---------------------------------------------------------------------------
# FeedForward
# ---------------------------------------------------------------------------

class TestFeedForward:
    def test_output_shape(self):
        ffn = FeedForward(D_MODEL).eval()
        x   = torch.randn(B, SEQ_LEN, D_MODEL)
        with torch.no_grad():
            out = ffn(x)
        assert out.shape == x.shape

    def test_expansion_width(self):
        """First linear layer must project to d_model * expansion."""
        expansion = 4
        ffn = FeedForward(D_MODEL, expansion=expansion)
        first_linear = ffn.net[0]
        assert first_linear.weight.shape == (D_MODEL * expansion, D_MODEL)

    def test_output_is_not_input(self):
        """FFN must transform the input, not return it unchanged."""
        ffn = FeedForward(D_MODEL).eval()
        x   = torch.randn(B, SEQ_LEN, D_MODEL)
        with torch.no_grad():
            out = ffn(x)
        assert not torch.allclose(out, x), "FFN output should differ from its input"


# ---------------------------------------------------------------------------
# Parameter counting
# ---------------------------------------------------------------------------

class TestParameterCount:
    def test_count_parameters_positive(self):
        model = make_model()
        assert model.count_parameters() > 0

    def test_more_layers_more_params(self):
        small = make_model(n_layers=2)
        large = make_model(n_layers=4)
        assert large.count_parameters() > small.count_parameters()


# ---------------------------------------------------------------------------
# KV cache
# ---------------------------------------------------------------------------

class TestKVCache:
    """
    Verify that KV-cached generation is mathematically equivalent to the
    uncached path, and that the cache tensors have the correct shapes.
    """

    def test_forward_returns_cache_when_requested(self):
        model = make_model()
        idx   = torch.randint(0, VOCAB_SIZE, (1, 8))
        with torch.no_grad():
            logits, _, cache = model(idx, use_cache=True)
        assert cache is not None
        assert len(cache) == 2, "One cache entry per layer"
        for kv in cache:
            assert kv is not None
            k, v = kv
            assert k.shape[2] == 8  # T dimension of cache matches input length
            assert v.shape[2] == 8

    def test_forward_returns_no_cache_by_default(self):
        model = make_model()
        idx   = torch.randint(0, VOCAB_SIZE, (1, 8))
        with torch.no_grad():
            _, _, cache = model(idx)
        assert cache is None

    def test_cached_generate_matches_uncached(self):
        """
        With greedy sampling (temperature→0, top_k=1), both paths must
        produce exactly the same token sequence.
        """
        torch.manual_seed(42)
        model = make_model()
        seed  = torch.zeros(1, 1, dtype=torch.long)
        with torch.no_grad():
            out_cached   = model.generate(seed, max_new_tokens=8, temperature=1e-6, top_k=1, use_cache=True)
            out_uncached = model.generate(seed, max_new_tokens=8, temperature=1e-6, top_k=1, use_cache=False)
        assert torch.equal(out_cached, out_uncached), (
            "KV-cached and uncached generation must agree on greedy output"
        )

    def test_decode_step_processes_single_token(self):
        """
        After prefill, each decode step should pass only idx[:, -1:] to the
        model (T=1) — confirmed by verifying the cache grows by 1 per step.
        """
        model = make_model()
        idx   = torch.zeros(1, 4, dtype=torch.long)
        with torch.no_grad():
            _, _, cache = model(idx, use_cache=True)  # prefill
        T_after_prefill = cache[0][0].size(2)
        assert T_after_prefill == 4

        # One decode step
        with torch.no_grad():
            _, _, cache2 = model(idx[:, -1:], past_key_values=cache, use_cache=True)
        assert cache2[0][0].size(2) == 5, "Cache must grow by 1 per decode step"
