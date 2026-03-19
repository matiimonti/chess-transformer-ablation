"""
model.py — Full Transformer (decoder-only, GPT-style) from scratch.

Building blocks:
  - Sinusoidal Positional Encoding
  - Transformer Block (Attention + FFN + LayerNorm + residuals)
  - Full ChessTransformer model with configurable attention variant

All architectural choices are explicit and documented.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, List, Optional, Tuple

from attention import causal_mask

## CONSTANTS
DEFAULT_FFN_EXPANSION = 4  # FFN hidden dim = d_model * this (standard GPT-style)
DEFAULT_MAX_SEQ_LEN = 256  # default maximum sequence length for ChessTransformer
EMBEDDING_INIT_STD = 0.02  # GPT-2 style embedding weight initialisation std

##############################
#### Feed-Forward Network ####
##############################

class FeedForward(nn.Module):
    """
    Two linear layers with GELU activation
    """
    def __init__(self, d_model: int, expansion: int = DEFAULT_FFN_EXPANSION, dropout: float = 0.1):
        super().__init__()
        d_ff = d_model * expansion
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model, bias=False),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


###########################
#### Transformer Block ####
###########################

class TransformerBlock(nn.Module):
    """
    Pre-LN transformer block: normalize before attention and FFN.
    """
    def __init__(self, d_model: int, attention: nn.Module, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.attention = attention
        self.ffn = FeedForward(d_model, dropout=dropout)

    def forward(self, x: torch.Tensor, mask=None, past_kv=None, use_cache: bool = False):
        # Pre-LN attention with residual — unpack (output, present_kv) tuple
        attn_out, present_kv = self.attention(self.ln1(x), mask=mask, past_kv=past_kv, use_cache=use_cache)
        x = x + attn_out
        # Pre-LN FFN with residual
        x = x + self.ffn(self.ln2(x))
        return x, present_kv


#############################
#### Positional Encoding ####
#############################

class SinusoidalPositionalEncoding(nn.Module):
    """
    Fixed (non-learned) positional encoding — adds a unique sine/cosine pattern
    to each position so the model knows where each token is.
    """
    def __init__(self, d_model: int, max_seq_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_seq_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        # x: (B, T, d_model)
        # offset: number of already-cached tokens, so decode steps use the
        # correct absolute position rather than always starting from 0.
        x = x + self.pe[:, offset : offset + x.size(1)]
        return self.dropout(x)



########################################
#### Full Model: Chess Transformeer ####
########################################

class ChessTransformer(nn.Module):
    """
    Decoder-only transformer for chess move sequence modelling.
    """
    def __init__(
            self, 
            vocab_size: int,
            attention_factory: Callable[[], nn.Module],
            d_model: int = 128,
            n_heads: int = 4,
            n_layers: int = 4,
            max_seq_len: int = DEFAULT_MAX_SEQ_LEN,
            dropout: float = 0.1,
            use_sinusoidal_pe: bool = True
    ):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.use_sinusoidal_pe = use_sinusoidal_pe

        # Token embedding
        self.token_emb = nn.Embedding(vocab_size, d_model)
        nn.init.normal_(self.token_emb.weight, std=EMBEDDING_INIT_STD)

        # Positional encoding
        # RoPE variants encode position inside attention — use plain dropout instead
        if use_sinusoidal_pe:
            self.pos_enc = SinusoidalPositionalEncoding(d_model, max_seq_len, dropout)
        else:
            self.emb_dropout = nn.Dropout(dropout)

        # Stack of transformer blocks
        # attention_factory() is called once per layer — each block gets its own instance
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, attention_factory(), dropout)
            for _ in range(n_layers)
        ])

        # Final layer norm + projection to vocab
        self.ln_final = nn.LayerNorm(d_model)
        self.head     = nn.Linear(d_model, vocab_size, bias=False)
 
        # Weight tying: share token embedding and output projection weights
        # Reduces parameters and typically improves perplexity
        self.head.weight = self.token_emb.weight
 
    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
    ):
        B, T = idx.shape
        assert T <= self.max_seq_len, f"Sequence length {T} exceeds max {self.max_seq_len}"

        x = self.token_emb(idx)

        # During decode (past_key_values provided), all cached tokens are already
        # in the past — no causal mask needed for the new query token(s).
        past_seq_len = past_key_values[0][0].size(2) if past_key_values is not None else 0
        mask = None if past_key_values is not None else causal_mask(T, idx.device)

        # Sinusoidal PE uses offset so decode steps receive the correct absolute
        # position embedding rather than always starting from position 0.
        if self.use_sinusoidal_pe:
            x = self.pos_enc(x, offset=past_seq_len)
        else:
            x = self.emb_dropout(x)

        present_key_values: List[Optional[Tuple[torch.Tensor, torch.Tensor]]] = []
        for i, block in enumerate(self.blocks):
            past_kv = past_key_values[i] if past_key_values is not None else None
            x, present_kv = block(x, mask=mask, past_kv=past_kv, use_cache=use_cache)
            present_key_values.append(present_kv)

        x      = self.ln_final(x)
        logits = self.head(x)  # (B, T, vocab_size)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
            )

        return logits, loss, (present_key_values if use_cache else None)
 
    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        use_cache: bool = True,
    ) -> torch.Tensor:
        """
        Autoregressive generation with temperature scaling and top-k sampling.

        use_cache=True (default):
            Prefill processes the full prompt once and builds a KV cache.
            Each decode step then processes only the single new token —
            O(T) compute per step instead of O(T^2).

        use_cache=False:
            Re-processes the full growing sequence every step (original behaviour).

        temperature < 1.0 -> sharper, more confident outputs
        temperature > 1.0 -> flatter, more random outputs
        top_k             -> only sample from the top-k most likely tokens
        """
        was_training = self.training
        self.eval()
        try:
            past_key_values = None

            for _ in range(max_new_tokens):
                if use_cache:
                    if past_key_values is None:
                        # Prefill: process full prompt, populate the cache
                        idx_cond = idx[:, -self.max_seq_len :]
                        logits, _, past_key_values = self(idx_cond, use_cache=True)
                    else:
                        # Decode: one new token, reuse cached K/V
                        logits, _, past_key_values = self(
                            idx[:, -1:],
                            past_key_values=past_key_values,
                            use_cache=True,
                        )
                else:
                    idx_cond = idx[:, -self.max_seq_len :]
                    logits, _, _ = self(idx_cond)

                logits = logits[:, -1, :] / temperature

                if top_k is not None:
                    values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < values[:, [-1]]] = float("-inf")

                probs      = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                idx        = torch.cat([idx, next_token], dim=1)
        finally:
            if was_training:
                self.train()

        return idx
 
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
 

