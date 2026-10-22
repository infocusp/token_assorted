"""A plain ("vanilla") transformer block family — the alternative to the vendored searchformer
blocks, selected by ``Config.transformer_impl``.

Written from scratch. What "vanilla" means here, versus the searchformer blocks it stands in for:

  | | searchformer | vanilla (this file) |
  |---|---|---|
  | norm       | ``RMSLayerNorm``                          | ``nn.LayerNorm``               |
  | positions  | RoPE, inside every attention layer        | learned absolute, added at the |
  |            |                                           | stack input by ``TokenVQVAE``  |
  | FFN        | SwiGLU, hidden hard-coded 4*d -> 2/3 -> ``multiple_of`` (1536 at d=512) |
  |            |                                           | ``Linear -> GELU -> Linear``,  |
  |            |                                           | hidden = ``ffn_mult * d`` (2048) |
  | attention  | ``F.scaled_dot_product_attention``        | same primitive                 |

Both families are pre-norm residual with the same topology, so the arms differ in *components*, not
in *structure* — that keeps the comparison a single-variable ablation.

Two things here are load-bearing and must not be "cleaned up":

1. **Submodule names mirror searchformer exactly** (``attention`` with ``wq/wk/wv/wo``,
   ``feed_forward``, ``attention_norm``, ``ffn_norm``, plus ``attention_feat`` /
   ``attention_feat_norm`` on the cross block). ``tests/test_smoke.py`` probes gradient flow through
   ``model.enc_layers[0].attention.wq.weight``; matching the names lets those tests cover both impls
   with no impl-specific accessors.
2. **``forward`` signatures match the vendored blocks** — ``blk(input)`` for self-attention and
   ``blk(input, features=..., mask_features=...)`` for the cross block. Every call site in
   ``src/model.py`` is therefore shared between the two impls.

Self-attention is **always causal** here. Stage 1 passes no attention mask on the self-attention
paths at all: pad-invariance is guaranteed structurally by right-padding plus causality (see
``tests/test_masking.py``). A non-causal variant would leak pad content into real positions and would
require introducing key-padding masks across Stage 1 — a masking-semantics change, deliberately out
of scope. Cross-attention is not causal (it attends a memory) and takes the additive mask that
``TokenVQVAE._feature_mask`` already builds.
"""
from __future__ import annotations

from typing import Optional

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class VanillaAttention(nn.Module):
    """Multi-head attention over ``F.scaled_dot_product_attention``, no biases.

    ``is_causal`` is fixed at construction, not per-call: ``True`` for the self-attention sites,
    ``False`` for cross-attention. Causality and an explicit mask are never combined — the causal
    sites pass no mask (right-padding makes one unnecessary) and the cross site passes a mask but is
    not causal.
    """

    def __init__(
        self,
        dim: int,
        n_heads: int,
        dropout: float = 0.0,
        is_causal: bool = False,
    ):
        super().__init__()
        assert dim % n_heads == 0, f"dim ({dim}) must be divisible by n_heads ({n_heads})."
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.dropout = dropout
        self.is_causal = is_causal

        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, dim, bias=False)
        self.wv = nn.Linear(dim, dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)

    def forward(
        self,
        queries: Tensor,
        keys: Tensor,
        values: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """(B, Lq, d), (B, Lk, d), (B, Lk, d) -> (B, Lq, d).

        ``mask`` is the additive form SDPA broadcasts: (B, 1, 1, Lk) with 0 for valid keys and -inf
        for masked ones (what ``TokenVQVAE._feature_mask`` produces).
        """
        assert not (self.is_causal and mask is not None), (
            "causal attention is used without an explicit mask in Stage 1; passing both is "
            "ambiguous."
        )
        B, Lq, _ = queries.shape
        Lk = keys.shape[1]

        xq = self.wq(queries).view(B, Lq, self.n_heads, self.head_dim).transpose(1, 2)
        xk = self.wk(keys).view(B, Lk, self.n_heads, self.head_dim).transpose(1, 2)
        xv = self.wv(values).view(B, Lk, self.n_heads, self.head_dim).transpose(1, 2)

        out = F.scaled_dot_product_attention(
            query=xq,
            key=xk,
            value=xv,
            attn_mask=mask,
            # unlike the vendored Attention, gate dropout on training mode
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=self.is_causal,
        )
        out = out.transpose(1, 2).contiguous().view(B, Lq, -1)
        return self.wo(out)


class VanillaFeedForward(nn.Module):
    """``Linear -> GELU -> Linear``. ``hidden_dim`` is used as given (no SwiGLU 2/3 rounding), so
    ``Config.ffn_mult`` is live on this path — unlike the searchformer arm, where it is
    informational because the vendored block hard-codes 4*d."""

    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim, bias=False)
        self.fc2 = nn.Linear(hidden_dim, dim, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, input: Tensor) -> Tensor:
        return self.fc2(self.drop(F.gelu(self.fc1(input))))


class VanillaSelfAttentionBlock(nn.Module):
    """Pre-norm causal self-attention + FFN. ``forward`` takes one argument, matching the vendored
    ``SelfAttentionBlock``."""

    def __init__(
        self,
        dim: int,
        n_heads: int,
        hidden_dim: int,
        dropout: float = 0.0,
        norm_eps: float = 1e-5,
        layer_id: int = 0,
    ):
        super().__init__()
        self.layer_id = layer_id
        self.attention = VanillaAttention(dim, n_heads, dropout=dropout, is_causal=True)
        self.feed_forward = VanillaFeedForward(dim, hidden_dim, dropout=dropout)
        self.attention_norm = nn.LayerNorm(dim, eps=norm_eps)
        self.ffn_norm = nn.LayerNorm(dim, eps=norm_eps)

    def forward(self, input: Tensor) -> Tensor:
        input_norm = self.attention_norm(input)
        h = input + self.attention(input_norm, input_norm, input_norm)
        return h + self.feed_forward(self.ffn_norm(h))


class VanillaCrossAttentionBlock(nn.Module):
    """Pre-norm causal self-attention + cross-attention to a memory + FFN.

    ``forward`` matches the vendored ``FeatureConditionedBlock`` (minus its unused ``cache`` kwarg —
    the AR rollout in ``src/model.py`` runs without a KV cache). Cross-attention gets no positional
    encoding, matching searchformer, which leaves its ``rope=rope`` commented out.
    """

    def __init__(
        self,
        dim: int,
        n_heads: int,
        hidden_dim: int,
        dropout: float = 0.0,
        norm_eps: float = 1e-5,
        layer_id: int = 0,
    ):
        super().__init__()
        self.layer_id = layer_id
        self.attention = VanillaAttention(dim, n_heads, dropout=dropout, is_causal=True)
        self.attention_feat = VanillaAttention(dim, n_heads, dropout=dropout, is_causal=False)
        self.feed_forward = VanillaFeedForward(dim, hidden_dim, dropout=dropout)
        self.attention_norm = nn.LayerNorm(dim, eps=norm_eps)
        self.attention_feat_norm = nn.LayerNorm(dim, eps=norm_eps)
        self.ffn_norm = nn.LayerNorm(dim, eps=norm_eps)

    def forward(
        self,
        input: Tensor,
        features: Tensor,
        mask_features: Optional[Tensor] = None,
    ) -> Tensor:
        input_norm = self.attention_norm(input)
        h = input + self.attention(input_norm, input_norm, input_norm)
        h_feat = self.attention_feat(
            queries=self.attention_feat_norm(h),
            keys=features,
            values=features,
            mask=mask_features,
        )
        h = h + h_feat
        return h + self.feed_forward(self.ffn_norm(h))
