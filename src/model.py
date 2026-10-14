"""The Stage-1 VQ-VAE (paper §3.1, Eq. 2).

Wires five components into one module:
  1. token embedding  (written here; nn.Embedding over the tokenizer vocab)
  2. encoder f_enc     (causal self-attention stack)
  3. quantizer q       (``vector-quantize-pytorch`` ``VectorQuantize`` — not hand-rolled)
  4. prompt encoder g(P)(written here; masked mean-pool of prompt token embeddings)
  5. decoder f_dec     (AR=cross-attention block over the codes; parallel=causal self-attention
                        over the up-sampled codes), conditioned on g(P)

The transformer block family behind (2) and (5) is selected by ``cfg.transformer_impl``:
  * ``searchformer`` (default) — vendored blocks: RMSNorm + RoPE + SwiGLU. The validated recipe
    behind every existing run and checkpoint.
  * ``vanilla`` — ``src/vanilla_blocks.py``: LayerNorm + GELU + learned absolute positions.
Both are pre-norm with identical topology and identical block ``forward`` signatures, so every call
site below is shared; they differ only in components. Self-attention is causal in both.

Down-sampling T -> T/r codes uses a learnable ``Conv1d(d, d, kernel=r, stride=r)`` (README "Design choices").
The output head is tied to the token embedding by default (README "Design choices").

What is reused vs written:
  * reused (vendored, CC-BY-NC 4.0): RoPE, Attention, SelfAttentionBlock, FeatureConditionedBlock,
    RMSLayerNorm, FeedForward, HyperParams — see third_party/searchformer_transformer/. Used only
    when ``transformer_impl=searchformer``.
  * reused (MIT, pip): VectorQuantize.
  * written from scratch: token embedding, Conv1d down-sampler, up-sampling, g(P), the AR/parallel
    decoder wiring, tied head, padding/mask handling, the whole forward/encode/decode flow, and the
    entire ``vanilla`` block family.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from vector_quantize_pytorch import VectorQuantize

from third_party.searchformer_transformer.model import (
    FeatureConditionedBlock,
    HyperParams,
    RMSLayerNorm,
    SelfAttentionBlock,
)
from third_party.searchformer_transformer.rotary import RoPE

from .vanilla_blocks import VanillaCrossAttentionBlock, VanillaSelfAttentionBlock

# Headroom over max_seq for RoPE: the parallel decoder runs length T+1 (g(P) prepended), and
# RoPE asserts offset+seq_len < max_seq_len (strict), so we need > T+1.
_ROPE_HEADROOM = 8

# Headroom over max_seq for the vanilla impl's learned absolute position table, for the same reason
# (the parallel decoder runs length T+1). Kept separate from _ROPE_HEADROOM so the two impls' bounds
# stay independent even though the value coincides.
_POS_HEADROOM = 8


@dataclass
class VQVAEOutput:
    logits: Tensor          # (B, T, V)
    vq_loss: Tensor         # scalar: beta*commitment (EMA), or codebook+commitment (non-EMA)
    indices: Tensor         # (B, n_codes) long — the latent token indices
    chunk_valid: Tensor     # (B, n_codes) bool — which codes correspond to real (non-pad) chunks
    xbar_rms: Tensor        # scalar: RMS of pre-quantization latents over valid chunks (drift watch)


def _build_hparams(cfg, vocab_size: int, rope_max: int) -> HyperParams:
    """Construct searchformer HyperParams for our block stacks.

    Note: ``SelfAttentionBlock``/``FeatureConditionedBlock`` hard-code FFN hidden = 4*dim and
    apply a SwiGLU 2/3 + ``multiple_of`` rounding internally, so ``cfg.ffn_mult`` is informational
    (the vendored block fixes it at 4x). With dim=512, multiple_of=256 the effective FFN ~ 1536.
    """
    return HyperParams(
        name="vqvae",
        dim=cfg.d_model,
        n_layers=cfg.n_layers,
        n_heads=cfg.n_heads,
        dropout=cfg.dropout,
        multiple_of=256,
        norm_eps=cfg.norm_eps,
        norm_affine=True,
        max_seq_len=rope_max,
        vocab_size=vocab_size,
        output_size=vocab_size,
        rope_freq=cfg.rope_freq,
    )


# ---------------------------------------------------------------------- block factories
# The three construction points that differ between transformer_impl arms. Everything downstream
# (forward, the decoders, decode()) is shared, because both block families expose the same
# forward signatures: blk(h) for self-attention, blk(h, features=, mask_features=) for cross.


def _vanilla_kwargs(cfg) -> dict:
    """Shared constructor kwargs for the vanilla blocks. ffn_mult is honored here (see Config.ffn_mult)."""
    return dict(
        dim=cfg.d_model,
        n_heads=cfg.n_heads,
        hidden_dim=cfg.ffn_mult * cfg.d_model,
        dropout=cfg.dropout,
        norm_eps=cfg.norm_eps,
    )


def _make_self_block(cfg, hp: HyperParams, rope, layer_id: int) -> nn.Module:
    """One causal self-attention block (encoder and parallel decoder)."""
    if cfg.transformer_impl == "vanilla":
        return VanillaSelfAttentionBlock(layer_id=layer_id, **_vanilla_kwargs(cfg))
    return SelfAttentionBlock(hp=hp, layer_id=layer_id, rope=rope, is_causal=True)


def _make_cross_block(cfg, hp: HyperParams, rope, layer_id: int) -> nn.Module:
    """One causal-self-attention + cross-attention block (AR decoder)."""
    if cfg.transformer_impl == "vanilla":
        return VanillaCrossAttentionBlock(layer_id=layer_id, **_vanilla_kwargs(cfg))
    return FeatureConditionedBlock(hp=hp, layer_id=layer_id, rope=rope)


def _make_norm(cfg) -> nn.Module:
    """The final norm applied after a block stack."""
    if cfg.transformer_impl == "vanilla":
        return nn.LayerNorm(cfg.d_model, eps=cfg.norm_eps)
    return RMSLayerNorm((cfg.d_model,), eps=cfg.norm_eps)


class TokenVQVAE(nn.Module):
    def __init__(self, cfg, vocab_size: int, pad_id: int, bos_id: int, eos_id: int):
        super().__init__()
        self.cfg = cfg
        self.vocab_size = vocab_size
        self.pad_id = pad_id
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.r = cfg.compression_rate
        self.decoder_mode = cfg.decoder_mode
        self.upsample_mode = cfg.upsample_mode

        rope_max = cfg.max_seq + _ROPE_HEADROOM
        hp = _build_hparams(cfg, vocab_size, rope_max)
        self.hp = hp

        # 1. token embedding (shared; output head tied to it by default)
        self.tok_embed = nn.Embedding(vocab_size, cfg.d_model)

        # Position information. searchformer applies RoPE inside every attention layer; vanilla adds
        # one learned absolute table at each stack's input (see _add_pos). Exactly one is built, so a
        # checkpoint's state_dict identifies its impl and a cross-impl load fails loudly.
        self.rope = None
        self.pos_embed = None
        if cfg.transformer_impl == "vanilla":
            self.pos_embed = nn.Embedding(cfg.max_seq + _POS_HEADROOM, cfg.d_model)
        else:
            # shared RoPE (head_dim, used by encoder self-attn and decoder self-attn)
            self.rope = RoPE(dim=hp.dim // hp.n_heads, max_seq_len=rope_max, base=cfg.rope_freq)

        # 2. encoder f_enc — causal self-attention stack
        self.enc_layers = nn.ModuleList(
            _make_self_block(cfg, hp, self.rope, i) for i in range(cfg.n_layers)
        )
        self.enc_norm = _make_norm(cfg)

        # down-sampler T -> T/r  (default)
        self.down = nn.Conv1d(cfg.d_model, cfg.d_model, kernel_size=self.r, stride=self.r)

        # 3. quantizer q (Eq. 2). codebook_dim defaults to d_model (paper: codebook in R^d);
        #    codebook_dim < d_model is buggy with forward(mask=...) in vq-pytorch 1.29.1.
        vq_kwargs = dict(
            dim=cfg.d_model,
            codebook_size=cfg.codebook_size,
            codebook_dim=cfg.codebook_dim_resolved(),
            decay=cfg.vq_decay,
            commitment_weight=cfg.commitment_beta,
            kmeans_init=cfg.kmeans_init,
            kmeans_iters=cfg.kmeans_iters,
            threshold_ema_dead_code=cfg.threshold_ema_dead_code,
            rotation_trick=cfg.rotation_trick,  # False -> paper's plain straight-through estimator
            use_cosine_sim=cfg.use_cosine_sim,  # False -> Euclidean codebook; True -> cosine-sim (§ config)
            ema_update=cfg.vq_ema,
        )
        if not cfg.vq_ema:
            vq_kwargs["learnable_codebook"] = True  # codebook updated by gradient, not EMA
        self.vq = VectorQuantize(**vq_kwargs)
        # VectorQuantize.__init__ does not forward reset_cluster_size to the inner Codebook, so
        # it defaults to threshold_ema_dead_code — a restarted code then starts AT the expiry
        # threshold and is re-expired after a single unused batch. Set it directly to give
        # restarted codes a grace period (e.g. 2.0 at decay 0.99 -> ~300 unused steps).
        if cfg.vq_reset_cluster_size is not None:
            self.vq._codebook.reset_cluster_size = float(cfg.vq_reset_cluster_size)

        # 5. decoder f_dec. Two block types: AR = FeatureConditionedBlock (cross-attends the codes as
        #    memory, teacher-forced); parallel = causal SelfAttentionBlock over the up-sampled codes.
        #    hybrid builds BOTH (curriculum: an AR warmup then parallel) under distinct names so they
        #    coexist; self.active_mode selects which one runs on each forward.
        def _ar_stack():
            return nn.ModuleList(
                _make_cross_block(cfg, hp, self.rope, i) for i in range(cfg.n_layers)
            )

        def _par_stack():
            return nn.ModuleList(
                _make_self_block(cfg, hp, self.rope, i) for i in range(cfg.n_layers)
            )

        def _conv_up():
            # learned up-sampler T/r -> T: the exact inverse of self.down (kernel=stride=r, no
            # overlap), so output length is exactly T and each of the r intra-chunk positions gets
            # its own d->d projection of the code instead of an identical copy.
            return nn.ConvTranspose1d(cfg.d_model, cfg.d_model, kernel_size=self.r, stride=self.r)

        if cfg.decoder_mode == "ar":
            self.dec_layers = _ar_stack()
            self.dec_norm = _make_norm(cfg)
        elif cfg.decoder_mode == "parallel":
            self.dec_layers = _par_stack()
            self.dec_norm = _make_norm(cfg)
            if cfg.upsample_mode == "conv_transpose":
                self.up = _conv_up()
        else:  # hybrid: both decoders present at once
            assert cfg.upsample_mode == "conv_transpose", (
                "hybrid decoder's parallel phase requires the learned up-sampler "
                "(upsample_mode=conv_transpose)."
            )
            self.ar_dec_layers = _ar_stack()
            self.ar_dec_norm = _make_norm(cfg)
            self.par_dec_layers = _par_stack()
            self.par_dec_norm = _make_norm(cfg)
            self.up = _conv_up()

        # which decoder runs on each forward. For hybrid the train loop flips this between "ar" and
        # "parallel" (and evaluate() pins it to cfg.hybrid_eval_mode); otherwise it is decoder_mode.
        self.active_mode = "parallel" if cfg.decoder_mode == "hybrid" else cfg.decoder_mode

        # output head
        self.tie_head = cfg.tie_head
        if not cfg.tie_head:
            self.head = nn.Linear(cfg.d_model, vocab_size, bias=False)

        # GPT-style small init for the (tied) embedding: keeps initial logits ~uniform so the
        # starting recon CE is ~ln(V) rather than huge — important at the paper's lr=1e-5.
        nn.init.normal_(self.tok_embed.weight, mean=0.0, std=0.02)
        if not cfg.tie_head:
            nn.init.normal_(self.head.weight, mean=0.0, std=0.02)
        if self.pos_embed is not None:
            # MUST stay at tok_embed's std. `nn.Embedding` defaults to N(0, 1), and `_add_pos` does
            # `h = tok + pos` at BOTH the encoder and decoder stack inputs — so the default left
            # position 50x larger in norm than content.
            nn.init.normal_(self.pos_embed.weight, mean=0.0, std=0.02)

    # ------------------------------------------------------------------ factory
    @classmethod
    def build_from_config(cls, cfg, info) -> "TokenVQVAE":
        return cls(cfg, info.vocab_size, info.pad_id, info.bos_id, info.eos_id)

    # ------------------------------------------------------------------ pieces
    def _add_pos(self, h: Tensor) -> Tensor:
        """Add learned absolute positions at a stack's input (vanilla impl only).

        A no-op under searchformer, whose RoPE is applied inside each attention layer instead.
        Positions always start at 0 — correct for the AR rollout too, since decode() re-runs the
        whole stack over the growing prefix rather than caching.
        """
        if self.pos_embed is None:
            return h
        L = h.shape[1]
        n_pos = self.pos_embed.num_embeddings
        assert L <= n_pos, (
            f"sequence length {L} exceeds the learned position table ({n_pos} rows = max_seq + "
            f"{_POS_HEADROOM}). Raise _POS_HEADROOM or lower max_seq."
        )
        pos = torch.arange(L, device=h.device)
        return h + self.pos_embed(pos).unsqueeze(0)

    def _encode_to_latents(self, input_ids: Tensor, attention_mask: Tensor):
        """embed -> causal encoder -> mask pads -> Conv1d down-sample -> X̄, chunk_valid."""
        tok = self.tok_embed(input_ids)            # (B, T, d)
        h = self._add_pos(tok)
        for blk in self.enc_layers:
            h = blk(h)
        h = self.enc_norm(h)

        token_mask = attention_mask.bool()         # (B, T)
        h = h * token_mask.unsqueeze(-1)           # zero pad positions before pooling
        xbar = self.down(h.transpose(1, 2)).transpose(1, 2)  # (B, n_codes, d)

        # a chunk is valid if it overlaps >= 1 real token
        chunk_valid = (
            F.max_pool1d(token_mask.float().unsqueeze(1), kernel_size=self.r, stride=self.r)
            .squeeze(1)
            .bool()
        )  # (B, n_codes)
        return tok, xbar, chunk_valid

    def _prompt_embedding(self, tok: Tensor, prompt_len: Tensor) -> Tensor:
        """g(P): masked mean of prompt token embeddings -> (B, d)."""
        B, T, _ = tok.shape
        pos = torch.arange(T, device=tok.device).unsqueeze(0)      # (1, T)
        pmask = (pos < prompt_len.unsqueeze(1)).float().unsqueeze(-1)  # (B, T, 1)
        denom = pmask.sum(1).clamp_min(1.0)
        return (tok * pmask).sum(1) / denom

    @staticmethod
    def _feature_mask(valid: Tensor) -> Tensor:
        """Additive cross-attention mask (B,1,1,K): 0 for valid keys, -inf for invalid."""
        fm = (1.0 - valid.float())[:, None, None, :]
        return fm.masked_fill(fm == 1.0, float("-inf"))

    def _head_logits(self, h: Tensor) -> Tensor:
        if self.tie_head:
            return F.linear(h, self.tok_embed.weight)
        return self.head(h)

    def _shift_right(self, input_ids: Tensor) -> Tensor:
        out = input_ids.new_full(input_ids.shape, self.bos_id)
        out[:, 1:] = input_ids[:, :-1]
        return out

    # ------------------------------------------------------------------ decoders
    def _decoder_stack(self):
        """(layers, norm) for the currently-active decoder — resolves hybrid's dual stacks."""
        if self.decoder_mode == "hybrid":
            if self.active_mode == "ar":
                return self.ar_dec_layers, self.ar_dec_norm
            return self.par_dec_layers, self.par_dec_norm
        return self.dec_layers, self.dec_norm

    def _decode_ar(self, input_ids, quant, gP, chunk_valid) -> Tensor:
        layers, norm = self._decoder_stack()
        B = quant.shape[0]
        memory = torch.cat([gP.unsqueeze(1), quant], dim=1)            # (B, 1+n_codes, d)
        mem_valid = torch.cat(
            [torch.ones(B, 1, dtype=torch.bool, device=quant.device), chunk_valid], dim=1
        )
        mem_mask = self._feature_mask(mem_valid)                       # (B,1,1,1+n_codes)

        dec_in_ids = self._shift_right(input_ids)
        if self.training and self.cfg.dec_input_dropout > 0:
            drop = torch.rand(dec_in_ids.shape, device=dec_in_ids.device) < self.cfg.dec_input_dropout
            drop[:, 0] = False  # keep BOS
            dec_in_ids = dec_in_ids.masked_fill(drop, self.pad_id)

        h = self._add_pos(self.tok_embed(dec_in_ids))
        for blk in layers:
            h = blk(h, features=memory, mask_features=mem_mask)
        return self._head_logits(norm(h))

    def _decode_parallel(self, quant, gP) -> Tensor:
        layers, norm = self._decoder_stack()
        if self.upsample_mode == "conv_transpose":
            up = self.up(quant.transpose(1, 2)).transpose(1, 2)       # (B, T, d) learned inverse
        else:
            up = quant.repeat_interleave(self.r, dim=1)               # (B, T, d) copy each code r times
        dec_in = torch.cat([gP.unsqueeze(1), up], dim=1)              # (B, T+1, d)
        h = self._add_pos(dec_in)
        for blk in layers:
            h = blk(h)
        h = norm(h)[:, 1:, :]                                         # drop g(P) position
        return self._head_logits(h)

    def _eq2_vq_loss(self, xbar: Tensor, indices: Tensor, chunk_valid: Tensor) -> Tensor:
        """Paper Eq. 2 VQ terms with a loss-trained (non-EMA) codebook:

            || sg[X̄] - q(X̄) ||^2  +  beta * || X̄ - sg[q(X̄)] ||^2

        The first (VQ/codebook) term has weight 1 and its gradient trains the codebook embedding
        (a learnable Parameter when ``vq_ema=False``); the second is the commitment term (weight
        beta) whose gradient trains the encoder. Both are masked to real (non-pad) chunks and
        mean-reduced. This is the literal Eq. 2 — unlike vector-quantize-pytorch's ``learnable``
        mode, which folds both into a single ``beta``-weighted term (under-weighting the codebook).
        Reconstruction still reaches the encoder via the quantizer's straight-through estimator,
        exactly as Eq. 2 prescribes.
        """
        codes = self.codes_from_indices(indices)              # e = q(X̄); grad -> codebook Parameter
        m = chunk_valid.unsqueeze(-1).to(xbar.dtype)          # (B, n_codes, 1)
        denom = m.sum().clamp_min(1.0) * xbar.shape[-1]       # masked mean over valid chunks * d
        codebook_loss = (((xbar.detach() - codes) ** 2) * m).sum() / denom
        commit_loss = (((xbar - codes.detach()) ** 2) * m).sum() / denom
        return codebook_loss + self.cfg.commitment_beta * commit_loss

    # ------------------------------------------------------------------ forward
    def forward(self, input_ids, attention_mask, prompt_len) -> VQVAEOutput:
        tok, xbar, chunk_valid = self._encode_to_latents(input_ids, attention_mask)
        quant, indices, vq_loss = self.vq(xbar, mask=chunk_valid)
        if not self.cfg.vq_ema:
            # Non-EMA: replace the library's merged (beta-weighted) term with the literal Eq. 2
            # VQ + commitment loss (weight-1 codebook term). The quantizer still gave us the
            # straight-through `quant` for reconstruction and ran kmeans-init / dead-code restart.
            vq_loss = self._eq2_vq_loss(xbar, indices, chunk_valid)
        gP = self._prompt_embedding(tok, prompt_len)
        if self.active_mode == "ar":
            logits = self._decode_ar(input_ids, quant, gP, chunk_valid)
        else:
            logits = self._decode_parallel(quant, gP)
        with torch.no_grad():
            m = chunk_valid.unsqueeze(-1).to(xbar.dtype)
            denom = m.sum().clamp_min(1.0) * xbar.shape[-1]
            xbar_rms = (((xbar**2) * m).sum() / denom).sqrt()
        return VQVAEOutput(logits=logits, vq_loss=vq_loss, indices=indices,
                           chunk_valid=chunk_valid, xbar_rms=xbar_rms)

    # ------------------------------------------------------------------ inference API
    @torch.no_grad()
    def encode(self, input_ids, attention_mask):
        """Encode to discrete code indices (no codebook update). Used by encode_dataset.py."""
        _, xbar, chunk_valid = self._encode_to_latents(input_ids, attention_mask)
        quant, indices, _ = self.vq(xbar, mask=chunk_valid, freeze_codebook=True)
        return indices, chunk_valid, quant

    def codes_from_indices(self, indices: Tensor) -> Tensor:
        """Map code indices -> quantized embeddings (B, n_codes, d)."""
        vq = self.vq
        if hasattr(vq, "get_output_from_indices"):
            return vq.get_output_from_indices(indices)
        if hasattr(vq, "get_codes_from_indices"):
            return vq.get_codes_from_indices(indices)
        return F.embedding(indices, vq.codebook)

    @torch.no_grad()
    def decode(
        self,
        indices: Tensor,
        gP: Optional[Tensor] = None,
        chunk_valid: Optional[Tensor] = None,
        max_len: Optional[int] = None,
    ) -> Tensor:
        """Reconstruct token ids from code indices (+ optional g(P)). For sanity/interpretability.

        AR mode does greedy autoregressive rollout; parallel mode does a single argmax pass.
        """
        quant = self.codes_from_indices(indices)                      # (B, n_codes, d)
        B, n_codes, d = quant.shape
        if gP is None:
            gP = quant.new_zeros(B, d)
        if chunk_valid is None:
            chunk_valid = torch.ones(B, n_codes, dtype=torch.bool, device=quant.device)
        max_len = max_len or self.cfg.max_seq

        if self.active_mode == "parallel":
            logits = self._decode_parallel(quant, gP)
            return logits.argmax(-1)

        # AR greedy rollout (no KV cache; fine for a handful of examples)
        layers, norm = self._decoder_stack()
        memory = torch.cat([gP.unsqueeze(1), quant], dim=1)
        mem_valid = torch.cat(
            [torch.ones(B, 1, dtype=torch.bool, device=quant.device), chunk_valid], dim=1
        )
        mem_mask = self._feature_mask(mem_valid)
        ids = torch.full((B, 1), self.bos_id, dtype=torch.long, device=quant.device)
        finished = torch.zeros(B, dtype=torch.bool, device=quant.device)
        for _ in range(max_len):
            h = self._add_pos(self.tok_embed(ids))
            for blk in layers:
                h = blk(h, features=memory, mask_features=mem_mask)
            logits = self._head_logits(norm(h)[:, -1:, :])            # (B,1,V)
            nxt = logits.argmax(-1)                                    # (B,1)
            ids = torch.cat([ids, nxt], dim=1)
            finished |= nxt.squeeze(1) == self.eos_id
            if finished.all():
                break
        return ids[:, 1:]  # drop BOS


def load_checkpoint(path: str, device, tokenizer_override: Optional[str] = None):
    """Rebuild (model, tokenizer, info, cfg) from a training checkpoint, in eval mode.

    The checkpoint stores the tokenizer name; encode/decode must use the SAME tokenizer the
    model was trained with (the embedding/vocab depend on it), so `tokenizer_override` should be
    left unset except for deliberate experiments.
    """
    from .data import load_tokenizer, token_info  # lazy import to avoid a module cycle
    from .utils import Config

    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = Config.from_dict(ckpt["config"])
    tok_name = tokenizer_override or ckpt.get("tokenizer_name", cfg.tokenizer_name)
    tokenizer = load_tokenizer(tok_name)
    info = token_info(tokenizer)
    model = TokenVQVAE.build_from_config(cfg, info).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, tokenizer, info, cfg
