# Provenance: searchformer Transformer

These files are **vendored verbatim (unmodified)** from Meta's `searchformer` repository
and used as the building blocks for the encoder/decoder of this project's VQ-VAE
(paper: *Token Assorted* synthetic models use exactly this encode-decoder Transformer with
rotary embeddings and no dropout).

| | |
|---|---|
| **Upstream repo** | https://github.com/facebookresearch/searchformer |
| **Commit** | `d33c2673f36e74258a2192c8c662365d8114f8b8` |
| **Upstream path** | `searchformer/transformer/` |
| **Files vendored** | `model.py`, `rotary.py`, `optim.py`, `utils.py`, `__init__.py` |
| **License** | **Creative Commons Attribution-NonCommercial 4.0 International (CC-BY-NC 4.0)** — see `LICENSE` in this directory |
| **Modifications** | **None.** Files are copied byte-for-byte. All adaptation lives in `src/model.py`, which *imports* these classes and composes them. |

## ⚠️ NON-COMMERCIAL LICENSE

CC-BY-NC 4.0 permits use, sharing, and adaptation **for non-commercial purposes only**.
Because this code is vendored into the repository and imported by `src/model.py`, the
**combined work is effectively non-commercial**. Do not use this repository (as-is) for
commercial purposes without obtaining a separate license/waiver from the upstream authors
or replacing these files with a permissively licensed Transformer implementation.

## What this project reuses from here
`src/model.py` imports and composes:
- `RoPE` (rotary position embeddings) — `rotary.py`
- `Attention` (multi-head attention over `F.scaled_dot_product_attention`) — `model.py`
- `SelfAttentionBlock` (pre-norm self-attention + SwiGLU FFN) — used causally for `f_enc` and the `parallel` decoder
- `FeatureConditionedBlock` (causal self-attn + cross-attn to a memory, cross-attn has no RoPE) — used for the `ar` decoder
- `RMSLayerNorm`, `FeedForward` (SwiGLU), `HyperParams` — `model.py`
- `lr_constant`, `lr_cosine` LR-schedule helpers — `optim.py`
- `sample_greedy` — `utils.py`

The token embedding, Conv1d down-sampler, prompt encoder `g(P)`, the vector quantizer
(`vector-quantize-pytorch`), the tied output head, and all wiring are written from scratch.
