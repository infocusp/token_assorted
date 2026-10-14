# Method details

How the two stages work, moved out of the main README. For setup, usage and
results, see the [main README](../README.md).

## Stage 1 — VQ-VAE: CoT compression into latent codes

Trains a **VQ-VAE** on MetaMathQA that compresses each chain-of-thought into a short sequence of
discrete latent codes (16 text tokens → 1 code, codebook of 1024 entries) and can decode those codes
back to text. The trained model is used once, by `src/encode_dataset.py`, to write the code artifact
that Stage 2 consumes.

### The method

Each example is `X = P ⊕ C ⊕ S` (`⊕` = concat): `P` = prompt/question, `C` = chain-of-thought,
`S` = solution/answer. The VQ-VAE learns a codebook `E` of `|E|` vectors and has five components:

1. **token embedding** — learned table over the tokenizer vocab `V` → `d`-dim vectors (standalone; no LLM weights).
2. **encoder `f_enc`** — causal Transformer mapping a length-`L` chunk to `L/r` latent embeddings `X̄`.
3. **quantizer `q`** — nearest-neighbour snap `q(x̄) = argmin_{e∈E} ‖e − x̄‖²`, returns the chosen vector + integer index.
4. **prompt encoder `g(P)`** — masked mean-pool over the token embeddings of `P` → one `d`-dim conditioning vector for the decoder (parameter-free; no separate module).
5. **decoder `f_dec`** — reconstructs the `L` tokens from the `L/r` quantized embeddings, conditioned on `g(P)`.

**Loss (Eq. 2)** with `X̄ = f_enc(embed(X))` and `sg[·]` = stop-gradient:

```
L(X) = L_recon  +  ‖ sg[X̄] − q(X̄) ‖²  +  β · ‖ X̄ − sg[q(X̄)] ‖²
```

`L_recon` = token-level cross-entropy of decoder logits vs. the original tokens; `β = 0.25`.
The straight-through estimator and the VQ/commitment terms are provided by
[`vector-quantize-pytorch`](https://github.com/lucidrains/vector-quantize-pytorch) — we use its
returned quantized output and loss rather than re-deriving them (see [Design choices](../README.md#design-choices)).

> The VQ-VAE is **trained on the whole sequence `X`**, but at encode time (`encode_dataset.py`) it is
> **applied to `C`**. The code keeps this distinction (see [encode/serve consistency](../README.md#design-choices)).

#### Architecture & tensor walkthrough (`decoder_mode: parallel`, `upsample_mode: conv_transpose`; paper dims `max_seq=256, L=r=16, d=512, |E|=1024`)

```
               X = P ⊕ C ⊕ S   ·   token ids, right-padded   (B, 256)
                                         │
                                         ▼
╭──────────────────────────────────── ENCODER ─────────────────────────────────────╮
│ token embedding                                                    (B, 256, 512) │
│ f_enc  ·  2 × causal self-attention block                          (B, 256, 512) │
│ Conv1d, kernel = stride = 16   (16 tokens → 1 latent)          (B, 16, 512)  = X̄ │
╰──────────────────────────────────────────────────────────────────────────────────╯
                          │                                          │
                          ▼                                          ▼
╭─────────────────── QUANTIZER ────────────────────╮  ╭─────── PROMPT g(P) ────────╮
│ nearest of 1024 codebook entries                 │  │ token embeddings of P,     │
│ (EMA-updated, straight-through)                  │  │ masked mean-pool           │
│ quantized latents                   (B, 16, 512) │  │ g(P)              (B, 512) │
│ code indices  =  the latent tokens       (B, 16) │  │                            │
╰──────────────────────────────────────────────────╯  ╰────────────────────────────╯
                          │                                          │
                          └────────────────────┬─────────────────────┘
                                               ▼
╭──────────────────── DECODER   (parallel, no teacher forcing) ────────────────────╮
│ ConvTranspose1d, kernel = stride = 16   (1 latent → 16 positions)  (B, 256, 512) │
│ prepend g(P) as a conditioning position                            (B, 257, 512) │
│ f_dec  ·  2 × causal self-attention block                          (B, 257, 512) │
│ drop the g(P) position  ·  tied output head                          (B, 256, V) │
╰──────────────────────────────────────────────────────────────────────────────────╯
                                         │
                                         ▼
                     loss  =  CE(logits, X)  +  β · commitment

   B = batch   V = vocab size   256 = max_seq   16 = L = r   512 = d   1024 = |E|
```

Padding: the encoder is causal, so real tokens never attend to pads; a latent slot is valid only if its
16-token chunk overlaps at least one real token, and invalid slots are masked out of the quantizer.

`decoder_mode: ar` is the alternative: a teacher-forced causal decoder (`FeatureConditionedBlock`) that
takes the shifted-right ground-truth tokens as input and cross-attends to `[g(P); quant]` as memory.

The **16 indices are the latent tokens**; compression rate `r = #text_tokens / #codes = 16`.

### Hyperparameters

All hyperparameters live in `config/vqvae_llama.yaml` (paper §4.2 / Appendix A.1): `|E|=1024`,
`d=512`, 2-layer/4-head causal encoder & decoder (no dropout), `L=r=16`, `max_seq=256`, Adam
`lr=1e-5`, `batch=32`, `100k` steps; `β=0.25` is not specified in the paper (the VQ-VAE standard). Plus the
[design-choice](../README.md#design-choices) switches (`decoder_mode`, `vq_decay`, `rotation_trick`,
`tie_head`, `compress_target`, …). CLI flags override any field; `--smoke` applies a tiny CPU
override set (`d=64`, `max_seq=64`, `r=8`, `lr=3e-4`, 60 steps).

> **FFN size:** the vendored block hard-codes FFN hidden = `4·d` with a SwiGLU `2/3` + `multiple_of=256`
> rounding, so the effective hidden ≈ 1536 at `d=512` (`ffn_mult` in config is informational).

## Stage 2 — LLM fine-tuning on assorted tokens

Fine-tunes **`unsloth/Llama-3.2-1B`** (ungated mirror of the paper's model) so it reasons over
**assorted latent+text** sequences (paper §3.2 Eq.1 + §4.2 + Appendix A.4). It reads the Stage-1
artifact (`data/metamath_latent_codes.jsonl`) and, for each example, replaces the first *m* CoT text
tokens with their `m/r` latent codes:

```
X~ = [BOS] + P + [<boLatent> z_1 … z_{m/r} <eoLatent>] + c_{m+1..} + S + [EOS]
```

### The method

1. **Extend the vocabulary** with 1026 unseen tokens: `<boLatent>`, `<eoLatent>`, and
   `<Latent-code-0..1023>` (one per codebook entry). `len(tokenizer)` 128256 → 129282;
   `resize_token_embeddings` grows the embedding; each code row starts at the mean base embedding of the
   text chunks it encodes (`new_token_init: chunk_mean`), so the rows are distinct from step 0, while
   `<boLatent>` / `<eoLatent>` are mean-initialised.
2. **Randomized replacement (paper §4.2)** — per example sample `m_max ∈ {0,72,128,160,192,224,256}`,
   then `m ∈ {0,16,32,…,m_max}`; use the first `m/16` codes as latent, keep `cot_token_ids[m:]` as
   text. `m=0` → pure-text CoT. This single-stage randomization is what lets the model adapt to the
   new tokens without curriculum learning.
3. **Full fine-tuning** of all parameters, so the 1026 new-token rows are trainable (the input and
   output embeddings stay tied) and the LLM can *read* and *emit* latent tokens. LoRA is also
   supported: set `full_finetune: false` (the `lora_*` keys then apply, and `lora_modules_to_save:
   [embed_tokens, lm_head]` must stay so the new-token rows are trainable).
4. **Training loop** is Llama Cookbook's `train()` (batch 32, sequence-packing 4096, 1 epoch,
   AdamW `lr=1e-4`, bf16), driven **single-GPU** on `cuda:0` (`enable_fsdp=False`; the cookbook's
   FSDP branch is never entered, so there is no `DistributedSampler` and no sharded checkpointing).
   At inference the LLM generates latent tokens directly — **no VQ-VAE**
   (paper §3.2); the eval harness greedily decodes and reads the final `The answer is:`.
