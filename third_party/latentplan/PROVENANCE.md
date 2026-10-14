# Provenance: latentplan (TAP) — design reference only

No source code from latentplan is vendored into this repository. latentplan is cited by
*Token Assorted* as the reference VQ-VAE pattern, and it informed this project's
**architecture pattern** — but the concrete pieces it provides are superseded here by
`vector-quantize-pytorch` (quantizer) and `searchformer` (Transformer), so copying its
source would only ship unused code.

| | |
|---|---|
| **Upstream repo** | https://github.com/ZhengyaoJiang/latentplan |
| **Paper** | *Efficient Planning in a Compact Latent Action Space* (TAP), ICLR 2023, arXiv:2208.10291 |
| **License** | MIT |
| **Relevant files (for reference)** | `latentplan/models/vqvae.py`, `latentplan/models/autoencoders.py` |

## Pattern adapted (re-implemented, not copied)
- **Sequence → few discrete latent codes → conditioned reconstruction.** latentplan
  down-samples a trajectory with `MaxPool1d(kernel=latent_step, stride=latent_step)`; this
  project uses a learnable `Conv1d(d, d, kernel=L, stride=L)` instead (a design choice).
- **Decoder conditioning on side information.** latentplan conditions its decoder on the
  *state* by concatenation + linear mixing. This project conditions on the **prompt** `g(P)`:
  by default `g(P)` is prepended to the cross-attention memory of an autoregressive decoder
  (`decoder_mode: ar`), or prepended as a conditioning token in `decoder_mode: parallel`.

See the README "Borrow map" for the full reused-vs-written breakdown.
