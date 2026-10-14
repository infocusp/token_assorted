# Third-party code

What this repository vendors or borrows, where it comes from, and under which license.
Pinned commits, file lists and hashes are in each directory's `PROVENANCE.md`.

> **Non-commercial.** `searchformer_transformer/` is CC-BY-NC 4.0 and is imported by `src/model.py`,
> so the combined repository is non-commercial as-is. Commercial use needs a license from the
> upstream authors, or replacing those files with a permissively licensed Transformer.

## Vendored code

| Directory | What it provides | Upstream | License |
|---|---|---|---|
| [`searchformer_transformer/`](searchformer_transformer) | Causal Transformer blocks (RoPE attention, SwiGLU FFN, RMSNorm) and LR schedules used by the VQ-VAE | [facebookresearch/searchformer](https://github.com/facebookresearch/searchformer) `@d33c267` | **CC-BY-NC 4.0** |
| [`official_evals/`](official_evals) | Each benchmark's own answer-scoring code, behind `accuracy_official` | [hendrycks/math](https://github.com/hendrycks/math), [microsoft/unilm](https://github.com/microsoft/unilm) (MWPBench), [TheoremQA](https://github.com/TIGER-AI-Lab/TheoremQA), [OlympiadBench](https://github.com/OpenBMB/OlympiadBench), [dart-math](https://github.com/hkust-nlp/dart-math), [MetaMath](https://github.com/meta-math/MetaMath) | MIT (MetaMath: Apache-2.0) |
| [`official_evals/_latex2sympy2/`](official_evals/_latex2sympy2) | LaTeX parser for TheoremQA, bundled with its own antlr4 4.7.2 runtime | [latex2sympy2](https://github.com/OrangeX4/latex2sympy) 1.9.1, [antlr4](https://github.com/antlr/antlr4) 4.7.2 | MIT, BSD-3-Clause |

## Design references (no code copied)

| Source | What it informed | License |
|---|---|---|
| [latentplan (TAP)](https://github.com/ZhengyaoJiang/latentplan) | The VQ-VAE encode → quantize → conditioned-decode pattern | MIT |
| [coconut](https://github.com/facebookresearch/coconut) | The data pipeline and training-loop scaffold; the P/C/S split is custom | MIT |

## Installed, not vendored

| Dependency | Used for | License |
|---|---|---|
| [vector-quantize-pytorch](https://github.com/lucidrains/vector-quantize-pytorch) 1.29.1 | Quantizer and codebook (straight-through estimator, EMA updates, dead-code restart) | MIT |
| [MetaMathQA](https://huggingface.co/datasets/meta-math/MetaMathQA) | Training data | see the dataset card |
