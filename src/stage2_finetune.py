"""Stage 2: fine-tune Llama-3.2-1B on assorted latent+text CoT (paper §3.2 + §4.2 + App. A.4).

**Single-GPU only.** There is no FSDP/DDP/torchrun path: ``run_train`` places the model on
``cuda:0`` and llama-cookbook's ``train()`` is driven with ``enable_fsdp=False``. Scale by changing
``batch_size_training`` x ``gradient_accumulation_steps``, not by adding ranks.

We reuse the **Llama Cookbook** codebase (as the paper did) for the training loop, the
``train_config`` dataclass, sequence-packing (``ConcatDataset``), and dataloader kwargs — but we
build the PEFT model ourselves. That is required because:

  1. The vocabulary is extended by 1026 unseen latent tokens (``<boLatent>``, ``<eoLatent>``,
     ``<Latent-code-0..1023>``), so the embedding matrix must be resized.
  2. ``unsloth/Llama-3.2-1B`` has ``tie_word_embeddings=true``. For the LLM to both *read* and
     *emit* the new tokens under LoRA, we set ``modules_to_save=["embed_tokens","lm_head"]`` — a
     field llama-cookbook's ``lora_config`` dataclass does not expose. PEFT then trains full
     copies of both (untying them), on top of LoRA on the attention+MLP projections.

Flash attention: the paper does not mention it, so ``use_fast_kernels=False`` (no forced
sdpa/flash); weights load in bf16 and llama-cookbook's loop trains in bf16 (no fp16 GradScaler).

``full_finetune: true`` bypasses PEFT entirely (all params trainable, ``lora_*`` keys ignored):
by default embeddings stay TIED — the single shared embed/lm_head matrix carries both the read-
and emit-path gradients for the 1026 new rows. Setting ``untie_embeddings: true`` (full-FT only)
gives ``lm_head`` its own weight (cloned from the initialized ``embed_tokens`` after the vocab
init, then trained independently) and sets ``config.tie_word_embeddings=False`` so both matrices
are saved and reloaded separately — the full-FT analogue of LoRA's ``modules_to_save`` untying.
The full HF checkpoint is saved via ``save_pretrained`` in ``run_train`` (llama-cookbook's
non-PEFT save writes a bare, unloadable ``model.pt``).

Usage:
    python -m src.stage2_finetune --config config/stage2_llama1b.yaml
    python -m src.stage2_finetune --config config/stage2_llama1b.yaml --smoke   # CPU wiring check
    python -m src.stage2_finetune --config config/stage2_llama3b.yaml           # Llama-3.2-3B
"""
from __future__ import annotations

import argparse
import os
from typing import Any, Dict

import yaml


# --------------------------------------------------------------------------------------
# Compatibility shim: llama-cookbook 0.0.5 imports accelerate.utils.is_ccl_available, which
# accelerate>=1.1 renamed to is_xccl_available. Add the old name before importing llama_cookbook.
# --------------------------------------------------------------------------------------
def _shim_accelerate() -> None:
    import accelerate.utils as au

    if not hasattr(au, "is_ccl_available"):
        au.is_ccl_available = getattr(au, "is_xccl_available", lambda *a, **k: False)


# --------------------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------------------
# NOTE: `base_model` is deliberately NOT overridden here. It used to be pinned to
# "unsloth/Llama-3.2-1B", but because resolve_cfg applies these overrides LAST they silently beat
# both the config file and the --base-model flag — so `--smoke` always ran the 1B base model and
# reported it as if it had validated the config's model. That made every non-1B config (the 3B rows,
# the -Instruct rows) unverifiable, and made init-scale sweeps read as "identical across models"
# when they had simply all measured the same model. Pass `--base-model <small>` if you explicitly
# want a cheaper smoke; the keys below are genuine smoke-shaping (tiny subset/context) and stay forced.
SMOKE_OVERRIDES: Dict[str, Any] = dict(
    context_length=256,          # small so ConcatDataset yields >=1 packed chunk on a tiny subset
    batch_size_training=2,
    val_batch_size=2,
    n_val=8,                     # keep most of the (tiny) subset in train so packing has data
    num_workers=0,
    save_model=False,
    run_validation=False,
    mlflow=False,
)


def load_cfg(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--smoke", action="store_true",
                   help="CPU/GPU wiring check: tiny subset, one fwd/bwd, assert new-token grads. "
                        "Does NOT run llama-cookbook's train() loop.")
    p.add_argument("--limit", type=int, default=None, help="Truncate the artifact to N records.")
    # a few common overrides (any other field: edit the YAML)
    p.add_argument("--base-model", dest="base_model", default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--batch-size", dest="batch_size_training", type=int, default=None)
    p.add_argument("--output-dir", dest="output_dir", default=None)
    p.add_argument("--no-mlflow", dest="mlflow", action="store_false", default=None)
    p.add_argument("--mlflow-tracking-uri", dest="mlflow_tracking_uri", default=None)
    p.add_argument("--full-finetune", dest="full_finetune", action="store_true", default=None,
                   help="Full-parameter FT instead of LoRA/PEFT (overrides the YAML key).")
    p.add_argument("--deterministic", dest="deterministic", action=argparse.BooleanOptionalAction,
                   default=None,
                   help="Bit-reproducible training (ON by default; ~7-8%% slower, +~32MB VRAM): seeds "
                        "torch, pins the cuBLAS workspace and forces deterministic kernels. "
                        "--no-deterministic opts out. Either flag overrides the YAML key.")
    return p


# Keys that named the removed multi-GPU path. A config still carrying one set to true asked for
# behaviour this repo no longer has, so say so instead of training single-GPU under a name that
# promises sharding — the silent-wrong-thing failure mode this repo has already been bitten by
# twice (`max_train_step: 2`, SMOKE_OVERRIDES pinning base_model). `false` is accepted in silence
# so every pre-existing config and saved `stage2_config.resolved.yaml` still loads.
_REMOVED_DISTRIBUTED_KEYS = ("enable_fsdp", "low_cpu_fsdp")


def _reject_removed_distributed_keys(cfg: Dict[str, Any]) -> None:
    asked = [k for k in _REMOVED_DISTRIBUTED_KEYS if cfg.get(k)]
    if asked:
        raise SystemExit(
            f"{', '.join(asked)} set to true, but the FSDP/distributed path was removed — Stage 2 "
            f"is single-GPU only (see the module docstring). Drop the key; to fit a bigger "
            f"effective batch raise batch_size_training x gradient_accumulation_steps, or turn on "
            f"activation_checkpointing."
        )


# --------------------------------------------------------------------------------------
# Determinism (`deterministic: true` — ON by default since 2026-10-06)
# --------------------------------------------------------------------------------------
# Without it Stage-2 training is NOT bit-reproducible. The forward pass is; the backward is not.
# Full FT trains the 129282 x d embedding, and PyTorch's embedding gradient accumulates with
# atomicAdd, so the float additions land in a different order every run. Over one epoch that is
# worth ~0.008 val loss and ~1 GSM8K point. A config without the key gets the default, and
# resolve_cfg writes it in explicitly so the resolved config and MLflow params record it.
_CUBLAS_WORKSPACE_CONFIG = ":4096:8"
_DETERMINISTIC_DEFAULT = True
_DETERMINISM_APPLIED = False


def _assert_determinism_applied(cfg: Dict[str, Any]) -> None:
    """Fail loudly if the config asked for determinism but nothing set it up.

    ``_apply_determinism`` has to run from ``main()`` — before torch is imported — so a caller that
    invokes ``run_train``/``run_smoke`` directly (a script, a notebook) would otherwise train
    non-deterministically while its resolved config, and its MLflow params, claimed otherwise.
    """
    if cfg.get("deterministic", _DETERMINISTIC_DEFAULT) and not _DETERMINISM_APPLIED:
        raise SystemExit(
            "deterministic: true, but _apply_determinism() never ran — CUBLAS_WORKSPACE_CONFIG has "
            "to be set before torch initializes CUDA, which only main() can guarantee. Launch via "
            "`python -m src.stage2_finetune`, or call _apply_determinism(cfg) before importing "
            "torch if you are driving run_train()/run_smoke() yourself."
        )


def _apply_determinism(cfg: Dict[str, Any]) -> None:
    """Make Stage-2 training bit-reproducible. MUST run before torch initializes CUDA.

    Three pieces, all of them required:

    * ``CUBLAS_WORKSPACE_CONFIG`` — cuBLAS allocates its GEMM scratch workspace per CUDA stream;
      without a pinned workspace the same matmul may pick a different split-k reduction order
      between runs. torch REFUSES to enable deterministic algorithms on CUDA unless this is
      ``:4096:8`` (8 x 4 MiB, ~32 MiB VRAM, fast) or ``:16:8`` (negligible memory, slower). cuBLAS
      reads it when it creates its handle, so setting it after the first matmul is too late —
      which is why ``main()`` calls this before importing torch at all.
    * ``torch.use_deterministic_algorithms(True)`` — swaps the atomicAdd kernels (embedding
      backward above all) for deterministic ones, and RAISES on any op with no deterministic
      implementation. The raise is the feature: a silent fallback would leave the run
      irreproducible while this flag claimed otherwise. ``deterministic_warn_only: true``
      downgrades it to a warning if you hit an op this stack does not cover.
    * seeds — ``torch.manual_seed`` covers CPU, every CUDA device, and DataLoader worker seeding.
      ``random``/``numpy`` are seeded for completeness, but note the m-sampling does NOT depend on
      them: ``Stage2Dataset`` builds its own ``random.Random(seed*1_000_003 + i)`` per example.

    Cost: ~7-8% slower (deterministic kernels give up atomics) plus ~32 MiB VRAM. On by default so
    any two runs of one config are comparable bit for bit; opt out with ``deterministic: false`` /
    ``--no-deterministic`` only when speed matters more than exact repeatability.
    """
    seed = int(cfg.get("seed", 42))
    warn_only = bool(cfg.get("deterministic_warn_only", False))

    # Respect an explicit choice (":16:8" is the documented low-memory alternative); only fill in
    # the default when the caller has not set one.
    preset = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if preset is None:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = _CUBLAS_WORKSPACE_CONFIG

    import random

    import torch

    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:  # numpy is not a hard dependency of this path
        pass
    torch.manual_seed(seed)  # seeds CPU and all CUDA devices
    torch.use_deterministic_algorithms(True, warn_only=warn_only)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    global _DETERMINISM_APPLIED
    _DETERMINISM_APPLIED = True

    print(f"[determinism] ON — seed={seed}, "
          f"CUBLAS_WORKSPACE_CONFIG={os.environ['CUBLAS_WORKSPACE_CONFIG']}"
          f"{' (pre-set in the environment)' if preset else ''}, warn_only={warn_only}. "
          f"Opt out with --no-deterministic for speed.")


def resolve_cfg(args) -> Dict[str, Any]:
    cfg = load_cfg(args.config)
    _reject_removed_distributed_keys(cfg)
    for k in ("base_model", "lr", "batch_size_training", "output_dir", "mlflow",
              "full_finetune", "deterministic"):
        v = getattr(args, k, None)
        if v is not None:
            cfg[k] = v
    cfg.setdefault("deterministic", _DETERMINISTIC_DEFAULT)  # explicit in resolved config + MLflow
    if args.limit is not None:
        cfg["limit"] = args.limit
    if args.smoke:
        cfg.update(SMOKE_OVERRIDES)
        cfg.setdefault("limit", 64)
        # --base-model must win over the smoke defaults (it is the documented escape hatch for
        # running a cheaper model than the config names), so re-apply it after the update.
        if getattr(args, "base_model", None) is not None:
            cfg["base_model"] = args.base_model
    return cfg


# --------------------------------------------------------------------------------------
# Model + PEFT
# --------------------------------------------------------------------------------------
def _pair_cos(mat) -> float:
    """Mean off-diagonal pairwise cosine of the rows — the degeneracy metric for code inits
    (mean-init = 1.000; Llama-3.2 base rows = 0.131; Stage-1 codebook = 0.055)."""
    import torch

    n = torch.nn.functional.normalize(mat.float(), dim=-1)
    G = n @ n.T
    return float((G.sum() - G.diag().sum()) / (G.numel() - G.shape[0]))


def _chunk_mean_targets(base, vocab, cfg: Dict[str, Any]):
    """Per-code mean of the base embeddings of the CoT tokens that code c covers in the artifact
    (Stage2Dataset's sequential rule: code i <-> cot_token_ids[i*L:(i+1)*L]).

    Only the artifact's TRAIN rows are used (same deterministic tail-holdout as everywhere else).
    Returns ``(means, seen, n_rows)``: ``means`` is (codebook_size, d) with zero rows for codes
    that never appear in train, ``seen`` the boolean mask of the rest. Shared by ``chunk_mean``
    (which writes these directly) and ``codebook``+procrustes (which only uses them to orient the
    projection).
    """
    import torch

    from .data import deterministic_train_val_split
    from .stage2_data import load_stage2_records

    rows = load_stage2_records(cfg["artifact"], limit=cfg.get("limit"))
    rows, _ = deterministic_train_val_split(rows, min(cfg["n_val"], max(len(rows) - 1, 0)))
    L = int(cfg["chunk_length"])
    n_codes = int(cfg["codebook_size"])

    sums = torch.zeros(n_codes, base.shape[1], dtype=torch.float32)
    counts = torch.zeros(n_codes, dtype=torch.long)
    codes_buf: list[int] = []
    toks_buf: list[int] = []

    def _flush() -> None:
        if not codes_buf:
            return
        c = torch.tensor(codes_buf, dtype=torch.long)
        t = torch.tensor(toks_buf, dtype=torch.long)
        sums.index_add_(0, c, base[t].float())
        counts.index_add_(0, c, torch.ones_like(c))
        codes_buf.clear()
        toks_buf.clear()

    for rec in rows:
        cot = rec["cot_token_ids"]
        for i, c in enumerate(rec["code_indices"]):
            chunk = cot[i * L:(i + 1) * L]
            codes_buf.extend([c] * len(chunk))
            toks_buf.extend(chunk)
        if len(codes_buf) >= 500_000:
            _flush()
    _flush()

    seen = counts > 0
    means = torch.zeros_like(sums)
    means[seen] = sums[seen] / counts[seen].unsqueeze(1).float()
    return means, seen, len(rows)


def _chunk_mean_init(emb, vocab, cfg: Dict[str, Any]) -> None:
    """Give each ``<Latent-code-c>`` row a DISTINCT init: its chunk-mean base embedding
    (see ``_chunk_mean_targets``), rescaled to the average base-row norm.

    Why: plain mean-init starts all 1024 code rows identical; measured after a full epoch at
    lr=1e-4 they were still a near-degenerate cone (pairwise cos 0.83-0.88 vs 0.13 for base
    tokens, norms 0.39x base) — the model could read the latent block's LENGTH but not its
    content. Chunk-mean init separates the rows from step 0 and places each code near the
    content it encodes. Unseen codes and the two delimiters keep the global mean-init already
    applied by the caller. Embeddings stay tied — one write covers read (embed) and emit
    (lm_head) paths.

    Measured limit : this only reaches code-pair cos 0.60-0.65, because a mean over
    16 base rows is dominated by the base embedding's common (anisotropic) component rather than
    by the token-specific one. ``new_token_init: codebook`` removes that ceiling.
    """
    import torch

    base = emb[: vocab.base_vocab_size]
    n_codes = int(cfg["codebook_size"])
    means_all, seen, n_rows = _chunk_mean_targets(base, vocab, cfg)
    means = means_all[seen]

    # Norm scale: with TIED embeddings each code row is also an lm_head row, so full base-mean
    # norm (scale 1.0) makes 1024 content-aligned code logits interfere at EVERY position
    # (measured: initial smoke loss 21.4 vs 4.0). The DIRECTION carries the distinctness; keep
    # the norm modest via new_token_init_scale (multiplier on the mean base-row norm).
    scale = float(cfg.get("new_token_init_scale", 1.0))
    target_norm = base.float().norm(dim=-1).mean() * scale
    means = means / means.norm(dim=-1, keepdim=True).clamp_min(1e-8) * target_norm
    code_rows = torch.tensor(vocab.code_id, dtype=torch.long)
    emb[code_rows[seen]] = means.to(emb.dtype)

    # Log the resulting geometry so the run record shows the degeneracy is actually gone.
    print(f"[init] chunk_mean: {int(seen.sum())}/{n_codes} codes initialized from "
          f"{n_rows} train records; code-pair cos={_pair_cos(emb[code_rows]):.3f} "
          f"(mean-init would be 1.000); norm matched to base mean {target_norm:.3f}")


def _load_stage1_codebook(cfg: Dict[str, Any], n_codes: int):
    """Load the Stage-1 VQ codebook that PRODUCED this artifact's code indices.

    The checkpoint path comes from the artifact's ``.meta.json`` sidecar (written by
    ``encode_dataset.py``), so the codebook can never silently drift from the indices it labels —
    a wrong codebook would look like a working init while attaching every code to the geometry of
    a different code. An explicit ``stage1_ckpt:`` key overrides it and warns on disagreement.
    Returns ``(codebook (n_codes, d_stage1), checkpoint_path)``.
    """
    import json

    import torch

    meta_path = cfg["artifact"] + ".meta.json"
    meta_ckpt = None
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            meta_ckpt = json.load(f).get("checkpoint")
    ckpt_path = cfg.get("stage1_ckpt") or meta_ckpt
    if not ckpt_path:
        raise FileNotFoundError(
            f"new_token_init=codebook needs the Stage-1 checkpoint, but {meta_path} has no "
            f"'checkpoint' field — set stage1_ckpt: in the config."
        )
    if cfg.get("stage1_ckpt") and meta_ckpt and os.path.realpath(ckpt_path) != os.path.realpath(meta_ckpt):
        print(f"[init] WARNING: stage1_ckpt={ckpt_path} differs from the checkpoint that encoded "
              f"{cfg['artifact']} ({meta_ckpt}). The code indices and the codebook may not match.")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Stage-1 checkpoint not found: {ckpt_path}")

    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)["model"]
    cb = sd["vq._codebook.embed"]            # vector-quantize-pytorch stores (n_heads, |E|, d)
    cb = cb.reshape(-1, cb.shape[-1]).float()
    if cb.shape[0] != n_codes:
        raise ValueError(f"codebook has {cb.shape[0]} entries but codebook_size={n_codes} "
                         f"({ckpt_path})")
    return cb, ckpt_path


def _codebook_init(emb, vocab, cfg: Dict[str, Any]) -> None:
    """Init each ``<Latent-code-c>`` row from the Stage-1 codebook vector for code c, carried into
    the LLM's width by a SEMI-ORTHOGONAL projection ``W`` (``W Wᵀ = I``).

    Bridging the width mismatch (Stage-1 d=512 -> Llama 2048/3072) this way is exact: for any
    ``W`` with orthonormal rows, ``<vᵢW, vⱼW> = <vᵢ, vⱼ>``, so the codebook's geometry transfers
    with zero distortion. Measured on ``ema-parallel-llama-learnedup``: codebook pair-cos 0.055
    (vs 0.131 for real Llama tokens), against 0.60-0.65 for the best ``chunk_mean`` init — i.e.
    the code rows start as mutually distinct as ordinary vocabulary. All ``codebook_size`` rows
    are written, including codes absent from the train split (which ``chunk_mean`` has to leave
    at the global mean).

    ``codebook_init_rotation`` picks WHICH isometry, and the two arms separate the two factors:
      * ``procrustes`` (default) — the rotation minimising ``||V W - T||_F`` over semi-orthogonal
        ``W``, where ``T`` is the mean-centred ``chunk_mean`` target per code: ``W = U Qᵀ`` from
        ``svd(Vᵀ T) = U S Qᵀ``. Distinct AND placed near the base tokens each code encodes.
        Rank-deficient ``T`` cannot break this — ``U`` and ``Qᵀ`` are orthonormal regardless, so
        the isometry holds even when the targets collapse; only the orientation degrades.
      * ``random`` — a seeded QR isometry. Distinct but NOT content-aligned: the ablation arm
        that says whether any gain came from the codebook's geometry or from the placement.

    The codebook rows are used direction-only: under EMA their norms span 0.24-99 and track code
    usage, not content. Norms are matched to the base rows via ``new_token_init_scale``, as in
    ``chunk_mean``. Unlike ``chunk_mean`` this deliberately drops the base embedding's mean offset
    (norm 0.353 of 0.987) — that shared component is exactly what caps chunk_mean's distinctness.
    """
    import torch
    import torch.nn.functional as F

    n_codes = int(cfg["codebook_size"])
    base = emb[: vocab.base_vocab_size]
    d = emb.shape[1]

    cb, ckpt_path = _load_stage1_codebook(cfg, n_codes)
    d_stage1 = cb.shape[1]
    if d_stage1 > d:
        raise ValueError(f"Stage-1 codebook_dim {d_stage1} exceeds the LLM width {d}: no isometric "
                         f"up-projection exists (would have to discard directions).")
    V = F.normalize(cb, dim=-1)

    rotation = cfg.get("codebook_init_rotation", "procrustes")
    if rotation not in ("procrustes", "random"):
        raise ValueError(f"unknown codebook_init_rotation: {rotation!r} (expected procrustes | random)")

    T = seen = None
    if rotation == "procrustes":
        means_all, seen, n_rows = _chunk_mean_targets(base, vocab, cfg)
        # Centre the targets: the base embedding's mean row (norm 0.353 vs 0.987) is shared by
        # every token, so fitting to it would spend the rotation on an offset an isometry cannot
        # reproduce anyway. What is left is the content-specific direction.
        T = F.normalize(means_all[seen] - base.float().mean(dim=0), dim=-1)
        U, _, Qt = torch.linalg.svd(V[seen].T @ T, full_matrices=False)
        W = U @ Qt
        fit = f"oriented on {int(seen.sum())}/{n_codes} chunk targets from {n_rows} train records"
    else:
        g = torch.Generator().manual_seed(int(cfg.get("seed", 42)))
        Q, _ = torch.linalg.qr(torch.randn(d, d_stage1, generator=g))
        W = Q.T
        fit = f"seeded random rotation (seed={int(cfg.get('seed', 42))}), no content alignment"

    ortho_err = float((W @ W.T - torch.eye(d_stage1)).abs().max())
    assert ortho_err < 1e-3, f"projection is not semi-orthogonal (max|W Wᵀ - I| = {ortho_err:.2e})"

    # ||V @ W|| == ||V|| == 1 exactly, so this just sets the norm; see new_token_init_scale.
    scale = float(cfg.get("new_token_init_scale", 1.0))
    target_norm = base.float().norm(dim=-1).mean() * scale
    rows = (V @ W) * target_norm
    code_rows = torch.tensor(vocab.code_id, dtype=torch.long)
    emb[code_rows] = rows.to(emb.dtype)

    print(f"[init] codebook({rotation}): {n_codes} codes from {ckpt_path} "
          f"(d {d_stage1} -> {d}); {fit}; code-pair cos {_pair_cos(cb):.3f} (codebook) -> "
          f"{_pair_cos(emb[code_rows]):.3f} (embeddings), max|W Wᵀ - I|={ortho_err:.1e}; "
          f"norm matched to base mean {target_norm:.3f}")
    if rotation == "procrustes":
        align = float((F.normalize(rows[seen], dim=-1) * T).sum(-1).mean())
        print(f"[init] codebook(procrustes): mean cos(code row, its chunk-mean target)={align:.3f} "
              f"(a random rotation gives ~0)")


def build_model_and_tokenizer(cfg: Dict[str, Any]):
    import torch
    from transformers import AutoModelForCausalLM

    from .stage2_data import build_extended_tokenizer

    tok, vocab = build_extended_tokenizer(
        cfg["base_model"], cfg["codebook_size"],
        cfg["bo_token"], cfg["eo_token"], cfg["code_token_fmt"],
    )

    attn_impl = "sdpa" if cfg.get("use_fast_kernels", False) else None
    model = AutoModelForCausalLM.from_pretrained(
        cfg["base_model"], dtype=torch.bfloat16, attn_implementation=attn_impl,
    )
    model.resize_token_embeddings(len(tok))

    # Init the new rows BEFORE PEFT copies the weights. Input and output embeddings are tied
    # here, so editing embed_tokens covers lm_head too.
    #   mean       — every new row = global mean of the base rows (all 1026 identical at start).
    #   chunk_mean — delimiters/unseen codes as `mean`, but each seen <Latent-code-c> row gets a
    #                distinct, norm-matched init from the artifact chunks it encodes (see
    #                _chunk_mean_init for why plain mean-init leaves the codes unreadable).
    #   codebook   — delimiters as `mean`; EVERY code row is its Stage-1 codebook vector carried
    #                into the LLM width by a semi-orthogonal projection, so the codebook geometry
    #                transfers exactly (see _codebook_init; codebook_init_rotation picks which
    #                isometry). Strictly more distinct than chunk_mean (0.055 vs 0.60 pair-cos).
    init_mode = cfg.get("new_token_init", "mean")
    if init_mode not in ("mean", "chunk_mean", "codebook"):
        raise ValueError(f"unknown new_token_init: {init_mode!r} "
                         f"(expected mean | chunk_mean | codebook)")
    with torch.no_grad():
        emb = model.get_input_embeddings().weight
        old = vocab.base_vocab_size
        emb[old:] = emb[:old].mean(dim=0, keepdim=True)
        if init_mode == "chunk_mean":
            _chunk_mean_init(emb, vocab, cfg)
        elif init_mode == "codebook":
            _codebook_init(emb, vocab, cfg)

    if cfg.get("full_finetune", False):
        # Full-parameter FT: no PEFT wrapper, every param trainable.
        if cfg.get("untie_embeddings", False):
            # Give lm_head its OWN weight so the read (embed_tokens) and emit (lm_head) paths
            # train independently — the analogue of LoRA's modules_to_save untying, but for
            # full FT. Done AFTER the (chunk_mean) init above, so lm_head starts as a clone of
            # the initialized input embedding; the two then diverge under training. Setting
            # config.tie_word_embeddings=False makes save_pretrained persist BOTH matrices, and
            # eval's from_pretrained reload keeps them separate (no code change needed there).
            import torch.nn as nn
            in_emb = model.get_input_embeddings().weight
            new_lm = nn.Linear(in_emb.shape[1], in_emb.shape[0], bias=False)
            with torch.no_grad():
                new_lm.weight = nn.Parameter(in_emb.detach().clone())
            model.set_output_embeddings(new_lm)
            model.config.tie_word_embeddings = False
            assert model.get_output_embeddings().weight is not model.get_input_embeddings().weight, (
                "untie_embeddings=True but embed_tokens and lm_head still share storage"
            )
        else:
            # Embeddings stay TIED — the shared embed/lm_head matrix gets gradients from both
            # reading and emitting the new latent tokens, so no untying is needed.
            assert model.get_output_embeddings().weight is model.get_input_embeddings().weight, (
                "expected tie_word_embeddings=True in full-finetune mode (set untie_embeddings: true to untie)"
            )
        n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        n_all = sum(p.numel() for p in model.parameters())
        print(f"trainable params: {n_train:,} || all params: {n_all:,} "
              f"|| trainable%: {100 * n_train / n_all:.4f}")
    else:
        from peft import LoraConfig, get_peft_model

        peft_cfg = LoraConfig(
            r=cfg["lora_r"], lora_alpha=cfg["lora_alpha"], lora_dropout=cfg["lora_dropout"],
            bias="none", task_type="CAUSAL_LM",
            target_modules=list(cfg["lora_target_modules"]),
            modules_to_save=list(cfg["lora_modules_to_save"]),
        )
        model = get_peft_model(model, peft_cfg)
        model.print_trainable_parameters()
    return model, tok, vocab


def build_train_config(cfg: Dict[str, Any]):
    """Instantiate llama-cookbook's real train_config dataclass (so every field train() reads exists)."""
    from llama_cookbook.configs.training import train_config as TrainConfig

    # Also checked in resolve_cfg; repeated here so a caller that builds the dict itself (a test,
    # an ad-hoc script) cannot slip a removed key past into cookbook's dataclass.
    _reject_removed_distributed_keys(cfg)
    full_ft = cfg.get("full_finetune", False)

    return TrainConfig(
        model_name=cfg["base_model"],
        # Single-GPU only (see the module docstring). This stays False so llama-cookbook's
        # train() and get_dataloader_kwargs take their single-device branches — no
        # DistributedSampler, no dist.* calls, no sharded checkpointing.
        enable_fsdp=False,
        use_peft=not full_ft,
        peft_method="lora",  # unread when use_peft=False
        quantization=None,
        batching_strategy="packing",
        context_length=cfg["context_length"],
        batch_size_training=cfg["batch_size_training"],
        val_batch_size=cfg["val_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        gradient_clipping=cfg["gradient_clipping"],
        gradient_clipping_threshold=cfg["gradient_clipping_threshold"],
        num_epochs=cfg["num_epochs"],
        max_train_step=cfg.get("max_train_step", 0),
        max_eval_step=cfg.get("max_eval_step", 0),
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
        gamma=cfg["gamma"],
        seed=cfg["seed"],
        use_fp16=False,
        mixed_precision=True,
        use_fast_kernels=cfg.get("use_fast_kernels", False),
        num_workers_dataloader=cfg["num_workers"],
        # Full FT: cookbook's non-PEFT save writes a bare torch.save state_dict (model.pt) that
        # from_pretrained cannot load — disable it and save via save_pretrained in run_train.
        save_model=cfg["save_model"] and not full_ft,
        run_validation=cfg["run_validation"],
        output_dir=cfg["output_dir"],
        use_wandb=False,
        save_metrics=True,
    )


# --------------------------------------------------------------------------------------
# Data loaders (reuse llama-cookbook packing + collator)
# --------------------------------------------------------------------------------------
def build_loaders(cfg, tok, vocab, tc):
    import torch
    from llama_cookbook.data.concatenator import ConcatDataset
    from llama_cookbook.utils.config_utils import get_dataloader_kwargs

    from .stage2_data import build_stage2_datasets

    train_ds, val_ds = build_stage2_datasets(
        cfg["artifact"], tok, vocab, cfg["m_max_set"], cfg["chunk_length"],
        cfg["seed"], cfg["n_val"], limit=cfg.get("limit"),
        sol_only=cfg.get("sol_only", False),
    )
    train_packed = ConcatDataset(train_ds, chunk_size=tc.context_length)
    val_packed = ConcatDataset(val_ds, chunk_size=tc.context_length)

    tr_kw = get_dataloader_kwargs(tc, train_packed, tok, "train")
    va_kw = get_dataloader_kwargs(tc, val_packed, tok, "val")
    train_dl = torch.utils.data.DataLoader(
        train_packed, num_workers=tc.num_workers_dataloader, pin_memory=True, **tr_kw)
    val_dl = torch.utils.data.DataLoader(
        val_packed, num_workers=tc.num_workers_dataloader, pin_memory=True, **va_kw)
    return train_dl, val_dl, train_packed, val_packed


# --------------------------------------------------------------------------------------
# Smoke: device-agnostic wiring check (does NOT use train(), which hardcodes cuda:0)
# --------------------------------------------------------------------------------------
def run_smoke(cfg: Dict[str, Any]) -> None:
    from transformers import default_data_collator

    from .stage2_data import decode_example

    _assert_determinism_applied(cfg)
    model, tok, vocab = build_model_and_tokenizer(cfg)
    tc = build_train_config(cfg)
    _, _, train_packed, _ = build_loaders(cfg, tok, vocab, tc)

    assert len(tok) == vocab.base_vocab_size + vocab.n_added
    assert vocab.n_added == cfg["codebook_size"] + 2
    # State the model explicitly: a smoke that silently validated a DIFFERENT model than the config
    # names is exactly the failure this line exists to prevent (see SMOKE_OVERRIDES).
    print(f"[smoke] base_model={cfg['base_model']}  eos_id={vocab.eos_id} pad_id={tok.pad_token_id}")
    print(f"[smoke] len(tok)={len(tok)} (base {vocab.base_vocab_size} + {vocab.n_added} new); "
          f"bo_id={vocab.bo_id} eo_id={vocab.eo_id} code[278]={vocab.code_id[278]}")

    sol_only = cfg.get("sol_only", False)
    # CoT baseline (m_max_set == [0]): _sample_m -> cap_chunks = min(0 // L, ...) = 0 -> m = 0 -> the
    # latent block is skipped for EVERY example, so the checks below that require a latent block
    # (saw_latent, and the new-token read/emit gradient asserts) can never pass. Same "no latent
    # tokens ever appear" property as sol_only, which additionally drops the CoT text. Without this
    # branch every CoT-baseline config fails --smoke even though it trains fine.
    cot_only = not sol_only and list(cfg.get("m_max_set", [])) == [0]
    no_latent = sol_only or cot_only
    no_latent_why = "sol_only=True" if sol_only else "m_max_set=[0]"

    # Inspect a few rendered examples: find one with a latent block and one without.
    ds = train_packed.dataset  # the underlying Stage2Dataset
    saw_latent, saw_plain = False, False
    for i in range(min(len(ds), 200)):
        item = ds[i]
        has_bo = vocab.bo_id in item["input_ids"]
        # labels must be -100 exactly over [BOS]+prompt and real elsewhere
        assert item["labels"][0] == -100
        n_masked = sum(1 for x in item["labels"] if x == -100)
        assert all(lab == -100 for lab in item["labels"][:n_masked]), "mask must be a leading prefix"
        if no_latent:
            # Sol-Only / CoT baseline: no latent block may EVER appear. Asserting the absence is a
            # stronger check for these configs than the saw_latent search it replaces.
            assert not has_bo, f"example {i} has a latent block but {no_latent_why}"
            continue
        if has_bo and not saw_latent:
            print(f"[smoke] example {i} WITH latent block:\n"
                  f"        {decode_example(item, tok)[:300]!r}")
            saw_latent = True
        if not has_bo and not saw_plain:
            saw_plain = True
        if saw_latent and saw_plain:
            break
    if sol_only:
        print(f"[smoke] sol_only=True: no latent blocks in first {min(len(ds), 200)} examples "
              f"(P -> S only); example 0:\n        {decode_example(ds[0], tok)[:300]!r}")
    elif cot_only:
        print(f"[smoke] m_max_set=[0]: no latent blocks in first {min(len(ds), 200)} examples "
              f"(pure-text CoT: [BOS]+P+full_CoT+answer+[EOS]); example 0:\n"
              f"        {decode_example(ds[0], tok)[:300]!r}")
    else:
        assert saw_latent, "no example with a latent block found in first 200 records"
    print(f"[smoke] packed chunks={len(train_packed)} (chunk_size={tc.context_length})")
    assert len(train_packed) >= 1, "ConcatDataset produced 0 chunks; lower context_length or raise limit"

    chunk = train_packed[0]
    assert len(chunk["input_ids"]) == len(chunk["labels"]) == tc.context_length
    assert any(lab == -100 for lab in chunk["labels"]), "prompt masks (-100) not preserved by packing"

    # One forward/backward on a packed chunk; assert new-token rows receive gradient.
    batch = default_data_collator([chunk])
    model.train()
    out = model(**batch)
    out.loss.backward()

    if no_latent:
        # Sol-Only and the CoT baseline never read OR emit latent tokens, so the new-token rows
        # legitimately get no read-path gradient (they are never inputs) and no emit-path gradient
        # from any label. Skip those checks; just confirm full-FT keeps everything trainable.
        # (Under TIED embeddings the rows still take a small push-down gradient via the softmax
        # denominator, so asserting grad == 0 would be wrong too — hence skip, not invert.)
        print(f"[smoke] {no_latent_why}: skipping new-token grad checks "
              f"(no latent tokens are read or emitted)")
        if cfg.get("full_finetune", False):
            n_frozen = sum(1 for p in model.parameters() if not p.requires_grad)
            assert n_frozen == 0, f"{n_frozen} frozen params in full-FT mode"
    elif cfg.get("full_finetune", False) and cfg.get("untie_embeddings", False):
        # Untied full-FT: embed_tokens (read) and lm_head (emit) are independent tensors; BOTH
        # new-token row blocks must receive gradient (else the model can't read OR can't emit).
        in_emb = model.get_input_embeddings().weight
        out_emb = model.get_output_embeddings().weight
        assert out_emb is not in_emb, "untie_embeddings=True but embed_tokens and lm_head share storage"
        emb_grad = float(in_emb.grad[vocab.base_vocab_size:].abs().sum()) if in_emb.grad is not None else 0.0
        lm_grad = float(out_emb.grad[vocab.base_vocab_size:].abs().sum()) if out_emb.grad is not None else 0.0
        print(f"[smoke] full-FT UNTIED new-token grad L1: embed_tokens={emb_grad} lm_head={lm_grad}")
        assert emb_grad > 0.0, "embed_tokens new rows got no gradient (model can't read codes)!"
        assert lm_grad > 0.0, "lm_head new rows got no gradient (model can't emit codes)!"
        n_frozen = sum(1 for p in model.parameters() if not p.requires_grad)
        assert n_frozen == 0, f"{n_frozen} frozen params in full-FT mode"
    elif cfg.get("full_finetune", False):
        # Tied embeddings: lm_head.weight IS embed_tokens.weight (named_parameters() dedups it),
        # so this one tensor carries both the read-path and emit-path gradients.
        emb = model.get_input_embeddings().weight
        assert model.get_output_embeddings().weight is emb, "expected tied embeddings in full-FT mode"
        new_grad = float(emb.grad[vocab.base_vocab_size:].abs().sum()) if emb.grad is not None else 0.0
        print(f"[smoke] full-FT new-token grad L1 (tied embed_tokens/lm_head): {new_grad}")
        assert new_grad > 0.0, "new-token rows got no gradient!"
        n_frozen = sum(1 for p in model.parameters() if not p.requires_grad)
        assert n_frozen == 0, f"{n_frozen} frozen params in full-FT mode"
    else:
        checked = {}
        for name, p in model.named_parameters():
            if p.requires_grad and "modules_to_save" in name and name.endswith("weight"):
                if "embed_tokens" in name or "lm_head" in name:
                    key = "embed_tokens" if "embed_tokens" in name else "lm_head"
                    g = p.grad
                    new_grad = float(g[vocab.base_vocab_size:].abs().sum()) if g is not None else 0.0
                    checked[key] = new_grad
        print(f"[smoke] new-token grad L1 sums: {checked}")
        assert checked.get("embed_tokens", 0.0) > 0.0, "embed_tokens new rows got no gradient!"
        assert checked.get("lm_head", 0.0) > 0.0, "lm_head new rows got no gradient (model can't emit codes)!"
    print(f"[smoke] loss={float(out.loss):.4f}  — Stage-2 wiring OK")


# --------------------------------------------------------------------------------------
# Live metric logging: llama-cookbook's train()/evaluation() emit step/epoch metrics only
# through a wandb-style `wandb_run.log(dict, commit=...)` hook (it has no MLflow path). We pass
# this adapter as that "wandb run" so every micro-step train loss and each validation pass land
# in the active MLflow run as they happen — not just the end-of-run averages.
# --------------------------------------------------------------------------------------
class _MlflowRunLogger:
    """Forward llama-cookbook's ``wandb_run.log(...)`` calls to the active MLflow run.

    train() logs ``{'train/epoch','train/step','train/loss'}`` once per micro-step and
    ``{'eval/perplexity','eval/loss'}`` once per validation pass. ``train/step`` (the global
    micro-step index) becomes the MLflow step axis for training metrics; validation points are
    keyed by a running counter so successive epochs' val curves don't overwrite each other.
    ``train/epoch`` is metadata, not a metric, so it is dropped.
    """

    def __init__(self, mlflow) -> None:
        self._mlflow = mlflow
        self._eval_idx = 0

    def log(self, data: Dict[str, Any], commit: bool = True, **_) -> None:
        is_eval = any(k.startswith("eval/") for k in data)
        metrics: Dict[str, float] = {}
        for k, v in data.items():
            if k in ("train/step", "train/epoch"):
                continue
            try:
                metrics[k] = float(v)  # v is a 0-dim torch tensor or python number
            except (TypeError, ValueError):
                continue
        if not metrics:
            return
        if is_eval:
            self._mlflow.log_metrics(metrics, step=self._eval_idx)
            self._eval_idx += 1
        else:
            step = data.get("train/step")
            self._mlflow.log_metrics(metrics, step=int(step) if step is not None else 0)


# --------------------------------------------------------------------------------------
# Full training via llama-cookbook train()
# --------------------------------------------------------------------------------------
def run_train(cfg: Dict[str, Any], mlflow_tracking_uri: str | None,
              config_path: str | None = None) -> None:
    import torch
    from torch.optim.lr_scheduler import StepLR
    from llama_cookbook.utils.train_utils import train

    assert torch.cuda.is_available(), (
        "Full Stage-2 training uses llama-cookbook's train() loop, which places batches on "
        "'cuda:0'. Run on a GPU, or use --smoke on CPU."
    )

    _assert_determinism_applied(cfg)
    tc = build_train_config(cfg)

    os.makedirs(cfg["output_dir"], exist_ok=True)

    model, tok, vocab = build_model_and_tokenizer(cfg)
    train_dl, val_dl, _, _ = build_loaders(cfg, tok, vocab, tc)

    # Single GPU: the model lives on cuda:0, which is also where llama-cookbook's train() puts
    # every batch.
    model = model.to("cuda:0")
    # Activation checkpointing trades compute for memory, allowing a longer context length
    # within a fixed VRAM budget than would otherwise fit.
    # use_reentrant=False is required for PEFT (grads flow through modules_to_save copies).
    if cfg.get("activation_checkpointing", False):
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=tc.lr, weight_decay=tc.weight_decay,
    )
    # llama-cookbook steps the scheduler once per EPOCH; with num_epochs=1 the LR is constant
    # at `lr` for the whole run (matches the paper's fixed LR). StepLR mirrors the lib default.
    scheduler = StepLR(optimizer, step_size=1, gamma=tc.gamma)

    mlf = None
    run_logger = None
    run_id = None
    if cfg.get("mlflow", False):
        import mlflow  # lazy
        if mlflow_tracking_uri:
            mlflow.set_tracking_uri(mlflow_tracking_uri)
        mlflow.set_experiment(cfg.get("mlflow_experiment", "stage2(SFT)"))
        run_name = cfg.get("mlflow_run_name") or os.path.basename(cfg["output_dir"].rstrip("/"))
        run_id = mlflow.start_run(run_name=run_name).info.run_id
        mlflow.log_params({k: (str(v) if isinstance(v, (list, dict)) else v)
                           for k, v in cfg.items()})
        # Log the launch config yaml immediately (before train()) so it survives a mid-run
        # crash — unlike the resolved config below, which is only logged on success. CLI
        # overrides applied on top of this file live in the params above / the resolved copy.
        if config_path and os.path.exists(config_path):
            mlflow.log_artifact(config_path, artifact_path="config")
        mlf = mlflow
        run_logger = _MlflowRunLogger(mlflow)  # live per-step train + per-epoch val metrics

    # fsdp_config / local_rank / rank stay at their single-device defaults (None).
    results = train(
        model, train_dl, val_dl, tok, optimizer, scheduler,
        tc.gradient_accumulation_steps, tc, wandb_run=run_logger,
    )

    print(f"[stage2] train() results: {results}")

    # LoRA: train() saves the PEFT checkpoint (adapter + modules_to_save) to output_dir on
    # the best epoch. Full FT: cookbook's save is disabled (build_train_config), so save the
    # full HF checkpoint here (~2.5GB bf16; tied weights -> one embedding copy, or two under
    # untie_embeddings: config.tie_word_embeddings=False makes save_pretrained persist both). With
    # num_epochs=1 "final weights" == "best epoch". Then persist the tokenizer + resolved
    # config alongside for inference.
    if cfg.get("full_finetune", False) and cfg["save_model"]:
        model.save_pretrained(cfg["output_dir"])
    tok.save_pretrained(cfg["output_dir"])
    with open(os.path.join(cfg["output_dir"], "stage2_config.resolved.yaml"), "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    if mlf is not None:
        # End-of-run averages (avg_train_loss/prep, avg_eval_loss/prep, timings).
        mlf.log_metrics({k: float(v) for k, v in results.items()
                         if isinstance(v, (int, float))})
        # Record the run id so a later eval job can attach benchmark metrics to THIS run
        # (src/stage2_eval.py reads it from the adapter dir).
        with open(os.path.join(cfg["output_dir"], "mlflow_run_id.txt"), "w") as f:
            f.write(run_id)
        # Full per-step/per-epoch train+val loss & perplexity series that train() dumped to
        # JSON (complete record; the live curves above are logged from the same numbers).
        metrics_file = results.get("metrics_filename")
        if metrics_file and os.path.exists(metrics_file):
            mlf.log_artifact(metrics_file, artifact_path="metrics")
        mlf.log_artifact(os.path.join(cfg["output_dir"], "stage2_config.resolved.yaml"))
        # Checkpoint (LoRA: adapter + modules_to_save, ~1 GB; full FT: full HF model,
        # ~2.5 GB) + tokenizer, config, run id as an artifact. It lands under mlruns/
        # (gitignored), so gate it behind a flag.
        if cfg.get("mlflow_log_checkpoint", True):
            mlf.log_artifacts(cfg["output_dir"], artifact_path="checkpoint")
        mlf.end_run()


def main() -> None:
    # Argument/config resolution touches only argparse + yaml, so torch is still unimported here —
    # which is what lets _apply_determinism put CUBLAS_WORKSPACE_CONFIG in the environment before
    # cuBLAS can create its handle. Keep this order: determinism BEFORE _shim_accelerate(), which
    # imports accelerate and therefore torch.
    args = build_argparser().parse_args()
    cfg = resolve_cfg(args)
    if cfg.get("deterministic", _DETERMINISTIC_DEFAULT):
        _apply_determinism(cfg)
    _shim_accelerate()
    if args.smoke:
        run_smoke(cfg)
    else:
        run_train(cfg, args.mlflow_tracking_uri, config_path=args.config)


if __name__ == "__main__":
    main()
