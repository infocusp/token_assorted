"""Train the Stage-1 VQ-VAE on MetaMathQA (paper §4.2 "VQ-VAE Training").

Loop scaffold (AMP, grad-clip, checkpointing, periodic eval, logging) adapted from the pattern
in facebookresearch/coconut `run.py`. Optimizer is Adam (paper); the LR-schedule helpers
(`lr_constant`/`lr_cosine`) are reused from the vendored searchformer `optim.py`.

Examples
--------
# CPU smoke test (uses gpt2 + a tiny offline fixture; ~1-2 min):
    python -m src.train --config config/vqvae_llama.yaml --smoke

# Full training on GPU (the config sets the unsloth/Llama-3.2-1B tokenizer, 100k steps, batch 32):
    python -m src.train --config config/vqvae_llama.yaml
"""
from __future__ import annotations

import argparse
import functools
import json
import os
from contextlib import nullcontext
from datetime import datetime
from typing import Dict, Optional

import torch
import yaml
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR

from third_party.searchformer_transformer.optim import lr_constant, lr_cosine

from .data import build_loaders, load_tokenizer, token_info
from .losses import compute_loss_and_metrics
from .model import TokenVQVAE
from .utils import (
    SMOKE_OVERRIDES,
    Config,
    MetricLogger,
    apply_determinism,
    apply_overrides,
    assert_gpu_ok,
    code_usage_counts,
    codebook_metrics_from_counts,
    count_params,
    load_config,
    select_device,
    set_seed,
)

_EVAL_MAX_BATCHES = 63  # covers the full 2k-row val tail at batch 32 (global codebook metrics
#                         should see the whole corpus); still a cheap pass for the 2-layer model


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------
def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train the Stage-1 Token-Assorted VQ-VAE.")
    p.add_argument("--config", default="config/vqvae_llama.yaml", help="YAML config path.")
    p.add_argument("--smoke", action="store_true", help="Tiny CPU run on an offline fixture.")
    # config-field overrides (None => use YAML/smoke value)
    p.add_argument("--tokenizer", dest="tokenizer_name", default=None)
    p.add_argument("--device", default=None, help="auto|cuda|mps|cpu")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--deterministic", dest="deterministic", action=argparse.BooleanOptionalAction,
                   default=None,
                   help="Bit-reproducible training (ON by default; ~2-11%% slower, +~32MB VRAM): pins the "
                        "cuBLAS workspace and forces deterministic kernels. --no-deterministic opts "
                        "out. Either flag overrides the YAML key.")
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--batch-size", dest="batch_size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--max-seq", dest="max_seq", type=int, default=None)
    p.add_argument("--decoder-mode", dest="decoder_mode", default=None,
                   choices=["ar", "parallel", "hybrid"])
    p.add_argument("--upsample-mode", dest="upsample_mode", default=None,
                   choices=["repeat", "conv_transpose"])
    p.add_argument("--transformer-impl", dest="transformer_impl", default=None,
                   choices=["searchformer", "vanilla"],
                   help="Block family for the encoder + both decoders: searchformer (vendored; "
                        "RMSNorm+RoPE+SwiGLU) or vanilla (LayerNorm+GELU+learned positions).")
    p.add_argument("--ar-phase-frac", dest="ar_phase_frac", type=float, default=None,
                   help="hybrid only: fraction of steps run in the AR phase (runs first).")
    p.add_argument("--data-fixture", dest="data_fixture", default=None)
    p.add_argument("--out-dir", dest="out_dir", default=None)
    p.add_argument("--run-name", dest="run_name", default=None)
    p.add_argument("--num-workers", dest="num_workers", type=int, default=None)
    p.add_argument("--no-amp", action="store_true", help="Disable mixed precision.")
    # runtime-only
    p.add_argument("--limit", type=int, default=None, help="Truncate dataset (quick runs).")
    p.add_argument("--resume", default=None, help="Checkpoint path to resume from.")
    p.add_argument("--wandb", action="store_true", help="Log to Weights & Biases.")
    p.add_argument("--mlflow", action=argparse.BooleanOptionalAction, default=None,
                   help="Override the config's `mlflow` key (the shipped configs set it true; "
                        "--no-mlflow turns it off). --smoke forces it off unless --mlflow is passed.")
    p.add_argument("--mlflow-experiment", dest="mlflow_experiment", default=None,
                   help="MLflow experiment name (overrides config.mlflow_experiment; "
                        "Stage-1 runs are grouped here).")
    p.add_argument("--mlflow-run-name", dest="mlflow_run_name", default=None,
                   help="MLflow run name (overrides config.mlflow_run_name; "
                        "falls back to --run-name / timestamp if unset).")
    p.add_argument("--mlflow-tracking-uri", dest="mlflow_tracking_uri", default=None,
                   help="MLflow tracking URI (default: $MLFLOW_TRACKING_URI or ./mlruns).")
    return p


_OVERRIDE_FIELDS = [
    "tokenizer_name", "device", "seed", "deterministic", "steps", "batch_size", "lr", "max_seq",
    "decoder_mode", "upsample_mode", "transformer_impl", "ar_phase_frac", "data_fixture",
    "out_dir", "run_name", "num_workers", "mlflow", "mlflow_experiment", "mlflow_run_name",
]


def resolve_config(args: argparse.Namespace) -> Config:
    cfg = load_config(args.config)
    if args.smoke:
        cfg = apply_overrides(cfg, SMOKE_OVERRIDES)
    overrides = {f: getattr(args, f) for f in _OVERRIDE_FIELDS}
    cfg = apply_overrides(cfg, overrides)
    if args.no_amp:
        cfg = apply_overrides(cfg, {"amp": False})
    return cfg


# --------------------------------------------------------------------------------------
# Optim / schedule (Adam + vendored schedule fns)
# --------------------------------------------------------------------------------------
def build_optimizer(model: torch.nn.Module, cfg: Config):
    optimizer = Adam(
        model.parameters(), lr=cfg.lr, betas=cfg.betas, weight_decay=cfg.weight_decay
    )
    if cfg.lr_schedule == "constant":
        if cfg.warmup > 0:
            fn = functools.partial(lr_constant, warmup=cfg.warmup)
        else:
            fn = lambda step: 1.0  # noqa: E731  flat LR (paper)
    else:
        fn = functools.partial(
            lr_cosine, warmup=max(cfg.warmup, 1), n_steps=cfg.steps,
            cycle_length=1.0, theta=1.0, min_ratio=0.1,
        )
    return optimizer, LambdaLR(optimizer, fn)


# --------------------------------------------------------------------------------------
# Train / eval
# --------------------------------------------------------------------------------------
def _to_device(batch: Dict, device) -> Dict:
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def _cycle(loader):
    while True:
        for batch in loader:
            yield batch


@torch.no_grad()
def evaluate(model, val_loader, cfg, pad_id, device, autocast_ctx) -> Optional[Dict[str, float]]:
    """Averaged per-batch metrics + corpus-level codebook usage over the whole val pass.

    The per-batch `codebook_active_frac` has a ceiling of (valid codes per batch)/|E|, so the
    `global_*` metrics — code-usage counts accumulated across all eval batches — are the ones
    that distinguish churn from collapse. model.eval() freezes the codebook (EMA updates and
    dead-code expiry are gated on `self.training`), so this is a frozen-codebook measurement.
    """
    if len(val_loader) == 0:
        return None
    model.eval()
    # In hybrid runs the AR and parallel decoders coexist; hybrid_eval_mode picks which one eval
    # measures: "parallel"/"ar" pin it to that decoder (parallel -> recon_acc/best.pt stay comparable
    # across the switch and to existing parallel runs); "active" leaves it on the phase's current
    # decoder (diagnostic). Restore before returning either way.
    prev_active_mode = getattr(model, "active_mode", None)
    if getattr(model, "decoder_mode", None) == "hybrid" and cfg.hybrid_eval_mode != "active":
        model.active_mode = cfg.hybrid_eval_mode
    agg: Dict[str, float] = {}
    n = 0
    global_counts = torch.zeros(cfg.codebook_size)
    for i, batch in enumerate(val_loader):
        if i >= _EVAL_MAX_BATCHES:
            break
        batch = _to_device(batch, device)
        with autocast_ctx:
            out = model(batch["input_ids"], batch["attention_mask"], batch["prompt_len"])
            _, metrics = compute_loss_and_metrics(out, batch["input_ids"], pad_id, cfg.codebook_size)
        global_counts += code_usage_counts(out.indices, cfg.codebook_size, out.chunk_valid).cpu()
        for k, v in metrics.items():
            agg[k] = agg.get(k, 0.0) + float(v)
        n += 1
    model.train()
    if prev_active_mode is not None:
        model.active_mode = prev_active_mode
    result = {k: v / max(n, 1) for k, v in agg.items()}
    gcb = codebook_metrics_from_counts(global_counts)
    result["global_active_frac"] = gcb["active_fraction"]
    result["global_perplexity"] = gcb["perplexity"]
    result["global_codes_counted"] = gcb["n_codes_counted"]
    return result


def save_checkpoint(path, model, optimizer, scheduler, step, cfg, best_metric, tokenizer_name):
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "step": step,
            "config": cfg.to_dict(),
            "best_metric": best_metric,
            "tokenizer_name": tokenizer_name,
        },
        path,
    )


def main() -> None:
    args = build_argparser().parse_args()
    cfg = resolve_config(args)
    # Determinism BEFORE any CUDA work: cuBLAS fixes its workspace config at its first matmul, and
    # assert_gpu_ok() below runs one. apply_determinism() refuses once CUDA is initialized.
    if cfg.deterministic:
        apply_determinism(cfg.deterministic_warn_only)
    set_seed(cfg.seed)

    # smoke mode runs fully offline on a tiny synthetic fixture; create it if absent
    if args.smoke and cfg.data_fixture and not os.path.exists(cfg.data_fixture):
        from scripts.make_fixture import make_rows

        os.makedirs(os.path.dirname(os.path.abspath(cfg.data_fixture)), exist_ok=True)
        with open(cfg.data_fixture, "w") as f:
            for r in make_rows(128, seed=0):
                f.write(json.dumps(r) + "\n")
        print(f"[smoke] generated fixture -> {cfg.data_fixture}")

    device = select_device(cfg.device)
    if device.type == "cuda":
        assert_gpu_ok()
    print(f"[setup] device={device} decoder_mode={cfg.decoder_mode} "
          f"max_seq={cfg.max_seq} r={cfg.compression_rate} n_codes={cfg.n_codes}")

    # tokenizer + data
    tokenizer = load_tokenizer(cfg.tokenizer_name)
    info = token_info(tokenizer)
    print(f"[setup] tokenizer={cfg.tokenizer_name} vocab={info.vocab_size} "
          f"pad={info.pad_id} bos={info.bos_id} eos={info.eos_id}")
    train_loader, val_loader = build_loaders(cfg, tokenizer, info, limit=args.limit)
    print(f"[setup] train_batches/epoch={len(train_loader)} val_examples={len(val_loader.dataset)}")
    # An empty train_loader makes _cycle() spin forever yielding nothing (silent hang), so
    # fail loudly instead — e.g. dataset mis-split, or batch_size > #train examples with drop_last.
    if len(train_loader) == 0:
        raise RuntimeError(
            f"train_loader has 0 batches (train_examples={len(train_loader.dataset)}, "
            f"batch_size={cfg.batch_size}, drop_last=True). Refusing to start an infinite "
            "no-op loop — check the dataset load/split."
        )

    # model + optim
    model = TokenVQVAE.build_from_config(cfg, info).to(device)
    print(f"[setup] params: total={count_params(model):,} trainable={count_params(model, True):,}")
    optimizer, scheduler = build_optimizer(model, cfg)

    amp_enabled = cfg.amp and device.type == "cuda"
    autocast_ctx = (
        torch.autocast("cuda", dtype=torch.bfloat16) if amp_enabled else nullcontext()
    )

    # run dir + logging
    run_name = cfg.run_name or datetime.now().strftime("vqvae-%Y%m%d-%H%M%S")
    run_dir = os.path.join(cfg.out_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "config.resolved.yaml"), "w") as f:
        yaml.safe_dump(cfg.to_dict(), f, sort_keys=False)
    logger = MetricLogger(
        run_dir,
        use_wandb=args.wandb,
        wandb_kwargs=dict(project="tokenassorted-vqvae", name=run_name, config=cfg.to_dict()),
        use_mlflow=cfg.mlflow,
        mlflow_kwargs=dict(experiment_name=cfg.mlflow_experiment,
                           run_name=cfg.mlflow_run_name or run_name,
                           tracking_uri=args.mlflow_tracking_uri),
    )
    logger.log_params({
        **cfg.to_dict(),
        "device": str(device),
        "vocab_size": info.vocab_size,
        "params_total": count_params(model),
        "params_trainable": count_params(model, True),
    })
    logger.log_artifact(os.path.join(run_dir, "config.resolved.yaml"))
    # Also log the raw launch config yaml (pre-CLI-override, as passed on the command line)
    # under config/ — captured immediately so it survives a mid-run crash.
    logger.log_artifact(args.config, artifact_path="config")
    print(f"[setup] run_dir={run_dir}")

    # resume
    start_step, best_metric = 0, -1.0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_step = ckpt["step"]
        best_metric = ckpt.get("best_metric", -1.0)
        print(f"[resume] from {args.resume} at step {start_step}")

    # train loop
    model.train()
    data_iter = _cycle(train_loader)
    smoke_recons = []  # track recon_ce to report decrease for the smoke acceptance check
    # hybrid curriculum: run the AR decoder for the first ar_phase_frac of steps, then flip the
    # shared encoder/codebook onto the parallel decoder for the rest (encoder + quantizer train
    # through the whole run). ar_switch is the last AR step.
    ar_switch = round(cfg.ar_phase_frac * cfg.steps) if cfg.decoder_mode == "hybrid" else 0
    if cfg.decoder_mode == "hybrid":
        print(f"[setup] hybrid decoder: AR steps 1..{ar_switch}, parallel steps "
              f"{ar_switch + 1}..{cfg.steps} (ar_phase_frac={cfg.ar_phase_frac}); "
              f"eval always uses '{cfg.hybrid_eval_mode}'")
    for step in range(start_step + 1, cfg.steps + 1):
        if cfg.decoder_mode == "hybrid":
            model.active_mode = "ar" if step <= ar_switch else "parallel"
            if step == ar_switch + 1:
                print(f"[hybrid] step {step}: switching AR -> parallel decoder")
        batch = _to_device(next(data_iter), device)
        optimizer.zero_grad(set_to_none=True)
        with autocast_ctx:
            out = model(batch["input_ids"], batch["attention_mask"], batch["prompt_len"])
            loss, metrics = compute_loss_and_metrics(out, batch["input_ids"], info.pad_id, cfg.codebook_size)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        scheduler.step()

        if args.smoke:
            smoke_recons.append(float(metrics["recon_ce"]))

        if step % cfg.log_every == 0 or step == 1:
            metrics["lr"] = scheduler.get_last_lr()[0]
            metrics["grad_norm"] = float(grad_norm)
            logger.log(step, metrics, split="train")

        if cfg.eval_every and step % cfg.eval_every == 0:
            val = evaluate(model, val_loader, cfg, info.pad_id, device, autocast_ctx)
            if val is not None:
                logger.log(step, val, split="val")
                # best.pt tracks the deployment (parallel) decoder only. For hybrid runs the eval
                # may have used the AR decoder (hybrid_eval_mode=ar, or =active during the AR
                # phase) — its teacher-forced recon_acc is inflated and its parallel decoder may be
                # untrained, so those evals are NOT best.pt candidates. Non-hybrid runs are
                # unaffected (every eval qualifies).
                if cfg.decoder_mode == "hybrid":
                    eval_decoder = (
                        ("ar" if step <= ar_switch else "parallel")
                        if cfg.hybrid_eval_mode == "active"
                        else cfg.hybrid_eval_mode
                    )
                    parallel_candidate = eval_decoder == "parallel"
                else:
                    parallel_candidate = True
                if parallel_candidate and val["recon_acc"] > best_metric:
                    best_metric = val["recon_acc"]
                    save_checkpoint(os.path.join(run_dir, "best.pt"), model, optimizer,
                                    scheduler, step, cfg, best_metric, cfg.tokenizer_name)
                    print(f"[best] step {step}: new best val recon_acc={best_metric:.4f} "
                          f"(parallel decoder)")

        if cfg.ckpt_every and step % cfg.ckpt_every == 0:
            save_checkpoint(os.path.join(run_dir, "last.pt"), model, optimizer,
                            scheduler, step, cfg, best_metric, cfg.tokenizer_name)

    # final checkpoint
    save_checkpoint(os.path.join(run_dir, "last.pt"), model, optimizer,
                    scheduler, cfg.steps, cfg, best_metric, cfg.tokenizer_name)
    logger.finish()

    if args.smoke and len(smoke_recons) >= 10:
        early = sum(smoke_recons[:5]) / 5
        late = sum(smoke_recons[-5:]) / 5
        print(f"[smoke] recon_ce: early={early:.4f} late={late:.4f} delta={early - late:+.4f} "
              f"({'DECREASED ✓' if late < early else 'did NOT decrease ✗'})")
    print(f"[done] best val recon_acc={best_metric:.4f} ckpts in {run_dir}")


if __name__ == "__main__":
    main()
