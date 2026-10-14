"""Config, reproducibility, device, and metrics helpers.

Paper: *Token Assorted* (Su et al., ICML 2025, arXiv:2502.03275), §4 / Appendix A.1.
Nothing about the model/training is hard-coded in the other modules: every knob lives
on :class:`Config`, loaded from YAML and overridable from the CLI.
"""
from __future__ import annotations

import csv
import json
import os
import random
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Dict, Optional

import numpy as np
import torch
import yaml


# --------------------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------------------
@dataclass
class Config:
    """All hyperparameters for the Stage-1 VQ-VAE.

    Defaults are the paper's math-reasoning setting (§4.2 / Appendix A.1). The CPU smoke
    path overrides a subset via :data:`SMOKE_OVERRIDES`.
    """

    # ---- tokenizer / data ----
    tokenizer_name: str = "gpt2"  # open/non-gated default. Swap to meta-llama/Llama-3.2-1B
    #                               (paper default; gated -> `huggingface-cli login`) to match Stage 2.
    dataset_name: str = "meta-math/MetaMathQA"
    dataset_split: str = "train"
    data_fixture: Optional[str] = None  # path to a JSONL fixture (offline smoke/tests)
    n_val: int = 2000                   # held-out tail rows (no HF val split exists)
    compress_target: str = "cot"        # {cot, full}: what encode_dataset.py compresses
    num_workers: int = 2

    # ---- chunking (§4 "Chunking") ----
    max_seq: int = 256                  # T: pad/truncate X = P (+) C (+) S to this length
    compression_rate: int = 16          # r: text tokens per latent code (Conv1d stride/kernel)
    chunk_length: int = 16              # L: paper's chunk length; == r here (1 code / chunk)

    # ---- model (Appendix A.1) ----
    d_model: int = 512                  # d
    n_layers: int = 2                   # both f_enc and f_dec
    n_heads: int = 4
    ffn_mult: int = 4                   # FFN hidden = ffn_mult * d. NOTE: live only under
    #                                     transformer_impl=vanilla; under searchformer the vendored
    #                                     block hard-codes 4*d + SwiGLU 2/3 + multiple_of=256
    #                                     rounding (~1536 at d=512), so this field is informational
    #                                     there. Same YAML => 1536 (searchformer) vs 2048 (vanilla).
    dropout: float = 0.0                # paper: no dropout
    norm_eps: float = 1e-5
    rope_freq: float = 10000.0          # searchformer only (vanilla uses learned absolute positions)
    transformer_impl: str = "searchformer"  # {searchformer, vanilla}: which block family builds the
    #                                     encoder and both decoders.
    #                                     searchformer -> vendored blocks (RMSNorm + RoPE + SwiGLU);
    #                                       the validated recipe behind every existing run/checkpoint.
    #                                     vanilla -> src/vanilla_blocks.py (LayerNorm + GELU +
    #                                       learned absolute positions), written from scratch.
    #                                     MUST default to searchformer: Config.from_dict silently
    #                                     defaults missing keys, so old checkpoints (whose stored
    #                                     config predates this field) rebuild their original
    #                                     architecture and keep loading strict. Self-attention is
    #                                     causal in BOTH impls — Stage-1 pad-invariance relies on
    #                                     right-padding + causality with no attention mask
    #                                     (tests/test_masking.py), so there is no non-causal option.

    # ---- decoder wiring (design choices) ----
    decoder_mode: str = "ar"            # {ar, parallel, hybrid}. ar -> teacher-forced (README "Design choices");
    #                                     parallel -> paper f_dec (no teacher forcing); hybrid ->
    #                                     builds BOTH decoders and runs an AR warmup then parallel
    #                                     (curriculum): first ar_phase_frac of steps use the AR
    #                                     decoder, the rest use the parallel decoder. The encoder +
    #                                     quantizer are shared and train through the whole run.
    ar_phase_frac: float = 0.25         # hybrid only: fraction of `steps` run in the AR phase (runs
    #                                     first); the remainder run the parallel decoder. Ignored
    #                                     unless decoder_mode == "hybrid".
    hybrid_eval_mode: str = "parallel"  # hybrid only: which decoder evaluate() uses.
    #                                     "parallel" -> always the deployment decoder, so recon_acc /
    #                                       best.pt stay comparable to existing parallel runs.
    #                                     "ar" -> always the teacher-forced decoder.
    #                                     "active" -> whichever decoder is training right now (AR in
    #                                       the warmup, parallel after the switch): surfaces each
    #                                       phase's own reconstruction on val. NOTE: with "active"
    #                                       the logged recon_acc is NOT comparable across the switch
    #                                       (AR teacher-forced acc is inflated). best.pt is safe
    #                                       regardless — train.py restricts best.pt candidates to
    #                                       parallel-decoder evals for every hybrid run.
    tie_head: bool = True               # tie output projection to the token embedding
    dec_input_dropout: float = 0.0      # AR only: randomly drop decoder input tokens to force latent use
    upsample_mode: str = "repeat"       # parallel decoder only: {repeat, conv_transpose}. repeat ->
    #                                     each code copied r times (repeat_interleave, current behavior);
    #                                     conv_transpose -> learned ConvTranspose1d(d,d,k=s=r), the exact
    #                                     inverse of the Conv1d down-sampler, giving each intra-chunk
    #                                     position its own d->d map (vs an identical copy).

    # ---- vector quantizer (§3.1 Eq.2; vector-quantize-pytorch) ----
    codebook_size: int = 1024           # |E|
    codebook_dim: Optional[int] = None  # None -> d_model (paper: codebook vectors in R^d).
    #                                     NOTE: codebook_dim < d_model is buggy together with
    #                                     forward(mask=...) in vector-quantize-pytorch 1.29.1.
    commitment_beta: float = 0.25       # beta in Eq.2
    vq_decay: float = 0.99              # EMA decay (>0.8 default: collapse guard for 1024 codes)
    vq_ema: bool = True                 # EMA codebook (codebook term realized by EMA, not the loss)
    kmeans_init: bool = True            # init codebook from first batch (cold-start guard)
    kmeans_iters: int = 10
    threshold_ema_dead_code: float = 2  # dead-code restart: expire codes whose EMA usage/batch
    #                                     drops below this. Steady-state EMA usage equals a code's
    #                                     mean per-batch count (~0.4 uniform at 512 codes/batch over
    #                                     |E|=1024), so values >= ~0.4 force permanent churn.
    vq_reset_cluster_size: Optional[float] = None  # cluster_size assigned to freshly restarted codes.
    #                                     None -> library default (= threshold), which re-expires an
    #                                     unused restarted code after ONE batch; set higher (e.g. 2.0)
    #                                     for a grace period (~300 unused steps at decay 0.99).
    rotation_trick: bool = False        # False -> paper's plain straight-through estimator
    use_cosine_sim: bool = False        # False -> Euclidean codebook (paper/validated recipe).
    #                                     True -> cosine-similarity codebook: X̄ and codebook rows are
    #                                     L2-normalized before nearest-code lookup (Yu et al. 2022,
    #                                     "Improved VQGAN"), a known collapse guard that decouples code
    #                                     assignment from latent norm. Opt-in only; toggling it for a
    #                                     real run changes quantizer geometry.

    # ---- optimization (§4.2) ----
    lr: float = 1e-5                    # Adam
    betas: tuple = (0.9, 0.999)
    weight_decay: float = 0.0           # paper uses Adam (not AdamW)
    batch_size: int = 32
    steps: int = 100_000
    grad_clip: float = 1.0
    warmup: int = 0                     # LR warmup steps (0 -> flat lr; uses vendored schedule fns)
    lr_schedule: str = "constant"       # {constant, cosine}
    amp: bool = True                    # mixed precision (CUDA only; auto-disabled on CPU/MPS)

    # ---- run / logging ----
    seed: int = 0
    deterministic: bool = True          # true -> bit-reproducible training (see apply_determinism):
    #                                     pins CUBLAS_WORKSPACE_CONFIG=:4096:8 and forces deterministic
    #                                     kernels. ON by default since 2026-10-06 (GPU-verified: r1 = r2
    #                                     to 55k steps, identical weights); costs ~2-11% per step. false
    #                                     (or --no-deterministic) is faster, but then the GPU kernels
    #                                     reorder float atomics (kmeans-init scatter_add_,
    #                                     repeat_interleave / SDPA backward, cuDNN conv backward) and two
    #                                     identical GPU runs drift apart. Seeds are set either way.
    deterministic_warn_only: bool = False  # deterministic only: downgrade "no deterministic kernel
    #                                     for this op" from an error to a warning. The run is then NOT
    #                                     reproducible — warn-only also leaves flash/mem-efficient SDPA
    #                                     backward on its nondeterministic kernel.
    device: str = "auto"                # {auto, cuda, mps, cpu}
    out_dir: str = "runs"
    run_name: Optional[str] = None
    mlflow: bool = False                        # MLflow logging; the shipped configs set true (--mlflow/--no-mlflow override)
    mlflow_experiment: str = "stage1(VQ-VAE)"  # MLflow experiment (groups all Stage-1 runs)
    mlflow_run_name: Optional[str] = None       # MLflow run name; None -> falls back to run_name / timestamp
    novelty: Optional[str] = None       # free-text: what this run changes vs. prior runs; metadata
    #                                     only — logged to MLflow params / config.resolved.yaml,
    #                                     read by nothing else.
    log_every: int = 20
    eval_every: int = 500
    ckpt_every: int = 1000

    def __post_init__(self) -> None:
        assert self.decoder_mode in ("ar", "parallel", "hybrid"), self.decoder_mode
        assert self.transformer_impl in ("searchformer", "vanilla"), self.transformer_impl
        assert self.upsample_mode in ("repeat", "conv_transpose"), self.upsample_mode
        assert self.hybrid_eval_mode in ("ar", "parallel", "active"), self.hybrid_eval_mode
        assert 0.0 <= self.ar_phase_frac <= 1.0, self.ar_phase_frac
        # conv_transpose upsampling lives on the parallel decode path (_decode_parallel); under AR
        # it would build an unused module and silently no-op. It is valid for decoder_mode=parallel
        # and for hybrid (whose parallel phase requires the learned up-sampler) — reject only AR.
        assert self.upsample_mode == "repeat" or self.decoder_mode in ("parallel", "hybrid"), (
            "upsample_mode=conv_transpose only applies to decoder_mode=parallel or hybrid; "
            f"got decoder_mode={self.decoder_mode}."
        )
        assert self.compress_target in ("cot", "full"), self.compress_target
        assert self.lr_schedule in ("constant", "cosine"), self.lr_schedule
        assert self.max_seq % self.compression_rate == 0, (
            f"max_seq ({self.max_seq}) must be divisible by compression_rate "
            f"({self.compression_rate}) so the Conv1d down-sampler emits a whole number of codes."
        )
        assert self.chunk_length == self.compression_rate, (
            "This implementation emits one code per chunk, so chunk_length (L) must equal "
            f"compression_rate (r). Got L={self.chunk_length}, r={self.compression_rate}."
        )
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads."
        self.betas = tuple(self.betas)

    @property
    def n_codes(self) -> int:
        """Number of latent codes per sequence (= T / r)."""
        return self.max_seq // self.compression_rate

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    def codebook_dim_resolved(self) -> int:
        return self.codebook_dim if self.codebook_dim is not None else self.d_model

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Config":
        known = {f.name for f in fields(Config)}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"Unknown config keys: {sorted(unknown)}")
        return Config(**d)


# Overrides applied by `--smoke`: tiny, CPU-friendly, non-gated tokenizer, high LR so that
# ~60 steps visibly move the reconstruction loss (the smoke test validates wiring, not fidelity).
SMOKE_OVERRIDES: Dict[str, Any] = dict(
    tokenizer_name="gpt2",
    data_fixture="tests/fixtures/metamath_smoke.jsonl",  # offline; auto-generated if missing
    device="cpu",
    amp=False,
    max_seq=64,
    compression_rate=8,
    chunk_length=8,
    d_model=64,
    n_layers=2,
    n_heads=4,
    codebook_size=64,
    vq_decay=0.95,
    lr=3e-4,
    batch_size=16,
    steps=60,
    warmup=0,
    n_val=16,
    num_workers=0,
    log_every=5,
    eval_every=30,
    ckpt_every=60,
    mlflow=False,                # a wiring check never writes into the tracking store
)


def load_config(path: str) -> Config:
    with open(path, "r") as f:
        raw = yaml.safe_load(f) or {}
    return Config.from_dict(raw)


def apply_overrides(cfg: Config, overrides: Dict[str, Any]) -> Config:
    """Return a new Config with `overrides` applied (None values ignored)."""
    d = cfg.to_dict()
    for k, v in overrides.items():
        if v is not None and k in d:
            d[k] = v
    return Config.from_dict(d)


# --------------------------------------------------------------------------------------
# Reproducibility & device
# --------------------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    """DataLoader worker seeding for reproducibility."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


CUBLAS_WORKSPACE_CONFIG = ":4096:8"
_DETERMINISTIC_CUBLAS_CONFIGS = (":4096:8", ":16:8")


def apply_determinism(warn_only: bool = False) -> None:
    """Make Stage-1 training bit-reproducible (``deterministic: true``, the default).

    Seeds alone are not enough on GPU: these Stage-1 CUDA kernels reorder float atomics run to run
    — vq-pytorch's kmeans-init ``scatter_add_``, the ``repeat_interleave`` backward of the parallel
    decoder's up-sampler, SDPA backward (flash / mem-efficient / cuDNN attention), cuDNN conv
    backward, and cuBLAS GEMMs without a pinned workspace. This switches all of them:

    * ``CUBLAS_WORKSPACE_CONFIG`` — pinned GEMM workspace -> fixed reduction order. torch refuses
      deterministic mode on CUDA without ``:4096:8`` (~32 MiB, fast) or ``:16:8`` (low-memory). A
      preset value is respected if it is one of those two, and rejected otherwise.
    * ``torch.use_deterministic_algorithms(True)`` — swaps the atomicAdd kernels for deterministic
      ones and RAISES on any op that has none (the raise is the feature: a silent fallback would
      leave the run irreproducible while its config claims otherwise). ``warn_only`` downgrades the
      raise to a warning — and also leaves SDPA backward nondeterministic, so it is a debugging
      escape hatch, not a reproducible mode.
    * cuDNN (and oneDNN, for CPU runs) deterministic algorithms, cuDNN autotuning off.

    MUST run before CUDA is initialized: torch and cuBLAS read ``CUBLAS_WORKSPACE_CONFIG`` once, at
    the first matmul (``assert_gpu_ok`` runs one), so ``train.main()`` calls this right after the
    config resolves. Importing torch is harmless — it never touches cuBLAS — hence the guard is
    "CUDA not initialized" rather than Stage 2's "torch not imported". It raises before changing
    anything, so a refusal never leaves the setup half-applied.

    Seeds are NOT set here: ``set_seed(cfg.seed)`` runs unconditionally right after this call.
    Cost: ~2-11% per step (deterministic kernels give up atomics) plus ~32 MiB VRAM. On by default so
    that any two runs of one config are comparable bit for bit; opt out with ``deterministic: false``
    / ``--no-deterministic`` only when speed matters more than exact repeatability.
    """
    if torch.cuda.is_initialized():
        raise RuntimeError(
            "apply_determinism() must run before CUDA is initialized: cuBLAS reads "
            "CUBLAS_WORKSPACE_CONFIG once, at its first matmul. Call it right after the config "
            "resolves (train.main() does), before select_device()/assert_gpu_ok()."
        )
    preset = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if preset is not None and preset not in _DETERMINISTIC_CUBLAS_CONFIGS:
        raise RuntimeError(
            f"CUBLAS_WORKSPACE_CONFIG={preset!r} is set but is not a deterministic cuBLAS config "
            f"({' or '.join(_DETERMINISTIC_CUBLAS_CONFIGS)}). Unset it or pick one of those."
        )
    if preset is None:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = CUBLAS_WORKSPACE_CONFIG
    torch.use_deterministic_algorithms(True, warn_only=warn_only)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if hasattr(torch.backends, "mkldnn") and hasattr(torch.backends.mkldnn, "deterministic"):
        torch.backends.mkldnn.deterministic = True
    print(f"[determinism] ON — CUBLAS_WORKSPACE_CONFIG={os.environ['CUBLAS_WORKSPACE_CONFIG']}"
          f"{' (pre-set in the environment)' if preset else ''}, warn_only={warn_only}. "
          f"Opt out with --no-deterministic for speed.")


def select_device(pref: str = "auto") -> torch.device:
    if pref != "auto":
        return torch.device(pref)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def assert_gpu_ok() -> None:
    """Verify the CUDA wheel actually has kernels for this GPU.

    Blackwell (RTX PRO 6000) reports compute capability 12.x. The cu128 wheels embed
    sm_120 SASS + compute_120 PTX, but the workstation Pro 6000 has been reported as both
    sm_120 and sm_122 in the wild, so we *launch a real kernel* rather than trust the arch
    list. A "no kernel image is available for execution on the device" error here means the
    torch wheel does not cover this card -> bump torch in requirements.txt (see its header).
    """
    assert torch.cuda.is_available(), "CUDA not available."
    cap = torch.cuda.get_device_capability()
    archs = torch.cuda.get_arch_list()
    name = torch.cuda.get_device_name()
    x = torch.randn(8, 8, device="cuda")
    _ = (x @ x).sum().item()  # raises here if no kernel image for this card
    print(f"[gpu] {name} cap={cap} arch_list={archs} — kernel launch OK")
    if cap[0] < 12:
        print(f"[gpu] warning: expected Blackwell (>=12.x), got {cap}")


def count_params(model: torch.nn.Module, trainable_only: bool = False) -> int:
    return sum(
        p.numel() for p in model.parameters() if (p.requires_grad or not trainable_only)
    )


# --------------------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------------------
@torch.no_grad()
def reconstruction_accuracy(
    logits: torch.Tensor, targets: torch.Tensor, ignore_index: int
) -> float:
    """Teacher-forced token accuracy over non-pad positions.

    logits: (B, T, V), targets: (B, T). Returns the fraction of non-ignored positions where
    argmax(logits) == target.
    """
    pred = logits.argmax(dim=-1)
    valid = targets != ignore_index
    if valid.sum() == 0:
        return 0.0
    correct = (pred == targets) & valid
    return (correct.sum().float() / valid.sum().float()).item()


@torch.no_grad()
def code_usage_counts(
    indices: torch.Tensor, codebook_size: int, mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Bincount of code usage over the batch. indices: (B, N) long; mask: (B, N) bool
    (True = real code). Returns a (codebook_size,) float count vector."""
    idx = indices.reshape(-1)
    if mask is not None:
        idx = idx[mask.reshape(-1)]
    return torch.bincount(idx, minlength=codebook_size).float()


@torch.no_grad()
def codebook_metrics_from_counts(counts: torch.Tensor) -> Dict[str, float]:
    """Usage statistics from a pre-accumulated (codebook_size,) count vector.

    Returns active-code fraction and codebook perplexity = exp(entropy of the empirical code
    distribution). Perplexity near 1 means collapse onto a single code; near `codebook_size`
    means uniform usage. Counts may span one batch or a whole corpus.
    """
    total = counts.sum()
    if total == 0:
        return {"active_fraction": 0.0, "perplexity": 0.0, "n_codes_counted": 0}
    probs = counts / total
    nz = probs > 0
    entropy = -(probs[nz] * probs[nz].log()).sum()
    return {
        "active_fraction": nz.sum().item() / counts.numel(),
        "perplexity": entropy.exp().item(),
        "n_codes_counted": int(total.item()),
    }


@torch.no_grad()
def codebook_metrics(
    indices: torch.Tensor, codebook_size: int, mask: Optional[torch.Tensor] = None
) -> Dict[str, float]:
    """Per-batch codebook usage statistics to flag collapse.

    NOTE: per-batch active fraction has a ceiling of (valid codes in batch) / codebook_size
    (~0.4-0.5 at batch 32 x 16 codes over |E|=1024); use the corpus-level `global_*` val
    metrics (accumulated counts -> `codebook_metrics_from_counts`) to judge collapse.
    """
    return codebook_metrics_from_counts(code_usage_counts(indices, codebook_size, mask))


# --------------------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------------------
class MetricLogger:
    """Console + JSONL + CSV metric logging, with optional Weights & Biases and MLflow.

    wandb / mlflow are imported lazily and only when enabled. mlflow is enabled by `mlflow: true` in the
    config (the shipped configs set it; `--mlflow` / `--no-mlflow` override) and is listed in
    requirements.txt; wandb stays opt-in.
    """

    def __init__(
        self,
        log_dir: str,
        use_wandb: bool = False,
        wandb_kwargs: Optional[dict] = None,
        use_mlflow: bool = False,
        mlflow_kwargs: Optional[dict] = None,
    ):
        os.makedirs(log_dir, exist_ok=True)
        self.log_dir = log_dir
        self.jsonl_path = os.path.join(log_dir, "metrics.jsonl")
        self.csv_path = os.path.join(log_dir, "metrics.csv")
        self._csv_keys: Optional[list] = None
        self.use_wandb = use_wandb
        self._wandb = None
        self.use_mlflow = use_mlflow
        self._mlflow = None
        self.mlflow_run_id: Optional[str] = None
        if use_wandb:
            import wandb  # type: ignore

            self._wandb = wandb
            wandb.init(**(wandb_kwargs or {}))
        if use_mlflow:
            import mlflow  # type: ignore

            mk = mlflow_kwargs or {}
            if mk.get("tracking_uri"):
                mlflow.set_tracking_uri(mk["tracking_uri"])
            mlflow.set_experiment(mk.get("experiment_name", "Default"))
            run = mlflow.start_run(run_name=mk.get("run_name"))
            self._mlflow = mlflow
            # Persist the run id so post-training eval (src.roundtrip_eval) can attach its metrics
            # to THIS run — mirrors what Stage-2 training writes for src.stage2_eval.
            self.mlflow_run_id = run.info.run_id
            with open(os.path.join(log_dir, "mlflow_run_id.txt"), "w") as f:
                f.write(self.mlflow_run_id)

    def log_params(self, params: Dict[str, Any]) -> None:
        """Log hyperparameters once (no-op unless MLflow is enabled)."""
        if self._mlflow is not None:
            self._mlflow.log_params({k: _param_value(v) for k, v in params.items()})

    def log_artifact(self, path: str, artifact_path: Optional[str] = None) -> None:
        """Upload a single file to the MLflow artifact store (no-op otherwise).

        artifact_path nests the file under a subdirectory in the run's artifact tree.
        """
        if self._mlflow is not None and os.path.exists(path):
            self._mlflow.log_artifact(path, artifact_path=artifact_path)

    def log(self, step: int, metrics: Dict[str, float], split: str = "train") -> None:
        row = {"step": step, "split": split, **{k: _to_float(v) for k, v in metrics.items()}}
        with open(self.jsonl_path, "a") as f:
            f.write(json.dumps(row) + "\n")
        self._write_csv(row)
        if self._wandb is not None:
            self._wandb.log({f"{split}/{k}": v for k, v in metrics.items()}, step=step)
        if self._mlflow is not None:
            self._mlflow.log_metrics(
                {f"{split}/{k}": _to_float(v) for k, v in metrics.items()}, step=step
            )
        pretty = "  ".join(f"{k}={_fmt(v)}" for k, v in metrics.items())
        print(f"[{split:>5} step {step:>7}] {pretty}", flush=True)

    def _write_csv(self, row: Dict[str, Any]) -> None:
        # CSV header is fixed on first write; new keys later are dropped from the CSV
        # (still present in metrics.jsonl). Keeps the CSV rectangular.
        if self._csv_keys is None:
            self._csv_keys = list(row.keys())
            with open(self.csv_path, "w", newline="") as f:
                csv.writer(f).writerow(self._csv_keys)
        with open(self.csv_path, "a", newline="") as f:
            csv.writer(f).writerow([row.get(k, "") for k in self._csv_keys])

    def finish(self) -> None:
        if self._mlflow is not None:
            # config + best checkpoint + metric files (last.pt intentionally excluded)
            for fn in ("config.resolved.yaml", "best.pt"):
                self.log_artifact(os.path.join(self.log_dir, fn))
            self.log_artifact(self.jsonl_path)
            self.log_artifact(self.csv_path)
            self._mlflow.end_run()
        if self._wandb is not None:
            self._wandb.finish()


def _param_value(v: Any) -> Any:
    # MLflow params are stringified scalars; json-encode containers so they round-trip.
    if isinstance(v, (list, tuple, dict)):
        return json.dumps(v)
    return v


def _to_float(v: Any) -> Any:
    if isinstance(v, torch.Tensor):
        return v.item()
    return v


def _fmt(v: Any) -> str:
    if isinstance(v, torch.Tensor):
        v = v.item()
    if isinstance(v, float):
        # Tiny values (e.g. lr=1e-5) round to 0.0000 under %.4f and read as zero;
        # show those in scientific notation so the real magnitude is preserved.
        if v != 0.0 and abs(v) < 1e-4:
            return f"{v:.3e}"
        return f"{v:.4f}"
    return str(v)
