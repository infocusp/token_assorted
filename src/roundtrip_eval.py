"""Post-training round-trip eval for the Stage-1 VQ-VAE (honest code-quality signal).

Encodes held-out validation CoT with the trained encoder and decodes it back with the trained
decoder — free autoregressive rollout for ``decoder_mode=ar``, a single argmax pass for
``parallel`` — then measures how faithfully the discrete codes reconstruct the chain-of-thought.

Why this exists: the ``recon_acc`` logged during training is *teacher-forced* (the AR decoder sees
ground-truth previous tokens) when using AR decoder, which massively inflates it and is not how
inference works. This eval runs the decoder the way it is actually used, so it is the trustworthy
comparison across runs and decoder modes (the AR runs looked best on ``recon_acc`` yet reconstruct
far worse here).

Metrics are alignment-free so parallel and AR are comparable:
  * ``token_f1``         — multiset (bag-of-token) F1 of decoded vs gold CoT token ids.
  * ``number_preserved`` — fraction of the DISTINCT numbers in the gold CoT that survive the round
                           trip. Numbers are the crux for math; a code set that drops them is useless
                           downstream regardless of how fluent the reconstruction reads.
  * ``dec_len``/``gold_len`` — mean token lengths; a ``dec_len`` far above ``gold_len`` flags AR
                           rollout degeneration (runaway repetition).

Val split is the SAME deterministic tail-holdout as training (last ``n_val`` rows), so this never
peeks at training data. Metrics attach to the training MLflow run via ``<run_dir>/mlflow_run_id.txt``.

Usage:
    python -m src.roundtrip_eval --config config/vqvae_llama.yaml \
        --num-examples 100 --mlflow --mlflow-tracking-uri sqlite:///mlflow.db
    python -m src.roundtrip_eval --run-dir runs/vqvae-llama --num-examples 100
"""
from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter
from typing import Dict, List, Optional, Tuple

import torch
import yaml

from .data import assemble_example, deterministic_train_val_split, load_rows
from .decode_codes import prompt_embedding
from .encode_dataset import cot_code_span
from .model import load_checkpoint
from .utils import select_device

_NUM_RE = re.compile(r"-?\d+\.?\d*")


def _numbers(text: str) -> set:
    return set(_NUM_RE.findall(text))


@torch.no_grad()
def roundtrip_metrics(
    model, tokenizer, info, cfg, val_rows, num_examples: int, num_show: int, device
) -> Tuple[Dict[str, float], List[Tuple[str, str]]]:
    """Encode->decode each val example's CoT and score the reconstruction. See module docstring."""
    r = cfg.compression_rate
    f1s: List[float] = []
    num_rec: List[float] = []
    dlen: List[int] = []
    glen: List[int] = []
    examples: List[Tuple[str, str]] = []
    seen = i = 0
    while seen < num_examples and i < len(val_rows):
        row = val_rows[i]
        i += 1
        ex = assemble_example(row["query"], row["response"], tokenizer, info, cfg.max_seq)
        cs, ce = ex["c_start"], ex["c_end"]
        if ce <= cs:
            continue  # empty CoT (e.g. answer-only response) — nothing to reconstruct
        k0, k1 = cot_code_span(cs, ce, r, cfg.n_codes)
        if k1 <= k0:
            continue
        ids = ex["input_ids"].unsqueeze(0).to(device)
        attn = ex["attention_mask"].unsqueeze(0).to(device)
        idx, _, _ = model.encode(ids, attn)                     # this run's OWN codes
        gP = prompt_embedding(model, tokenizer, row["query"], device, cfg.max_seq)
        out = model.decode(idx[:, k0:k1], gP=gP)                # free rollout (ar) / argmax (parallel)
        dec_ids = out[0].tolist()
        gold_ids = ex["input_ids"][cs:ce].tolist()
        dec_text = tokenizer.decode(dec_ids, skip_special_tokens=True)
        gold_text = tokenizer.decode(gold_ids, skip_special_tokens=True)

        dc, gc = Counter(dec_ids), Counter(gold_ids)
        inter = sum((dc & gc).values())
        prec = inter / max(1, sum(dc.values()))
        rec = inter / max(1, sum(gc.values()))
        f1s.append(2 * prec * rec / max(1e-9, prec + rec))
        dlen.append(len(dec_ids))
        glen.append(len(gold_ids))
        gnums = _numbers(gold_text)
        if gnums:
            num_rec.append(len(gnums & _numbers(dec_text)) / len(gnums))
        if len(examples) < num_show:
            examples.append((gold_text, dec_text))
        seen += 1

    n = max(1, len(f1s))
    metrics = {
        "token_f1": sum(f1s) / n,
        "number_preserved": sum(num_rec) / max(1, len(num_rec)),
        "dec_len": sum(dlen) / n,
        "gold_len": sum(glen) / n,
        "num_examples": float(len(f1s)),
    }
    return metrics, examples


def _resolve_run_dir(args) -> str:
    """Run dir = explicit --run-dir, else <out_dir>/<run_name> from the --config YAML."""
    if args.run_dir:
        return args.run_dir
    cfg = yaml.safe_load(open(args.config))
    return os.path.join(cfg.get("out_dir", "runs"), cfg["run_name"])


def log_metrics_to_mlflow(args, run_dir: str, metrics: Dict[str, float]) -> None:
    """Attach roundtrip/* metrics to the training run (via <run_dir>/mlflow_run_id.txt) or a fresh
    run. Mirrors src.stage2_eval.log_metrics_to_mlflow; the ``roundtrip/`` prefix keeps these clear
    of training's ``train/*`` and ``val/*`` curves."""
    import mlflow

    if args.mlflow_tracking_uri:
        mlflow.set_tracking_uri(args.mlflow_tracking_uri)

    run_id = args.mlflow_run_id
    if run_id is None:
        rid_path = os.path.join(run_dir, "mlflow_run_id.txt")
        if os.path.exists(rid_path):
            with open(rid_path) as f:
                run_id = f.read().strip() or None

    payload = {f"roundtrip/{k}": float(v) for k, v in metrics.items()}
    if run_id:
        try:
            with mlflow.start_run(run_id=run_id):
                mlflow.log_metrics(payload)
            print(f"[roundtrip] logged metrics to training run {run_id}")
            return
        except Exception as e:  # stale/missing run id -> standalone run
            print(f"[roundtrip] could not attach to run {run_id} ({e}); logging a new run instead")
    mlflow.set_experiment(args.mlflow_experiment)
    with mlflow.start_run(run_name=f"roundtrip-{os.path.basename(run_dir.rstrip('/'))}"):
        mlflow.log_params({"run_dir": run_dir})
        mlflow.log_metrics(payload)
    print("[roundtrip] logged metrics to a new MLflow run")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--config", help="Stage-1 config; run dir = <out_dir>/<run_name>.")
    g.add_argument("--run-dir", dest="run_dir",
                   help="Explicit runs/<name> dir (use when training overrode --run-name).")
    p.add_argument("--num-examples", dest="num_examples", type=int, default=100)
    p.add_argument("--num-show", dest="num_show", type=int, default=4,
                   help="How many gold/reconstruction pairs to print.")
    p.add_argument("--device", default="auto")
    p.add_argument("--smoke", action="store_true", help="Tiny check: 8 examples on CPU.")
    p.add_argument("--out", default=None, help="Optional JSON path for the metrics.")
    p.add_argument("--mlflow", action="store_true", default=False)
    p.add_argument("--mlflow-tracking-uri", dest="mlflow_tracking_uri", default=None)
    p.add_argument("--mlflow-experiment", dest="mlflow_experiment", default="stage1(VQ-VAE)")
    p.add_argument("--mlflow-run-id", dest="mlflow_run_id", default=None,
                   help="Override; else read <run_dir>/mlflow_run_id.txt; else a fresh run.")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    run_dir = _resolve_run_dir(args)
    ckpt = os.path.join(run_dir, "best.pt")
    device = select_device("cpu" if args.smoke else args.device)
    model, tok, info, cfg = load_checkpoint(ckpt, device)
    model.eval()

    rows = load_rows(cfg.dataset_name, cfg.dataset_split, cfg.data_fixture)
    _, val_rows = deterministic_train_val_split(rows, cfg.n_val)
    n = 8 if args.smoke else args.num_examples
    metrics, examples = roundtrip_metrics(model, tok, info, cfg, val_rows, n, args.num_show, device)

    print(f"[roundtrip] {run_dir}  decoder_mode={cfg.decoder_mode}  "
          f"n={int(metrics['num_examples'])} (of {n} requested)")
    print(f"[roundtrip] token_f1={metrics['token_f1']:.3f}  "
          f"number_preserved={metrics['number_preserved'] * 100:.1f}%  "
          f"dec_len={metrics['dec_len']:.1f}  gold_len={metrics['gold_len']:.1f}")
    for k, (gold, dec) in enumerate(examples):
        print(f"\n[example {k}] GOLD : {gold[:240]}")
        print(f"[example {k}] RECON: {dec[:240]}")

    if args.out:
        with open(args.out, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"[roundtrip] wrote {args.out}")
    if args.mlflow:
        log_metrics_to_mlflow(args, run_dir, metrics)


if __name__ == "__main__":
    main()
