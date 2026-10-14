"""Encode the chain-of-thought of each MetaMathQA example into latent code indices.

**This file is the Stage-1 -> Stage-2 hand-off boundary.** The trained VQ-VAE is run over the
prompt+CoT, and the discrete code indices covering the C span are written to disk. In Stage 2
(out of scope) these indices replace the first part of each reasoning trace when fine-tuning an LLM.

The VQ-VAE was *trained* on the full X = P (+) C (+) S, but here we *apply* it to C (paper §3.1).
To avoid train/serve skew (the encoder is causal, so a C-chunk's latent was trained with P as
left-context), we feed P (+) C through the encoder and keep only the latent chunks that overlap
the C span. `--compress-target full` keeps all codes over the real (non-pad) span instead.

Each output record (JSONL):
    {"id", "prompt", "c_start", "num_cot_tokens", "cot_token_ids": [...],
     "num_codes", "code_indices": [...], "answer"}

`cot_token_ids` (= input_ids[c_start:c_end], i.e. the post-truncation C span the VQ-VAE saw; never
contains PAD/EOS) and `c_start` make each record self-contained for Stage 2: mixed latent/text
sequences can be rebuilt without re-deriving the tokenization, `c_end = c_start +
len(cot_token_ids)`, and the first code's chunk is `k0 = c_start // r`. The invariant
`len(cot_token_ids) == num_cot_tokens` holds in `cot` mode (the mode Stage 2 consumes); in `full`
mode `num_cot_tokens` counts the whole real (non-pad) span instead. A `<out>.meta.json` sidecar
records the checkpoint/tokenizer/config that produced the artifact.

Example:
    python -m src.encode_dataset --checkpoint runs/<run>/best.pt \
        --split train --compress-target cot --out ./data/metamath_latent_codes.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone

import torch
from tqdm import tqdm

from .data import assemble_example, load_rows, split_cot_solution
from .model import load_checkpoint
from .utils import select_device


def cot_code_span(c_start: int, c_end: int, r: int, n_codes: int):
    """Half-open range [k0, k1) of code positions whose chunks overlap the C token span."""
    if c_end <= c_start:
        return 0, 0
    k0 = max(0, c_start // r)
    k1 = min(n_codes, (c_end - 1) // r + 1)
    return k0, max(k0, k1)


def build_record(rec_id: int, query: str, response: str, ex: dict, indices_row: torch.Tensor,
                 valid_row: torch.Tensor, target: str, r: int, n_codes: int) -> dict:
    """One output record. `ex` tensors live on CPU; `indices_row`/`valid_row` may be on GPU."""
    c_start, c_end = ex["c_start"], ex["c_end"]
    if target == "cot":
        k0, k1 = cot_code_span(c_start, c_end, r, n_codes)
        sel = torch.zeros(n_codes, dtype=torch.bool, device=valid_row.device)
        sel[k0:k1] = True
        sel &= valid_row
        num_text = c_end - c_start
    else:  # full: all valid codes over the real span
        sel = valid_row
        num_text = int(ex["attention_mask"].sum().item())
    code_indices = indices_row[sel].tolist()
    _, s_text = split_cot_solution(response)
    return {
        "id": rec_id,
        "prompt": query,
        "c_start": int(c_start),
        "num_cot_tokens": int(num_text),
        "cot_token_ids": ex["input_ids"][c_start:c_end].tolist(),
        "num_codes": len(code_indices),
        "code_indices": code_indices,
        "answer": s_text.strip(),
    }


def build_meta(args, cfg, info, tokenizer, target: str, dataset_name: str, n_written: int) -> dict:
    """Provenance sidecar: which checkpoint/tokenizer/config produced the artifact."""
    return {
        "schema_version": 2,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "checkpoint": os.path.abspath(args.checkpoint),
        "tokenizer_name": tokenizer.name_or_path,
        "dataset_name": dataset_name,
        "split": args.split,
        "data_fixture": args.data_fixture,
        "compress_target": target,
        "max_seq": cfg.max_seq,
        "compression_rate": cfg.compression_rate,
        "chunk_length": cfg.chunk_length,
        "n_codes": cfg.n_codes,
        "codebook_size": cfg.codebook_size,
        "pad_id": info.pad_id,
        "eos_id": info.eos_id,
        "limit": args.limit,
        "n_records": n_written,
    }


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--split", default="train")
    p.add_argument("--compress-target", dest="compress_target", default=None,
                   choices=["cot", "full"], help="Override the checkpoint's compress_target.")
    p.add_argument("--out", default="./data/metamath_latent_codes.jsonl")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", default="auto")
    p.add_argument("--data-fixture", default=None, help="Encode an offline JSONL fixture instead.")
    p.add_argument("--dataset-name", default=None, help="Override dataset name (default from ckpt).")
    return p


@torch.no_grad()
def main() -> None:
    args = build_argparser().parse_args()
    device = select_device(args.device)
    model, tokenizer, info, cfg = load_checkpoint(args.checkpoint, device)
    target = args.compress_target or cfg.compress_target
    r, n_codes, max_seq = cfg.compression_rate, cfg.n_codes, cfg.max_seq
    dataset_name = args.dataset_name or cfg.dataset_name
    print(f"[encode] target={target} r={r} n_codes={n_codes} max_seq={max_seq} "
          f"tokenizer={tokenizer.name_or_path} device={device}")

    rows = load_rows(dataset_name, args.split, args.data_fixture, limit=args.limit)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    n_written = 0
    with open(args.out, "w") as fout:
        for start in tqdm(range(0, len(rows), args.batch_size), desc="encoding"):
            chunk = [rows[i] for i in range(start, min(start + args.batch_size, len(rows)))]
            exs = [assemble_example(rw["query"], rw["response"], tokenizer, info, max_seq) for rw in chunk]
            input_ids = torch.stack([e["input_ids"] for e in exs]).to(device)
            attn = torch.stack([e["attention_mask"] for e in exs]).to(device)
            indices, chunk_valid, _ = model.encode(input_ids, attn)  # (B, n_codes)

            for j, (rw, ex) in enumerate(zip(chunk, exs)):
                rec = build_record(start + j, rw["query"], rw["response"], ex,
                                   indices[j], chunk_valid[j], target, r, n_codes)
                fout.write(json.dumps(rec) + "\n")
                n_written += 1

    meta_path = args.out + ".meta.json"
    with open(meta_path, "w") as fmeta:
        json.dump(build_meta(args, cfg, info, tokenizer, target, dataset_name, n_written),
                  fmeta, indent=2)

    print(f"[encode] wrote {n_written} records -> {args.out} (+ {meta_path})")


if __name__ == "__main__":
    main()
