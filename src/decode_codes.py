"""Decode latent code indices back into text — a sanity/interpretability round-trip.

Loads the JSONL artifact produced by `encode_dataset.py` and, for a few examples, reconstructs
the chain-of-thought text from the code indices (optionally conditioned on the prompt g(P)).
Reconstruction quality is the primary health metric: if the CoT cannot be
reconstructed, the codes are not meaningful. Records with `cot_token_ids` (schema v2) also get the
gold CoT printed next to the reconstruction for a direct comparison.

Example:
    python -m src.decode_codes --checkpoint runs/<run>/best.pt \
        --codes ./data/metamath_latent_codes.jsonl --num-examples 5
"""
from __future__ import annotations

import argparse
import json

import torch

from .model import load_checkpoint
from .utils import select_device


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--codes", required=True, help="JSONL from encode_dataset.py")
    p.add_argument("--num-examples", type=int, default=5)
    p.add_argument("--device", default="auto")
    p.add_argument("--no-prompt", action="store_true",
                   help="Decode without conditioning on the prompt g(P).")
    return p


def _read_jsonl(path: str, limit: int):
    out = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
            if len(out) >= limit:
                break
    return out


@torch.no_grad()
def prompt_embedding(model, tokenizer, prompt: str, device, max_seq: int) -> torch.Tensor:
    """g(P) for decoding: mean of prompt token embeddings -> (1, d)."""
    ids = tokenizer(prompt, add_special_tokens=False, truncation=True,
                    max_length=max_seq).input_ids
    if len(ids) == 0:
        return torch.zeros(1, model.cfg.d_model, device=device)
    p = torch.tensor([ids], device=device)
    return model.tok_embed(p).mean(dim=1)


@torch.no_grad()
def main() -> None:
    args = build_argparser().parse_args()
    device = select_device(args.device)
    model, tokenizer, info, cfg = load_checkpoint(args.checkpoint, device)
    records = _read_jsonl(args.codes, args.num_examples)
    print(f"[decode] {len(records)} examples | mode={cfg.decoder_mode} "
          f"tokenizer={tokenizer.name_or_path}\n")

    for rec in records:
        codes = rec["code_indices"]
        if len(codes) == 0:
            print(f"--- id={rec['id']} (no codes; skipped)\n")
            continue
        indices = torch.tensor([codes], dtype=torch.long, device=device)
        gP = None if args.no_prompt else prompt_embedding(model, tokenizer, rec["prompt"], device, cfg.max_seq)
        out_ids = model.decode(indices, gP=gP)
        text = tokenizer.decode(out_ids[0].tolist(), skip_special_tokens=True)

        print(f"--- id={rec['id']}  num_codes={rec['num_codes']}  num_cot_tokens={rec['num_cot_tokens']}")
        print(f"  prompt        : {rec['prompt'][:160]}")
        print(f"  answer (gold) : {rec['answer'][:120]}")
        gold_ids = rec.get("cot_token_ids")
        if gold_ids:
            print(f"  cot (gold)    : {tokenizer.decode(gold_ids, skip_special_tokens=True)[:400]}")
        print(f"  reconstructed : {text[:400]}")
        print()


if __name__ == "__main__":
    main()
