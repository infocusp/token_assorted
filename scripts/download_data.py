"""Optionally pre-download / snapshot meta-math/MetaMathQA so later runs are offline.

`load_dataset` already caches under ~/.cache/huggingface on first use, so this is convenience:
it triggers the download now and (optionally) saves an Arrow snapshot to `--out`.

Usage:
    python scripts/download_data.py --out ./data
"""
from __future__ import annotations

import argparse
import os


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="meta-math/MetaMathQA")
    ap.add_argument("--split", default="train")
    ap.add_argument("--out", default="./data", help="Directory for the Arrow snapshot.")
    ap.add_argument("--no-save", action="store_true", help="Only warm the HF cache; don't save_to_disk.")
    args = ap.parse_args()

    from datasets import load_dataset

    print(f"Loading {args.dataset} [{args.split}] (downloads + caches on first use)...")
    ds = load_dataset(args.dataset, split=args.split)
    print(f"  rows={len(ds)} columns={ds.column_names}")
    print(f"  example response tail: ...{ds[0]['response'][-60:]!r}")

    if not args.no_save:
        out = os.path.join(args.out, args.dataset.replace("/", "__"))
        os.makedirs(out, exist_ok=True)
        ds.save_to_disk(out)
        print(f"Saved Arrow snapshot -> {out}")


if __name__ == "__main__":
    main()
