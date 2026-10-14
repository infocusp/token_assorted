"""Sample a fixed-seed subset of nvidia/OpenMathInstruct-2 restricted to two problem sources.

Streams the requested split (`load_dataset(..., streaming=True)` — no full local
materialization; the HF cache is reused if the shards were already downloaded), keeps only rows
whose `problem_source` is `augmented_gsm8k` or `augmented_math`, and draws a uniform random
sample of each source via reservoir sampling. The per-source sizes are set so that
n_gsm8k / n_math == --ratio (default 0.2359375). Deterministic for a fixed --seed: the stream
order is fixed and a single seeded RNG drives both reservoirs.

Output: JSONL with the original record fields (problem, generated_solution, expected_answer,
problem_source), shuffled with the same seed, plus a `.meta.json` sidecar recording the
parameters and achieved counts.

Usage:
    python scripts/sample_openmathinstruct2.py                       # 100k, seed 0, train_1M
    python scripts/sample_openmathinstruct2.py --n 20000 --seed 1 --ratio 0.5
    python scripts/sample_openmathinstruct2.py --split train_5M --out data/my_subset.jsonl
"""
from __future__ import annotations

import argparse
import json
import random

GSM8K_SOURCE = "augmented_gsm8k"
MATH_SOURCE = "augmented_math"


def source_targets(n: int, ratio: float) -> dict:
    """Split `n` into per-source counts with n_gsm8k / n_math == ratio (rounded)."""
    n_gsm8k = round(n * ratio / (1.0 + ratio))
    return {GSM8K_SOURCE: n_gsm8k, MATH_SOURCE: n - n_gsm8k}


def reservoir_sample(stream, targets: dict, rng: random.Random,
                     log_every: int = 200_000) -> dict:
    """One pass over `stream`, keeping a uniform random `targets[src]`-sized reservoir per source."""
    reservoirs = {src: [] for src in targets}
    seen = {src: 0 for src in targets}
    scanned = 0
    for row in stream:
        scanned += 1
        if log_every and scanned % log_every == 0:
            kept = {s: len(r) for s, r in reservoirs.items()}
            print(f"[sample] scanned {scanned:,} rows; matched {seen}; reservoirs {kept}")
        src = row.get("problem_source")
        if src not in targets:
            continue
        seen[src] += 1
        reservoir, k = reservoirs[src], targets[src]
        if len(reservoir) < k:
            reservoir.append(row)
        else:
            j = rng.randrange(seen[src])
            if j < k:
                reservoir[j] = row
    print(f"[sample] done: scanned {scanned:,} rows; matched per source {seen}")
    return reservoirs


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=100_000,
                   help="Total number of samples to draw (default 100000).")
    p.add_argument("--ratio", type=float, default=0.2359375,
                   help="Target n_augmented_gsm8k / n_augmented_math (default 0.2359375).")
    p.add_argument("--seed", type=int, default=0,
                   help="RNG seed for reservoir sampling and the final shuffle (default 0).")
    p.add_argument("--split", default="train_1M",
                   help="OpenMathInstruct-2 split to stream (default train_1M; "
                        "also train_2M, train_5M, train).")
    p.add_argument("--out", default=None,
                   help="Output JSONL path (default data/omi2_<split>_n<n>_seed<seed>.jsonl).")
    args = p.parse_args()

    from datasets import load_dataset

    out = args.out or f"data/omi2_{args.split}_n{args.n}_seed{args.seed}.jsonl"
    targets = source_targets(args.n, args.ratio)
    print(f"[sample] split={args.split} seed={args.seed} targets={targets} "
          f"(requested ratio {args.ratio})")

    stream = load_dataset("nvidia/OpenMathInstruct-2", split=args.split, streaming=True)
    rng = random.Random(args.seed)
    reservoirs = reservoir_sample(stream, targets, rng)

    for src, target in targets.items():
        got = len(reservoirs[src])
        assert got == target, (
            f"split {args.split!r} has only {got} rows with problem_source={src!r} "
            f"but {target} were requested — lower --n or change --ratio/--split"
        )

    rows = reservoirs[GSM8K_SOURCE] + reservoirs[MATH_SOURCE]
    rng.shuffle(rows)
    with open(out, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    achieved = targets[GSM8K_SOURCE] / max(targets[MATH_SOURCE], 1)
    meta = {
        "dataset": "nvidia/OpenMathInstruct-2",
        "split": args.split,
        "n": args.n,
        "seed": args.seed,
        "requested_ratio": args.ratio,
        "achieved_ratio": achieved,
        "counts": targets,
        "sampling": "per-source reservoir (uniform), single pass, streaming",
    }
    with open(out + ".meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[sample] wrote {len(rows):,} rows -> {out} "
          f"(gsm8k/math = {achieved:.7f}); meta -> {out}.meta.json")


if __name__ == "__main__":
    main()
