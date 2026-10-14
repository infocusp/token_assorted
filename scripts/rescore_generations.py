"""Re-score a saved generations JSONL with the official scorer — no GPU, model or benchmark reload.

`src/stage2_eval.py --save-generations` writes one self-contained record per example (the full
benchmark row plus the raw generation), so a scorer can be changed or fixed and every past run
re-graded for free. That is the gap this closes: before 2026-09-20 only aggregate metrics were
kept, so an evaluator change meant re-running generation on every checkpoint.

It is also the way to bring an `accuracy_official` recorded by an older version of a scorer onto
the current scorer's scale (see `third_party/official_evals/PROVENANCE.md`).

Usage:
    python scripts/rescore_generations.py runs/<adapter>/eval_theoremqa_generations.jsonl
    python scripts/rescore_generations.py <file.jsonl> --dataset theoremqa --out rescored.json
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.official_eval import SCORER_SOURCE, availability, score  # noqa: E402


def _infer_dataset(path: str) -> str:
    """`.../eval_<dataset>[_suffixes]_generations.jsonl` -> `<dataset>`."""
    base = os.path.basename(path)
    for prefix, suffix in (("eval_", "_generations.jsonl"),):
        if base.startswith(prefix) and base.endswith(suffix):
            stem = base[len(prefix):-len(suffix)]
            # strip the _n<limit>/_seed<seed>/_mnt<budget> suffixes eval_stage2_all.sh appends
            for sep in ("_n", "_seed", "_mnt"):
                if sep in stem:
                    stem = stem.split(sep)[0]
            return stem
    raise SystemExit(f"cannot infer --dataset from {base!r}; pass it explicitly")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("generations", help="JSONL written by --save-generations")
    p.add_argument("--dataset", default=None, help="Benchmark name (default: infer from filename)")
    p.add_argument("--out", default=None, help="Optional JSON path for the re-scored metrics")
    args = p.parse_args()

    dataset = args.dataset or _infer_dataset(args.generations)
    ok, reason = availability(dataset)
    print(f"[rescore] {dataset}: {'ok' if ok else 'UNAVAILABLE'} — {reason}")
    if not ok:
        raise SystemExit(1)

    with open(args.generations) as f:
        records = [json.loads(line) for line in f if line.strip()]
    if not records:
        raise SystemExit(f"no records in {args.generations}")
    # Type check, not a key check: the first version of this format used `row` for the row
    # *index*. A key check passes there, every official scorer then fails on an int, and the
    # result is a plausible-looking accuracy_official=0.0 — the exact silent-zero failure this
    # tool exists to avoid.
    if not isinstance(records[0].get("row"), dict):
        raise SystemExit(
            f"{args.generations} predates the self-contained format (`row` is not the benchmark "
            "row dict); re-run the eval to regenerate it."
        )

    n_off = sum(bool(score(dataset, rec["generated"], rec["row"])) for rec in records)
    n = len(records)
    metrics = {
        "dataset": dataset,
        "n": n,
        "accuracy_official": n_off / n,
        "official_scorer": SCORER_SOURCE[dataset],
        "official_fidelity": reason,
        "rescored_from": args.generations,
    }
    print(f"[rescore] n={n}  accuracy_official={metrics['accuracy_official']:.4f}")
    if args.out:
        with open(args.out, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"[rescore] wrote {args.out}")


if __name__ == "__main__":
    main()
