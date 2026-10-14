"""Generate a tiny synthetic MetaMathQA-style JSONL fixture for the offline smoke test/tests.

The smoke path must run on CPU in ~1-2 min with **no network**, so we don't download the real
395k-row dataset. These rows imitate MetaMathQA's schema (`query`/`response`) and answer format
(`#### <val>` then `The answer is: <val>`), which is enough to exercise the full pipeline
(split -> tokenize -> chunk -> VQ -> reconstruct) and let the model overfit so loss decreases.

Usage:
    python scripts/make_fixture.py --out tests/fixtures/metamath_smoke.jsonl --n 128
"""
from __future__ import annotations

import argparse
import json
import os
import random

_TEMPLATES = [
    ("What is {a} + {b}?", lambda a, b: a + b,
     "We add the two numbers: {a} + {b} = {c}.\n#### {c}\nThe answer is: {c}"),
    ("What is {a} times {b}?", lambda a, b: a * b,
     "We multiply the two numbers: {a} * {b} = {c}.\n#### {c}\nThe answer is: {c}"),
    ("If a box has {a} apples and you add {b} more, how many are there?", lambda a, b: a + b,
     "Start with {a} apples and add {b}: {a} + {b} = {c} apples.\n#### {c}\nThe answer is: {c}"),
    ("A train travels {a} km then {b} km. How far in total?", lambda a, b: a + b,
     "Total distance is {a} + {b} = {c} km.\n#### {c}\nThe answer is: {c}"),
    ("What is {a} minus {b}?", lambda a, b: a - b,
     "We subtract: {a} - {b} = {c}.\n#### {c}\nThe answer is: {c}"),
]


def make_rows(n: int, seed: int = 0):
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        q_tmpl, fn, r_tmpl = rng.choice(_TEMPLATES)
        a, b = rng.randint(2, 99), rng.randint(2, 99)
        c = fn(a, b)
        rows.append(
            {
                "type": "SYNTH",
                "query": q_tmpl.format(a=a, b=b),
                "original_question": q_tmpl.format(a=a, b=b),
                "response": r_tmpl.format(a=a, b=b, c=c),
            }
        )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="tests/fixtures/metamath_smoke.jsonl")
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    rows = make_rows(args.n, args.seed)
    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"Wrote {len(rows)} synthetic rows -> {args.out}")


if __name__ == "__main__":
    main()
