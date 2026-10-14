#!/bin/bash
# =============================================================================
# Token Assorted — Stage 2 eval over the paper's 7 Table-4.2 benchmarks.
#
# Runs `python -m src.stage2_eval` once per benchmark, one after another on the local GPU:
#   in-domain       gsm8k, math
#   out-of-domain   gaokao-math-2023, deepmind-math, college-math, olympiad-math, theoremqa
# Each generation is scored only by its benchmark's official evaluator (-> accuracy_official).
# A failing benchmark does not stop the rest; a summary table is printed at the end and the exit
# status is 1 if any benchmark failed. The model is re-loaded for every benchmark.
#
# Usage (from the project root; every relative path below is relative to it):
#   bash scripts/eval_stage2_all.sh runs/stage2-llama1b
#   ADAPTER=runs/stage2-llama1b bash scripts/eval_stage2_all.sh
#   DRY_RUN=1 bash scripts/eval_stage2_all.sh runs/stage2-llama1b      # checks + commands, runs nothing
#   DATASETS="gsm8k theoremqa" bash scripts/eval_stage2_all.sh runs/stage2-llama1b
#   LIMIT=200 SAMPLE_SEED=0 OUT_DIR=/tmp/evals bash scripts/eval_stage2_all.sh runs/stage2-llama1b
#   bash scripts/eval_stage2_all.sh runs/stage2-llama1b --device cpu   # extra args go to src.stage2_eval
# A long run, detached (and its log):
#   nohup bash scripts/eval_stage2_all.sh runs/stage2-llama1b > runs/stage2-llama1b/eval_all.log 2>&1 &
#
# Environment (defaults in brackets):
#   DATASETS          [gsm8k math gaokao-math-2023 deepmind-math college-math olympiad-math theoremqa]
#   BATCH_SIZE        [64]   fp32 eval is batch-invariant, so this only affects speed/VRAM
#   MAX_NEW_TOKENS    [512]  the settled protocol; another value is a DECODING PARAM and suffixes the
#                            output file _mnt<N>, so it never overwrites the canonical numbers
#   LIMIT, SAMPLE_SEED       optional row cap / seeded subsample; suffixes the file _n<LIMIT>[_seed<S>].
#                            LIMIT is REQUIRED if DATASETS includes openmathinstruct-2 (14M rows)
#   OUT_DIR           [the adapter dir]   where eval_<dataset>*.json and *_generations.jsonl go
#   SAVE_GENERATIONS  [1]    keep raw generations so the scorer can be re-run without a GPU pass
#   MLFLOW            [0]    1 = log to MLflow (opt-in, so an ad-hoc run never touches the tracking
#                            DB unasked)
#   MLFLOW_TRACKING_URI [sqlite:///<project>/mlflow.db]
#   OVERWRITE         [0]    1 = replace existing outputs. Default REFUSES: eval_*.json files and
#                            their saved generations are the canonical results
#   PYTHON            [.venv/bin/python]      DRY_RUN [0]
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

usage() {
    cat <<'EOF'
usage: [ENV=...] bash scripts/eval_stage2_all.sh <adapter_dir> [extra src.stage2_eval args]
       ADAPTER=<adapter_dir> [ENV=...] bash scripts/eval_stage2_all.sh [extra src.stage2_eval args]
Paths are relative to the project root. The environment variables are listed in the script header.
EOF
}

# ---- arguments ----
case "${1:-}" in -h|--help) usage; exit 0 ;; esac
if [ $# -ge 1 ] && [[ "$1" != -* ]]; then
    ADAPTER="$1"
    shift
fi
ADAPTER="${ADAPTER:-}"
if [ -z "$ADAPTER" ]; then usage >&2; exit 2; fi
ADAPTER="${ADAPTER%/}"

DATASETS="${DATASETS:-gsm8k math gaokao-math-2023 deepmind-math college-math olympiad-math theoremqa}"
BATCH_SIZE="${BATCH_SIZE:-64}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
LIMIT="${LIMIT:-}"
SAMPLE_SEED="${SAMPLE_SEED:-}"
OUT_DIR="${OUT_DIR:-$ADAPTER}"
OUT_DIR="${OUT_DIR%/}"
SAVE_GENERATIONS="${SAVE_GENERATIONS:-1}"
MLFLOW="${MLFLOW:-0}"
MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-sqlite:///$PWD/mlflow.db}"
OVERWRITE="${OVERWRITE:-0}"
DRY_RUN="${DRY_RUN:-0}"
PY="${PYTHON:-.venv/bin/python}"

# ---- pre-flight: everything that can be wrong is checked before any model is loaded ----
if [ ! -d "$ADAPTER" ]; then
    echo "ERROR: adapter dir '$ADAPTER' not found (paths are relative to $PWD)"; exit 2
fi
if [ ! -f "$ADAPTER/adapter_config.json" ] && [ ! -f "$ADAPTER/config.json" ]; then
    echo "ERROR: '$ADAPTER' is not a Stage-2 checkpoint (no adapter_config.json / config.json)"; exit 2
fi
if ! command -v "$PY" >/dev/null 2>&1; then
    echo "ERROR: python '$PY' not found (create the .venv, or set PYTHON=...)"; exit 2
fi
for v in BATCH_SIZE MAX_NEW_TOKENS LIMIT; do
    if [ -n "${!v}" ] && ! [[ "${!v}" =~ ^[1-9][0-9]*$ ]]; then
        echo "ERROR: $v must be a positive integer, got '${!v}'"; exit 2
    fi
done
if [ -n "$SAMPLE_SEED" ] && ! [[ "$SAMPLE_SEED" =~ ^[0-9]+$ ]]; then
    echo "ERROR: SAMPLE_SEED must be a non-negative integer, got '$SAMPLE_SEED'"; exit 2
fi

read -r -a DS_ARR <<< "$DATASETS"
if [ "${#DS_ARR[@]}" -eq 0 ]; then echo "ERROR: DATASETS is empty"; exit 2; fi
"$PY" - "${DS_ARR[@]}" <<'PY' || exit 2
import sys

from src.stage2_eval import BENCHMARKS

bad = [d for d in sys.argv[1:] if d not in BENCHMARKS]
if bad:
    print(f"ERROR: unknown dataset(s) {bad}; choose from {sorted(BENCHMARKS)}")
    sys.exit(2)
PY
for ds in "${DS_ARR[@]}"; do
    if [ "$ds" = "openmathinstruct-2" ] && [ -z "$LIMIT" ]; then
        echo "ERROR: openmathinstruct-2 is 14M rows — set LIMIT (and SAMPLE_SEED) to subsample it"; exit 2
    fi
done

MNT_SUFFIX=""
if [ "$MAX_NEW_TOKENS" != "512" ]; then MNT_SUFFIX="_mnt${MAX_NEW_TOKENS}"; fi
# eval_<dataset>[_n<limit>][_seed<seed>][_mnt<budget>].json — scripts/rescore_generations.py parses it
out_path() { echo "${OUT_DIR}/eval_${1}${LIMIT:+_n${LIMIT}}${SAMPLE_SEED:+_seed${SAMPLE_SEED}}${MNT_SUFFIX}.json"; }

clash=()
for ds in "${DS_ARR[@]}"; do
    o="$(out_path "$ds")"
    if [ -e "$o" ]; then clash+=("$o"); fi
    if [ "$SAVE_GENERATIONS" = "1" ] && [ -e "${o%.json}_generations.jsonl" ]; then
        clash+=("${o%.json}_generations.jsonl")
    fi
done
if [ "${#clash[@]}" -gt 0 ] && [ "$OVERWRITE" != "1" ]; then
    printf 'REFUSING: these outputs already exist (move them aside, set OUT_DIR, narrow DATASETS, or OVERWRITE=1):\n'
    printf '  %s\n' "${clash[@]}"
    exit 2
fi

# options shared by every benchmark (extra command-line args come last, so they win)
OPTS=(--batch-size "$BATCH_SIZE" --max-new-tokens "$MAX_NEW_TOKENS")
if [ -n "$LIMIT" ]; then OPTS+=(--limit "$LIMIT"); fi
if [ -n "$SAMPLE_SEED" ]; then OPTS+=(--sample-seed "$SAMPLE_SEED"); fi
if [ "$MLFLOW" = "1" ]; then OPTS+=(--mlflow --mlflow-tracking-uri "$MLFLOW_TRACKING_URI"); fi

export HF_HOME="${HF_HOME:-$PWD/.cache/huggingface}"
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export HF_HUB_DISABLE_PROGRESS_BARS=1 HF_DATASETS_DISABLE_PROGRESS_BARS=1

echo "[eval-all] adapter     : $ADAPTER"
echo "[eval-all] datasets    : ${DS_ARR[*]}"
echo "[eval-all] options     : ${OPTS[*]} $*"
echo "[eval-all] outputs     : $OUT_DIR  (generations: $([ "$SAVE_GENERATIONS" = 1 ] && echo saved || echo not saved); MLflow: $([ "$MLFLOW" = 1 ] && echo on || echo off))"
echo "[eval-all] python      : $PY   host: $(hostname)   $(date -Is)"
if command -v nvidia-smi >/dev/null 2>&1; then
    echo "[eval-all] gpu (used/total MiB): $(nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader 2>/dev/null | head -1 || true)"
fi

if [ "$DRY_RUN" = "1" ]; then
    i=0
    for ds in "${DS_ARR[@]}"; do
        i=$((i + 1))
        out="$(out_path "$ds")"
        gen=()
        if [ "$SAVE_GENERATIONS" = "1" ]; then gen=(--save-generations "${out%.json}_generations.jsonl"); fi
        printf '  %d/%d  ' "$i" "${#DS_ARR[@]}"
        printf '%q ' "$PY" -m src.stage2_eval --adapter "$ADAPTER" --dataset "$ds" "${OPTS[@]}" \
            --out "$out" "${gen[@]}" "$@"
        echo
    done
    echo "DRY_RUN=1: checks passed, nothing was run."
    exit 0
fi

mkdir -p "$OUT_DIR"
trap 'echo; echo "[eval-all] interrupted — remaining benchmarks skipped"; exit 130' INT TERM

results=()   # flat triples: dataset, exit code, output json
nfail=0
i=0
for ds in "${DS_ARR[@]}"; do
    i=$((i + 1))
    out="$(out_path "$ds")"
    gen=()
    if [ "$SAVE_GENERATIONS" = "1" ]; then gen=(--save-generations "${out%.json}_generations.jsonl"); fi
    # OVERWRITE: clear stale outputs first, so a failure below can never leave (or report) an old file
    rm -f "$out" "${out%.json}_generations.jsonl"
    echo
    echo "================ [$i/${#DS_ARR[@]}] $ds  ($(date -Is)) ================"
    rc=0
    "$PY" -m src.stage2_eval --adapter "$ADAPTER" --dataset "$ds" "${OPTS[@]}" \
        --out "$out" "${gen[@]}" "$@" 2>&1 || rc=$?
    if [ "$rc" -ne 0 ]; then
        nfail=$((nfail + 1))
        echo "[eval-all] WARNING: $ds FAILED (rc=$rc) — continuing with the next benchmark"
    fi
    results+=("$ds" "$rc" "$out")
done

echo
echo "================ SUMMARY (accuracy_official / mean_gen_tokens / frac_truncated) ================"
"$PY" - "${results[@]}" <<'PY'
import json
import os
import sys

a = sys.argv[1:]
rows = list(zip(a[0::3], a[1::3], a[2::3]))
w = max(len(d) for d, _, _ in rows)
print(f"{'dataset'.ljust(w)}  {'n':>6}  {'accuracy_official':>17}  {'mean_gen_tokens':>15}  {'frac_truncated':>14}")
for ds, rc, path in rows:
    if rc != "0" or not os.path.exists(path):
        print(f"{ds.ljust(w)}  FAILED (rc={rc})")
        continue
    m = json.load(open(path))
    print(f"{ds.ljust(w)}  {m['n']:>6}  {m['accuracy_official']:>17.4f}  "
          f"{m['mean_gen_tokens']:>15.1f}  {m['frac_truncated']:>14.4f}")
PY
echo "[eval-all] $((${#DS_ARR[@]} - nfail))/${#DS_ARR[@]} benchmarks succeeded; results in $OUT_DIR   $(date -Is)"
if [ "$nfail" -ne 0 ]; then exit 1; fi
