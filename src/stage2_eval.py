"""Stage 2 evaluation: greedy-decode accuracy of the fine-tuned LLM (paper §4.2).

At inference the LLM directly generates latent tokens (``<boLatent> <Latent-code-*> <eoLatent>``)
followed by text and a final ``The answer is: <value>`` — *no VQ-VAE is involved* (paper §3.2). We
greedily decode and score each generation with the benchmark's OWN published evaluator
(`src/official_eval.py`, vendored verbatim in `third_party/official_evals/`), reported as
``accuracy_official``. That is the only scorer: there is no in-repo answer extraction and no flag to
switch scorers. We also report the mean number of generated tokens (the token-efficiency axis the
paper tracks).

Loading: the saved artifact in ``--adapter`` is a PEFT checkpoint containing the LoRA adapters AND
the full ``embed_tokens``/``lm_head`` (``modules_to_save``) at the extended vocab size. We load the
base model, resize it to the saved (extended) tokenizer, then attach the adapter.

Usage:
    python -m src.stage2_eval --adapter runs/stage2-llama1b --dataset gsm8k --split test
    python -m src.stage2_eval --adapter runs/stage2-llama1b --dataset gsm8k --smoke
    python -m src.stage2_eval --adapter runs/stage2-llama1b --dataset college-math   # OOD
    # huge corpus: seeded 10k subsample, 64 prompts per generate() call (left-padded batching)
    python -m src.stage2_eval --adapter runs/stage2-fullft-learnedup-embedfix \
        --dataset openmathinstruct-2 --limit 10000 --sample-seed 0 --batch-size 64
"""
from __future__ import annotations

import argparse
import os
from typing import Callable, Dict, List, Optional, Tuple

import yaml

from .official_eval import OFFICIAL_SCORERS, SCORER_SOURCE, availability, fidelity_report, score


# --------------------------------------------------------------------------------------
# Benchmark registry: name -> spec dict with path/config/split + question/gold getters.
# The `gold` getter is for DISPLAY only (debug prints, the `gold` field of saved generations):
# the official scorers rebuild the gold answer from the raw row the way their own harness does.
# Optional spec keys: `data_files` (URL for the HF "json" builder — its only split is "train"),
# `filter` (row predicate applied after loading), `drop_columns` (removed BEFORE row access;
# TheoremQA's `Picture` is an Image feature that would require Pillow to decode).
# --------------------------------------------------------------------------------------
# The out-of-domain suite (paper Table 4.2) is exactly DART-Math's eval suite; where HF hosts no
# faithful copy we load the curated files from the DART-Math / MathScale(MWPBench) repos, pinned
# to commit SHAs so the benchmarks can't drift. Row counts match the paper: gaokao 30,
# deepmind 1000, college 2818, olympiad 675, theoremqa 800.
_DART_MATH_RAW = ("https://raw.githubusercontent.com/hkust-nlp/dart-math/"
                  "219b32467b55f7c610195e148b56df3431e1a963/data/dsets")
_MWPBENCH_RAW = ("https://raw.githubusercontent.com/microsoft/unilm/"
                 "833df7e7832e5064a281131ee64a481afa8e5b95/mathscale/MWPBench/data")
GAOKAO_URL = f"{_DART_MATH_RAW}/mwpbench/fresh-gaokao-math-2023.jsonl"
DEEPMIND_MATH_URL = f"{_DART_MATH_RAW}/deepmind-mathematics.json"
OLYMPIAD_MATH_URL = f"{_DART_MATH_RAW}/olympiadbench/OE_TO_maths_en_COMP.json"
MWPBENCH_FULL_TEST_URL = f"{_MWPBENCH_RAW}/full_test.json"


def _gsm8k_gold(row: Dict) -> str:
    # GSM8K answers end with "#### <final answer>"
    ans = row["answer"]
    return ans.split("####")[-1].strip()


def _boxed_gold(row: Dict) -> str:
    # Competition-MATH solutions carry the final answer in \boxed{...}
    sol = row.get("solution", row.get("answer", ""))
    b = _extract_boxed(sol)
    return b if b is not None else sol.strip()


def _mwpbench_gold(row: Dict) -> str:
    # MWPBench-style records (gaokao / deepmind / college) carry a short `answer` string,
    # sometimes wrapped in \( ... \) or \[ ... \] (Fresh-Gaokao); College-Math's $-wrapping is
    # left as-is (the official scorer strips it itself).
    ans = row["answer"].strip()
    for lo, hi in ((r"\(", r"\)"), (r"\[", r"\]")):
        if ans.startswith(lo) and ans.endswith(hi):
            ans = ans[len(lo):-len(hi)].strip()
    return ans


def _olympiad_gold(row: Dict) -> str:
    # OlympiadBench stores `final_answer` as a list of strings (usually length 1, even for
    # is_multiple_answer rows, where the answers share one string).
    return ", ".join(s.strip() for s in row["final_answer"])


def _theoremqa_gold(row: Dict) -> str:
    # TIGER-Lab/TheoremQA `Answer` covers integer/float/bool/list answers, serialized as strings.
    return str(row["Answer"]).strip()


BENCHMARKS: Dict[str, Dict] = {
    # in-domain (used to build MetaMathQA)
    "gsm8k": dict(path="openai/gsm8k", config="main", split="test",
                  question=lambda r: r["question"], gold=_gsm8k_gold),
    # NOTE: the original host `hendrycks/competition_math` is disabled on the Hub (DMCA takedown;
    # files 401 despite the repo page existing). EleutherAI hosts the identical data — same
    # problem/solution fields and the same 5000-example test split — split into 7 subject
    # configs, which the loader concatenates.
    "math": dict(path="EleutherAI/hendrycks_math",
                 config=["algebra", "counting_and_probability", "geometry",
                         "intermediate_algebra", "number_theory", "prealgebra", "precalculus"],
                 split="test",
                 question=lambda r: r["problem"], gold=_boxed_gold),
    # out-of-domain (paper Table 4.2)
    "gaokao-math-2023": dict(path="json", config=None, data_files=GAOKAO_URL, split="train",
                             question=lambda r: r["question"], gold=_mwpbench_gold),
    "deepmind-math": dict(path="json", config=None, data_files=DEEPMIND_MATH_URL, split="train",
                          question=lambda r: r["question"], gold=_mwpbench_gold),
    "college-math": dict(path="json", config=None, data_files=MWPBENCH_FULL_TEST_URL,
                         split="train",
                         filter=lambda r: r["data_source"].startswith("college_math"),
                         question=lambda r: r["question"], gold=_mwpbench_gold),
    "olympiad-math": dict(path="json", config=None, data_files=OLYMPIAD_MATH_URL, split="train",
                          question=lambda r: r["question"], gold=_olympiad_gold),
    "theoremqa": dict(path="TIGER-Lab/TheoremQA", config=None, split="test",
                      drop_columns=["Picture"],
                      question=lambda r: r["Question"], gold=_theoremqa_gold),
    # NOT a test benchmark: a 14M-row TRAINING corpus (augmented GSM8K/MATH-style problems), so
    # numbers are not comparable to the Table-4.2 suite. Splits: train (14M) and fair-downsampled
    # train_1M / train_2M / train_5M; we default to train_1M. ALWAYS pass --limit (with
    # --sample-seed for an unbiased subsample) — a full-split sweep is days of GPU.
    "openmathinstruct-2": dict(path="nvidia/OpenMathInstruct-2", config=None, split="train_1M",
                               question=lambda r: r["problem"],
                               gold=lambda r: str(r["expected_answer"]).strip()),
    # local subsets of the above (scripts/sample_openmathinstruct2.py, seed 0, natural
    # augmented_gsm8k/augmented_math composition 0.1665); same caveat: training-style corpus.
    "omi2-natural-100k": dict(path="json", config=None,
                              data_files="data/omi2_train_1M_n100000_seed0_natural.jsonl",
                              split="train",
                              question=lambda r: r["problem"],
                              gold=lambda r: str(r["expected_answer"]).strip()),
    "omi2-natural-10k": dict(path="json", config=None,
                             data_files="data/omi2_train_1M_n10000_seed0_natural.jsonl",
                             split="train",
                             question=lambda r: r["problem"],
                             gold=lambda r: str(r["expected_answer"]).strip()),
}


def load_benchmark_rows(dataset: str, split: Optional[str] = None,
                        limit: Optional[int] = None,
                        sample_seed: Optional[int] = None) -> Tuple[List[Dict], str]:
    """Load a registered benchmark as a list of row dicts (spec's default split unless given).

    Honors the optional spec keys: `data_files` (json builder), `drop_columns` (removed before
    any row access — TheoremQA's Image column would need Pillow), `filter` (applied before
    `limit`, so e.g. college-math limits count college rows, not raw MWPBench rows). A list
    `config` loads every named config and concatenates them (MATH's 7 subject configs).

    `sample_seed` shuffles before the limit is applied, turning `limit` into a seeded random
    subsample instead of a head slice (essential for huge corpora like openmathinstruct-2).
    For filter-free specs the limit is applied via `.select` BEFORE materializing rows, so a
    limited eval never walks a multi-million-row dataset.
    """
    from datasets import concatenate_datasets, load_dataset

    spec = BENCHMARKS[dataset]
    split = split or spec["split"]
    load_kwargs = {}
    if spec.get("data_files"):
        load_kwargs["data_files"] = spec["data_files"]
    config = spec.get("config")
    if isinstance(config, (list, tuple)):
        ds = concatenate_datasets(
            [load_dataset(spec["path"], c, split=split, **load_kwargs) for c in config])
    else:
        ds = load_dataset(spec["path"], config, split=split, **load_kwargs)
    if spec.get("drop_columns"):
        ds = ds.remove_columns([c for c in spec["drop_columns"] if c in ds.column_names])
    if sample_seed is not None:
        ds = ds.shuffle(seed=sample_seed)
    if limit is not None and spec.get("filter") is None:
        ds = ds.select(range(min(limit, len(ds))))    # never ds[a:b] (silently collapses rows)
    rows = [ds[i] for i in range(len(ds))]
    if spec.get("filter"):
        rows = [r for r in rows if spec["filter"](r)]
    if limit is not None:
        rows = rows[:limit]
    return rows, split


def _extract_boxed(text: str) -> Optional[str]:
    """Return the content of the last \\boxed{...}, handling nested braces (display gold only)."""
    idx = text.rfind("\\boxed")
    if idx == -1:
        return None
    i = text.find("{", idx)
    if i == -1:
        return None
    depth, j = 0, i
    while j < len(text):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[i + 1:j].strip()
        j += 1
    return None


# --------------------------------------------------------------------------------------
# Model loading
# --------------------------------------------------------------------------------------
def _model_dtype(model) -> str:
    """Precision the generations were ACTUALLY produced in, read off the model rather than off the
    request — so the output JSON can never claim a precision the run did not use.

    Falls back to ``"unknown"`` for the non-``nn.Module`` stubs the eval-loop unit tests drive
    ``evaluate()`` with; a real model always has parameters.
    """
    try:
        return str(next(model.parameters()).dtype).replace("torch.", "")
    except (AttributeError, StopIteration):
        return "unknown"


def _fp32_dtype():
    """Return ``torch.float32`` and pin OFF every reduced-precision matmul path, TF32 included.

    **Eval runs in fp32, always — there is no bf16 option.** Why bf16 inference is not batch-invariant.
    Changing the batch dimension or the left-padding width changes how cuBLAS tiles the GEMM, hence
    the accumulation order, and in bf16's 8-bit mantissa that is worth ~0.19 absolute in the logits.
    Greedy decode argmaxes, so any token whose top-1/top-2 margin is below that flips — and one
    flipped token rewrites the rest of the generation. Batch 1 vs batch 48 agreed on only 37/48
    generations in bf16, worth 1.2 GSM8K points at n=1319; the same comparison in fp32 is 6000x
    smaller (max |dlogit| 3e-5) and agreed 48/48. Padding and batching each cause it independently,
    so length-bucketing cannot fix it — fp32 is the only thing that does.

    TF32 is disabled explicitly rather than left to the torch default (which has changed across
    releases): it is a 10-bit mantissa and only recovered 44/48 agreement, i.e. it would silently
    undo the reason for being in fp32 at all.

    Cost: 2x weight VRAM (1B: 2.9 -> 5.8 GB peak) and, on a 48-row batch-48 probe.
    """
    import torch

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    return torch.float32


def load_finetuned(adapter_dir: str, base_model: Optional[str], device: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    torch_dtype = _fp32_dtype()
    tok = AutoTokenizer.from_pretrained(adapter_dir)  # the EXTENDED tokenizer (1026 new tokens)

    if os.path.exists(os.path.join(adapter_dir, "adapter_config.json")):
        # LoRA checkpoint: load the base model, then apply the adapter.
        # base model name: from the saved resolved config, else the CLI override
        cfg_path = os.path.join(adapter_dir, "stage2_config.resolved.yaml")
        if base_model is None and os.path.exists(cfg_path):
            with open(cfg_path) as f:
                base_model = (yaml.safe_load(f) or {}).get("base_model")
        assert base_model, "Provide --base-model (no stage2_config.resolved.yaml in the adapter dir)."

        model = AutoModelForCausalLM.from_pretrained(base_model, dtype=torch_dtype)
        model.resize_token_embeddings(len(tok))       # match the saved modules_to_save shapes
        # Generate with the PeftModel directly. We do NOT merge_and_unload(): the base model has
        # tie_word_embeddings=True and we intentionally untied embed_tokens/lm_head via
        # modules_to_save, which PEFT warns can corrupt a merge — the trained embed/lm_head are
        # applied regardless.
        model = PeftModel.from_pretrained(model, adapter_dir)
    else:
        # Full-FT checkpoint: a self-contained HF dir (extended vocab, tied weights) saved by
        # run_train's save_pretrained — no base model or adapter needed.
        model = AutoModelForCausalLM.from_pretrained(adapter_dir, dtype=torch_dtype)
        assert model.get_input_embeddings().weight.shape[0] == len(tok), (
            "full-FT checkpoint vocab size does not match its tokenizer"
        )
    model.eval().to(device)
    print(f"[eval] loaded {adapter_dir} on {device} in "
          f"{next(model.parameters()).dtype} (fp32 always — bf16 is not batch-invariant; ")
    return model, tok


# --------------------------------------------------------------------------------------
# Evaluation loop
# --------------------------------------------------------------------------------------
def evaluate(model, tok, rows, get_q: Callable, get_gold: Callable, device: str,
             max_new_tokens: int, batch_size: int = 1, *, dataset: str,
             generations_out: Optional[str] = None) -> Dict[str, float]:
    """Greedy-decode `rows` and score them with `dataset`'s official evaluator.

    `batch_size > 1` generates several prompts per `generate` call using LEFT padding (the
    prompts stay right-aligned so generation continues seamlessly) with an attention mask, and
    prompts are length-sorted first to minimize padding waste — accuracy/mean_gen_tokens are
    order-independent, and per-row token counts stop at each row's own EOS, so the protocol
    (greedy, max_new_tokens, EOS-stop) is unchanged vs `batch_size=1`.

    Every generation is scored by the benchmark's own published evaluator (`src/official_eval.py`)
    and reported as `accuracy_official` — the only scorer; `dataset` must have one registered.
    `get_gold` is used for display only. `generations_out` writes every raw generation to a JSONL
    so the scorer can be re-run later without touching a GPU.
    """
    import torch

    if dataset not in OFFICIAL_SCORERS:
        raise ValueError(f"no official scorer registered for {dataset!r}; "
                         f"known: {sorted(OFFICIAL_SCORERS)}")
    print(fidelity_report(dataset))
    # `score()` scores an unparseable prediction as wrong, so a MISSING DEPENDENCY would otherwise
    # surface as a perfectly plausible accuracy_official=0.0 for every row. It is the only metric,
    # so refuse to run at all rather than report a fake zero.
    ok_avail, reason = availability(dataset)
    if not ok_avail:
        raise RuntimeError(f"official scorer for {dataset!r} unavailable: {reason}")

    # left-pad with pad_token (fall back to eos — pure padding, masked out by attention_mask)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    # base model, no chat template: prompt = [BOS] + question (matches training assembly)
    prompts = [[tok.bos_token_id] + tok(get_q(r), add_special_tokens=False).input_ids
               for r in rows]
    order = list(range(len(rows)))
    if batch_size > 1:
        order.sort(key=lambda i: len(prompts[i]))
    n_correct, n_tokens, n, n_trunc = 0, 0, 0, 0
    gen_fh = open(generations_out, "w") if generations_out else None
    for start in range(0, len(order), batch_size):
        idx = order[start:start + batch_size]
        width = max(len(prompts[i]) for i in idx)
        input_ids = torch.tensor(
            [[pad_id] * (width - len(prompts[i])) + prompts[i] for i in idx], device=device)
        attention_mask = torch.tensor(
            [[0] * (width - len(prompts[i])) + [1] * len(prompts[i]) for i in idx], device=device)
        with torch.no_grad():
            out = model.generate(
                input_ids, attention_mask=attention_mask,
                do_sample=False, num_beams=1,                     # greedy
                max_new_tokens=max_new_tokens,
                eos_token_id=tok.eos_token_id, pad_token_id=pad_id,
            )
        gen = out[:, width:]
        for b, i in enumerate(idx):
            row_gen = gen[b]
            # generated length = up to and including this row's own EOS; the tail beyond it is
            # batch padding for rows that finished earlier (matches the unbatched count exactly)
            eos_pos = (row_gen == tok.eos_token_id).nonzero()
            n_gen = int(eos_pos[0, 0]) + 1 if eos_pos.numel() else int(row_gen.shape[0])
            n_tokens += n_gen
            # no EOS anywhere in this row's span => it never terminated, so the batch ran the
            # full budget and this row was cut off at `max_new_tokens`. Reported so a change of
            # budget can be read as "the cap was/wasn't binding" instead of inferred from means.
            n_trunc += int(eos_pos.numel() == 0)
            text = tok.decode(row_gen[:n_gen], skip_special_tokens=True)  # latent codes drop out
            gold = get_gold(rows[i])
            # the official evaluator does its OWN extraction and rebuilds gold from the raw row
            ok = bool(score(dataset, text, rows[i]))
            n_correct += int(ok)
            n += 1
            if gen_fh is not None:
                import json as _json
                # the FULL row is stored, not just `gold`: official scorers rebuild the gold
                # answer from the raw fields themselves (TheoremQA needs `Answer_type`,
                # OlympiadBench needs `error`/`final_answer`), so this makes the file
                # self-contained — re-scoring never has to reload the benchmark.
                gen_fh.write(_json.dumps({
                    "row_idx": i, "row": rows[i], "gold": gold, "generated": text,
                    "correct_official": ok,
                }, default=str) + "\n")
            if n <= 3:
                print(f"[eval] gold={gold!r} official={ok} tail={text[-80:]!r}")
    if gen_fh is not None:
        gen_fh.close()
        print(f"[eval] wrote {n} generations to {generations_out}")
    metrics = {
        "accuracy_official": n_correct / max(n, 1),
        "n": n,
        "mean_gen_tokens": n_tokens / max(n, 1),
        "frac_truncated": n_trunc / max(n, 1),
        "max_new_tokens": max_new_tokens,
        # Provenance for the two settings that MOVE the number, recorded from what actually ran
        # rather than from what was requested. dtype: bf16 results are not comparable to fp32 ones.
        # batch_size: under bf16 it shifts accuracy by ~1.2 GSM8K points (left-padding changes the
        # GEMM reduction order); under fp32 it is invariant.
        "dtype": _model_dtype(model),
        "batch_size": batch_size,
    }
    return metrics


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--adapter", required=True, help="Stage-2 output_dir (PEFT checkpoint + tokenizer).")
    p.add_argument("--base-model", dest="base_model", default=None)
    p.add_argument("--dataset", default="gsm8k", choices=sorted(BENCHMARKS.keys()))
    p.add_argument("--split", default=None, help="Override the benchmark's default split.")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--sample-seed", dest="sample_seed", type=int, default=None,
                   help="Shuffle with this seed before --limit → seeded random subsample "
                        "instead of a head slice (use for huge corpora like openmathinstruct-2).")
    p.add_argument("--max-new-tokens", dest="max_new_tokens", type=int, default=512)
    p.add_argument("--batch-size", dest="batch_size", type=int, default=1,
                   help="Prompts per generate() call (left-padded greedy batching). 1 = the "
                        "original sequential loop.")
    p.add_argument("--device", default="cuda")
    p.add_argument("--smoke", action="store_true", help="Tiny end-to-end path check (limit 16).")
    p.add_argument("--out", default=None, help="Optional JSON path to write the metrics.")
    p.add_argument("--save-generations", dest="save_generations", default=None,
                   help="JSONL path for the raw generations, so the scorer can be re-run later "
                        "without regenerating.")
    # ---- MLflow logging (opt-in; does not touch the eval protocol) ----
    p.add_argument("--mlflow", dest="mlflow", action="store_true", default=False,
                   help="Log benchmark metrics to MLflow.")
    p.add_argument("--mlflow-tracking-uri", dest="mlflow_tracking_uri", default=None)
    p.add_argument("--mlflow-experiment", dest="mlflow_experiment", default="stage2(SFT)")
    p.add_argument("--mlflow-run-id", dest="mlflow_run_id", default=None,
                   help="Attach metrics to this existing run. Default: read "
                        "<adapter>/mlflow_run_id.txt (written by training); else start a new run.")
    return p


def log_metrics_to_mlflow(args, dataset: str, split: str, metrics: Dict[str, float]) -> None:
    """Log benchmark metrics to MLflow, attaching them to the training run when possible.

    Metrics are prefixed with the dataset name (``gsm8k/accuracy_official``,
    ``gsm8k/mean_gen_tokens``, …) so multiple benchmarks coexist and never collide with training's
    ``eval/loss`` (validation) curves. Run resolution: ``--mlflow-run-id`` >
    ``<adapter>/mlflow_run_id.txt`` > a fresh run.
    """
    import mlflow

    if args.mlflow_tracking_uri:
        mlflow.set_tracking_uri(args.mlflow_tracking_uri)

    run_id = args.mlflow_run_id
    if run_id is None:
        rid_path = os.path.join(args.adapter, "mlflow_run_id.txt")
        if os.path.exists(rid_path):
            with open(rid_path) as f:
                run_id = f.read().strip() or None

    # only numeric entries are MLflow metrics (the official-scorer provenance strings are not)
    payload = {f"{dataset}/{k}": float(v) for k, v in metrics.items()
               if isinstance(v, (int, float)) and not isinstance(v, bool)}

    def _log_fresh():
        run_name = f"eval-{os.path.basename(args.adapter.rstrip('/'))}-{dataset}"
        with mlflow.start_run(run_name=run_name):
            mlflow.log_params({"adapter": args.adapter, "dataset": dataset, "split": split})
            mlflow.log_metrics(payload)
        print(f"[eval] logged {dataset} metrics to a new MLflow run {run_name!r}")

    if run_id:
        try:
            with mlflow.start_run(run_id=run_id):
                mlflow.log_metrics(payload)
            print(f"[eval] logged {dataset} metrics to training run {run_id}")
            return
        except Exception as e:  # run id stale/missing -> fall back to a standalone run
            print(f"[eval] could not attach to run {run_id} ({e}); logging a new run instead")
    mlflow.set_experiment(args.mlflow_experiment)
    _log_fresh()


def main() -> None:
    import json

    args = build_argparser().parse_args()
    if args.smoke:
        args.limit = args.limit or 16
        args.max_new_tokens = min(args.max_new_tokens, 128)

    # fail before loading data or weights: without its scorer a benchmark has no metric at all
    ok_avail, fidelity = availability(args.dataset)
    if not ok_avail:
        raise SystemExit(f"[eval] official scorer for {args.dataset!r} unavailable: {fidelity}")

    spec = BENCHMARKS[args.dataset]
    rows, split = load_benchmark_rows(args.dataset, args.split, args.limit, args.sample_seed)

    model, tok = load_finetuned(args.adapter, args.base_model, args.device)
    metrics = evaluate(model, tok, rows, spec["question"], spec["gold"],
                       args.device, args.max_new_tokens, args.batch_size,
                       dataset=args.dataset, generations_out=args.save_generations)
    print(f"[eval] {args.dataset} ({split}, n={metrics['n']}): "
          f"accuracy_official={metrics['accuracy_official']:.4f}  "
          f"mean_gen_tokens={metrics['mean_gen_tokens']:.1f}  "
          f"frac_truncated={metrics['frac_truncated']:.4f}  "
          f"max_new_tokens={metrics['max_new_tokens']}")
    print(f"[eval]   scorer: {SCORER_SOURCE[args.dataset]}")
    if args.out:
        provenance = {"official_scorer": SCORER_SOURCE[args.dataset],
                      "official_fidelity": fidelity}
        if args.sample_seed is not None:
            provenance["sample_seed"] = args.sample_seed
        with open(args.out, "w") as f:
            json.dump({"dataset": args.dataset, "split": split, **provenance, **metrics},
                      f, indent=2)
    if args.mlflow:
        log_metrics_to_mlflow(args, args.dataset, split, metrics)


if __name__ == "__main__":
    main()
