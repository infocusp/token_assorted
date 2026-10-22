"""Stage-2 eval registry: shape of BENCHMARKS + the new gold getters (CPU-only, no network)."""
import pytest

from src.stage2_eval import (
    BENCHMARKS,
    _mwpbench_gold,
    _olympiad_gold,
    _theoremqa_gold,
)

TABLE_4_2 = {"gsm8k", "math", "gaokao-math-2023", "deepmind-math", "college-math",
             "olympiad-math", "theoremqa"}


def test_registry_has_all_table_4_2_benchmarks():
    assert TABLE_4_2 <= set(BENCHMARKS)
    for name, spec in BENCHMARKS.items():
        assert spec["path"], name
        assert spec["split"], name
        assert callable(spec["question"]) and callable(spec["gold"]), name


def test_mwpbench_gold_unwraps_latex_delimiters():
    # Fresh-Gaokao answers come wrapped in \( ... \); college/deepmind answers pass through
    # untouched (display only — the official scorer reads the raw row and strips $ itself).
    assert _mwpbench_gold({"answer": r"\( \{x | -2 \leq x < 1\} \)"}) == r"\{x | -2 \leq x < 1\}"
    assert _mwpbench_gold({"answer": r"\[ 80 \]"}) == "80"
    assert _mwpbench_gold({"answer": "diverges"}) == "diverges"
    assert _mwpbench_gold({"answer": "$10-4 n$"}) == "$10-4 n$"


def test_olympiad_gold_joins_final_answer_list():
    assert _olympiad_gold({"final_answer": ["2"]}) == "2"
    assert _olympiad_gold({"final_answer": [" $x=1$ ", "$x=2$"]}) == "$x=1$, $x=2$"


def test_theoremqa_gold_stringifies_typed_answers():
    assert _theoremqa_gold({"Answer": "11760"}) == "11760"
    assert _theoremqa_gold({"Answer": 1.0}) == "1.0"
    assert _theoremqa_gold({"Answer": True}) == "True"


def test_math_uses_all_seven_subject_configs():
    # hendrycks/competition_math is DMCA-disabled on the Hub; the EleutherAI mirror splits the
    # identical 5000-example test set into 7 subject configs that the loader concatenates.
    cfg = BENCHMARKS["math"]["config"]
    assert BENCHMARKS["math"]["path"] == "EleutherAI/hendrycks_math"
    assert sorted(cfg) == ["algebra", "counting_and_probability", "geometry",
                           "intermediate_algebra", "number_theory", "prealgebra", "precalculus"]


def test_college_math_filter_selects_only_college_rows():
    keep = BENCHMARKS["college-math"]["filter"]
    assert keep({"data_source": "college_math.Calculus"})
    assert not keep({"data_source": "gsm8k"})


def test_openmathinstruct2_entry():
    # training corpus (augmented GSM8K/MATH style), NOT part of the Table-4.2 suite; default
    # split is the fair-downsampled train_1M and evals are expected to pass --limit.
    spec = BENCHMARKS["openmathinstruct-2"]
    assert spec["path"] == "nvidia/OpenMathInstruct-2"
    assert spec["split"] == "train_1M"
    assert spec["question"]({"problem": "What is 2+2?"}) == "What is 2+2?"
    assert spec["gold"]({"expected_answer": " 4 "}) == "4"


# ---------------------------------------------------------------------------------------
# Batched generation: evaluate(batch_size>1) must reproduce the sequential loop exactly —
# same per-row prompt→prediction pairing (despite length-sorting) and same token counting
# (each row's count stops at its own EOS, never including batch padding).
# ---------------------------------------------------------------------------------------
class _StubTok:
    """Prompt length varies with question; decode maps the payload token back to a number."""
    bos_token_id, eos_token_id, pad_token_id = 0, 1, 2

    def __call__(self, text, add_special_tokens=False):
        import types
        return types.SimpleNamespace(input_ids=[10] * len(text))

    def decode(self, ids, skip_special_tokens=True):
        vals = [int(t) - 100 for t in ids if int(t) >= 100]
        return f"The answer is: {vals[0]}" if vals else ""


class _StubModel:
    """Echoes each row's true (unpadded) prompt length as its answer, then EOS."""

    def generate(self, input_ids, attention_mask=None, do_sample=None, num_beams=None,
                 max_new_tokens=None, eos_token_id=None, pad_token_id=None):
        import torch
        bsz, width = input_ids.shape
        lens = (attention_mask.sum(dim=1) if attention_mask is not None
                else torch.full((bsz,), width))
        gen = torch.full((bsz, 3), pad_token_id, dtype=input_ids.dtype)
        for b in range(bsz):
            gen[b, 0] = 100 + int(lens[b])   # payload: the row's own prompt length
            gen[b, 1] = eos_token_id         # then EOS; gen[b, 2] stays batch padding
        return torch.cat([input_ids, gen], dim=1)


def _stub_rows(lengths):
    """GSM8K-shaped rows whose gold is the stub model's answer (prompt = [BOS] + one token per
    char), so the official GSM8K scorer grades every correctly paired row as right."""
    return [{"q": "x" * k, "answer": f"reasoning\n#### {1 + k}"} for k in lengths]


def _stub_gold(r):
    return r["answer"].split("####")[-1].strip()


def test_batched_evaluate_matches_sequential():
    from src.stage2_eval import evaluate

    # varied lengths force real padding and a nontrivial length-sort within batches
    rows = _stub_rows((5, 1, 9, 3, 7, 2, 8))
    def get_q(r):
        return r["q"]

    seq = evaluate(_StubModel(), _StubTok(), rows, get_q, _stub_gold, "cpu",
                   max_new_tokens=4, batch_size=1, dataset="gsm8k")
    bat = evaluate(_StubModel(), _StubTok(), rows, get_q, _stub_gold, "cpu",
                   max_new_tokens=4, batch_size=3, dataset="gsm8k")
    assert seq["accuracy_official"] == 1.0 == bat["accuracy_official"]
    assert seq["n"] == bat["n"] == len(rows)
    # payload + EOS = 2 generated tokens per row; batch padding must NOT be counted
    assert seq["mean_gen_tokens"] == bat["mean_gen_tokens"] == 2.0


# --------------------------------------------------------------------------------------
# fp32 inference: bf16 is not batch-invariant, so eval no longer offers it.
# --------------------------------------------------------------------------------------
def test_eval_precision_is_fp32_and_not_selectable():
    """In bf16, batch 1 vs batch 48 agreed on only 37/48 generations (1.2 GSM8K points at n=1319);
    in fp32, 48/48 — and end-to-end at batch 1 vs 64 over 96 rows, every generation matched.
    Precision is therefore fixed, not a knob: a `--dtype bf16` escape hatch would quietly
    reintroduce batch-dependent accuracy."""
    import inspect

    from src import stage2_eval as ev

    assert "dtype" not in inspect.signature(ev.load_finetuned).parameters
    assert "--dtype" not in {s for a in ev.build_argparser()._actions for s in a.option_strings}


def test_fp32_helper_pins_tf32_off():
    """TF32 is a 10-bit mantissa and only recovered 44/48 batch agreement — it would silently undo
    the reason for being in fp32. torch's defaults here have moved across releases, so pin them."""
    import torch

    from src.stage2_eval import _fp32_dtype

    prev = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32,
            torch.get_float32_matmul_precision())
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("medium")
        assert _fp32_dtype() is torch.float32
        assert torch.backends.cuda.matmul.allow_tf32 is False
        assert torch.backends.cudnn.allow_tf32 is False
        assert torch.get_float32_matmul_precision() == "highest"
    finally:
        (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32) = prev[:2]
        torch.set_float32_matmul_precision(prev[2])


def test_metrics_record_dtype_and_batch_size():
    """Provenance for the two settings that move the number. Without them an old bf16 JSON and a
    new fp32 one are indistinguishable on disk."""
    from src.stage2_eval import evaluate

    m = evaluate(_StubModel(), _StubTok(), _stub_rows((5, 1, 9)), lambda r: r["q"], _stub_gold,
                 "cpu", max_new_tokens=4, batch_size=2, dataset="gsm8k")
    assert m["batch_size"] == 2
    assert m["dtype"] == "unknown"   # stub is not an nn.Module; a real model reports float32


# --------------------------------------------------------------------------------------
# Official scorer only: the in-repo scorer was removed 2026-10-07 and is not selectable.
# --------------------------------------------------------------------------------------
def test_eval_is_official_only_and_not_selectable():
    """`accuracy_official` (each benchmark's own published evaluator) is the only scorer. A
    `--no-official` switch or a resurrected in-repo extractor would bring back a second accuracy
    scale that historical `accuracy` numbers could be confused with."""
    import inspect

    from src import stage2_eval as ev

    flags = {s for a in ev.build_argparser()._actions for s in a.option_strings}
    assert "--no-official" not in flags
    assert "official" not in inspect.signature(ev.evaluate).parameters
    for name in ("extract_pred", "normalize_answer", "answers_match"):
        assert not hasattr(ev, name), name

    m = ev.evaluate(_StubModel(), _StubTok(), _stub_rows((2, 4)), lambda r: r["q"], _stub_gold,
                    "cpu", max_new_tokens=4, dataset="gsm8k")
    assert "accuracy" not in m and m["accuracy_official"] == 1.0
    with pytest.raises(ValueError):
        ev.evaluate(_StubModel(), _StubTok(), _stub_rows((2,)), lambda r: r["q"], _stub_gold,
                    "cpu", max_new_tokens=4, dataset="not-a-benchmark")
