"""Official per-benchmark scorers — THE scorer for Stage-2 eval.

`src/stage2_eval.py::evaluate()` scores every generation with the benchmark's own published
evaluator, routed here, and reports it as `accuracy_official`. There is no other scorer and no
flag to disable this one. (Until 2026-10-07 a simpler in-repo scorer reported `accuracy`
alongside; it was removed because it could not strip ``^\\circ``, ``\\left/\\right``, ``\\text{}``
units or a LaTeX ``\\$``, had no symbolic/set comparison, and used a 1e-4 absolute tolerance where
TheoremQA's official evaluator allows 4% *relative*. `accuracy` in older eval JSONs is that
retired scale — never compare it against `accuracy_official`.)

Faithfulness
------------
The evaluators themselves are vendored verbatim in `third_party/official_evals/` at pinned
upstream commits (see its `PROVENANCE.md`). This file only routes each benchmark to its own
evaluator and rebuilds the gold answer **the way that evaluator's own harness does** — which
is why scorers take the whole `row`, not a pre-extracted gold string. Where the official
harness's choice cannot be reproduced exactly, the scorer's `notes` say so; nothing is
silently approximated.

Scorer contract:  ``scorer(generated_text: str, row: dict) -> bool``

One assumption worth stating: the GSM8K and MATH scorers split on ``"The answer is: "`` *with*
the trailing space (MetaMath's exact string), whereas the training data's `ANSWER_MARKER` is
``"The answer is:"`` without it. Checked on 20k MetaMathQA responses: 100% use the spaced form,
so this never costs us an extraction. The marker being glued to the *preceding* token
("...correctly.The answer is: 5", see `src/data.py`) is upstream of the split and harmless.
"""
from __future__ import annotations

import ast
from types import SimpleNamespace
from typing import Callable, Dict, List, Optional, Tuple

from third_party.official_evals import metamath_extract as _metamath
from third_party.official_evals import mwpbench_util as _mwpbench
from third_party.official_evals import theoremqa_utils as _theoremqa
from third_party.official_evals.dartmath_eval import EvaluatorMath as _DartEvaluatorMath
from third_party.official_evals.dartmath_extract import EvaluatorBase as _DartEvaluatorBase
from third_party.official_evals.hendrycks_math_equivalence import is_equiv as _hendrycks_is_equiv


# --------------------------------------------------------------------------------------
# GSM8K — MetaMath `eval_gsm8k.py`
# --------------------------------------------------------------------------------------
def _score_gsm8k(generated: str, row: Dict) -> bool:
    """MetaMath's GSM8K protocol: `extract_answer_number` + float equality.

    MetaMath rather than OpenAI's original `grade-school-math/dataset.py`: that one extracts
    with ``ANS_RE = r"#### (\\-?[0-9\\.\\,]+)"`` from the *model's* completion, a format our
    MetaMathQA-trained model never emits (it emits ``The answer is: <v>``). MetaMath is the
    official evaluator for models trained on MetaMathQA, i.e. exactly this setup. Gold is
    built as upstream does: ``int(response.split('#### ')[1].replace(',', ''))``.
    """
    gold = row["answer"].split("#### ")[-1].strip()
    try:
        gold_num = int(gold.replace(",", ""))
    except ValueError:
        return False
    y_pred = _metamath.extract_answer_number(generated)
    if y_pred is None:
        return False
    return float(y_pred) == float(gold_num)


# --------------------------------------------------------------------------------------
# MATH — Hendrycks `is_equiv`, extraction per MetaMath `eval_math.process_results`
# --------------------------------------------------------------------------------------
def _score_math(generated: str, row: Dict) -> bool:
    """Gold = ``remove_boxed(last_boxed_only_string(solution))``; match = Hendrycks `is_equiv`.

    `last_boxed_only_string` comes from the vendored MWPBench copy, which is byte-identical to
    the one in Hendrycks' / MetaMath's `util.py`.
    """
    solution = row.get("solution", row.get("answer", ""))
    boxed = _mwpbench.last_boxed_only_string(solution)
    gold = _metamath.remove_boxed(boxed) if boxed is not None else None
    if gold is None:
        return False
    return _metamath.process_results(row.get("problem", ""), generated, gold)


def _score_omi2(generated: str, row: Dict) -> bool:
    """OpenMathInstruct-2 corpora: Hendrycks `is_equiv` against `expected_answer`.

    NOTE (documented approximation): NVIDIA grades OpenMathInstruct-2 with NeMo-Skills'
    `math_grader`, which is not vendored here. That grader is itself descended from the
    Hendrycks/Minerva normalizer used below, but it is NOT the same code — treat this column
    as "Hendrycks-family", not "official". These are training corpora anyway (see the
    BENCHMARKS note in stage2_eval.py), so the numbers are not Table-4.2 comparable.
    """
    gold = str(row["expected_answer"]).strip()
    return _metamath.process_results(row.get("problem", ""), generated, gold)


# --------------------------------------------------------------------------------------
# MWPBench suite (college-math / gaokao-math-2023) — MathScale `eval_vllm/util.py`
# --------------------------------------------------------------------------------------
def _score_mwpbench(generated: str, row: Dict) -> bool:
    """MWPBench `is_correct(completion, answer)` — self-contained: it lowercases both sides,
    runs its own extraction chain (last ``\\boxed{}`` -> last number if the gold is numeric ->
    ``answer is``/``solution is`` -> trailing ``is``), then `is_equiv` with the 19-step
    `strip_string`. Gold is the raw `answer` field, exactly as upstream passes it.
    """
    judge, _pred, _ref = _mwpbench.is_correct(generated, row["answer"])
    return bool(judge)


# --------------------------------------------------------------------------------------
# DeepMind-Mathematics — DART-Math `EvaluatorMath` (`dart_math/eval.py`)
# --------------------------------------------------------------------------------------
# `use_orig_eq_for_olympiadbench=False` only skips constructing the OlympiadBench judger, which
# `eval` consults solely for datasets named "olympiad*" — see `dartmath_eval.py`'s header.
_DART_MATH = _DartEvaluatorMath(use_orig_eq_for_olympiadbench=False)


def _score_deepmind(generated: str, row: Dict) -> bool:
    """DART-Math's own scorer for its DeepMind-Mathematics subset — the file `deepmind-math`
    loads, at the same pinned commit as the vendored evaluator.

    Mirrors one sample of upstream's `EvaluatorMathBatch.batch_eval`: ``ans =
    str(extract_ans(resp))``, then `EvaluatorMath.eval`, i.e. `eq(ref_ans, ans,
    is_querying4set(query))` — literal, numeric (fraction <-> decimal, percent) and sympy-symbolic
    equality, with one-element-list golds like ``[11]`` matching ``11``. Unlike MWPBench's
    `is_correct` it has no last-number fallback, so e.g. ``\\frac{93}{4}`` is not credited for
    gold ``4``.

    Two upstream behaviours this does NOT reproduce, both because the scorer contract carries
    only the text: DART's harness marks a generation wrong when it stopped at the token limit
    (``finish_reason == "length"``) -- saved generations record no finish reason -- and it gives
    each extraction/match a 5 s process timeout. Faithfully kept: upstream does not strip an
    option letter, so ``(b) 1`` vs gold ``1`` is wrong.
    """
    sample = SimpleNamespace(dataset="deepmind-mathematics", query=row.get("question", ""),
                             ref_ans=row["answer"], resp=generated, finish_reason=None, ans=None)
    sample.ans = str(_DART_MATH.extract_ans(sample.resp))
    return bool(_DART_MATH.eval(sample))


# --------------------------------------------------------------------------------------
# OlympiadBench — OpenBMB `AutoScoringJudge`
# --------------------------------------------------------------------------------------
_OLYMPIAD_JUDGE = None


def _get_olympiad_judge():
    """Construct the official judge lazily (its ``__init__`` calls sympy's `parse_latex`)."""
    global _OLYMPIAD_JUDGE
    if _OLYMPIAD_JUDGE is None:
        from third_party.official_evals.olympiadbench_judge import AutoScoringJudge
        _OLYMPIAD_JUDGE = AutoScoringJudge()
    return _OLYMPIAD_JUDGE


_DART_EXTRACTOR = _DartEvaluatorBase()


def _score_olympiad(generated: str, row: Dict) -> bool:
    """`AutoScoringJudge.judge(gold, pred, precision)` with the row's own tolerance.

    Two upstream-faithful details:

    * **Gold** is ``final_answer`` joined by ``", "`` — the judge splits on commas outside
      brackets and matches the two sides as *multisets*, so join order does not matter.
    * **Precision** is the row's ``error`` field when present, which is exactly how the
      OlympiadBench harness supplies a per-problem tolerance; otherwise the judge's 1e-8
      default.

    Extraction: OpenBMB ship the judge but not an extractor (their harness parses its own
    generation format), so we use DART-Math's `EvaluatorBase.extract_ans` + `clean` — the
    extractor of the suite these files are distributed with, and the one the paper reuses.
    """
    gold = ", ".join(str(s).strip() for s in row["final_answer"])
    pred = _DART_EXTRACTOR.clean(_DART_EXTRACTOR.extract_ans(generated))
    if not pred:
        return False

    precision = 1e-8
    err = row.get("error")
    if err not in (None, "", "None"):
        # upstream accepts a scalar or a comma-separated list, one per answer part
        try:
            parts = [float(e) for e in str(err).split(",")]
            precision = parts if len(parts) > 1 else parts[0]
        except ValueError:
            precision = 1e-8
    try:
        return bool(_get_olympiad_judge().judge(gold, pred, precision))
    except Exception:
        return False


# --------------------------------------------------------------------------------------
# TheoremQA — TIGER-AI-Lab `utils.py` / `number_utils.py`
# --------------------------------------------------------------------------------------
_THEOREMQA_TRIGGERS = ["The answer is:", "The answer is", "the answer is"]  # verbatim from run.py


def _theoremqa_groundtruth(row: Dict) -> List:
    """Rebuild upstream `data_loader.load_dataset`'s ``[str(Answer), numeric_or_None]`` pair.

    Upstream reads a JSON file where ``Answer`` keeps its native Python type and branches on
    ``isinstance``. The HF mirror stores every ``Answer`` as a *string* plus an ``Answer_type``
    column, so we restore the native type from ``Answer_type`` first and then apply upstream's
    exact branch: bool -> no numeric form; list/int/float -> numeric form; anything else
    (``option``) -> no numeric form.
    """
    raw, kind = row["Answer"], row["Answer_type"]
    try:
        if kind == "bool":
            return [str(raw), None]
        if kind == "integer":
            return [str(int(raw)), int(raw)]
        if kind == "float":
            return [str(float(raw)), float(raw)]
        if kind.startswith("list"):
            val = ast.literal_eval(raw)
            return [str(val), val]
    except (ValueError, SyntaxError):
        pass
    return [str(raw), None]


def _score_theoremqa(generated: str, row: Dict) -> bool:
    """`answer_clean` + `compare_answer_with_groundtruth` — the type-aware official matcher.

    Type-aware: yes/true -> ``True`` mapping (115 of the 800 golds are booleans), ``(a)``-style
    option matching by containment, order-insensitive list comparison, a **4% relative** numeric
    window (``eps = abs(gt) * 0.04``) with integers compared as ``round(pred) == gt``, and LaTeX
    answers (``\\frac{\\pi}{2}``, ``362,880``) evaluated through the vendored latex2sympy2.
    """
    pred = _theoremqa.answer_clean(_THEOREMQA_TRIGGERS, generated)
    return bool(_theoremqa.compare_answer_with_groundtruth(pred, *_theoremqa_groundtruth(row)))


# --------------------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------------------
OFFICIAL_SCORERS: Dict[str, Callable[[str, Dict], bool]] = {
    "gsm8k": _score_gsm8k,
    "math": _score_math,
    "gaokao-math-2023": _score_mwpbench,
    "deepmind-math": _score_deepmind,
    "college-math": _score_mwpbench,
    "olympiad-math": _score_olympiad,
    "theoremqa": _score_theoremqa,
    "openmathinstruct-2": _score_omi2,
    "omi2-natural-100k": _score_omi2,
    "omi2-natural-10k": _score_omi2,
}

#: Human-readable provenance, surfaced in the eval output so a number is never anonymous.
SCORER_SOURCE: Dict[str, str] = {
    "gsm8k": "meta-math/MetaMath eval_gsm8k.py (extract_answer_number + float equality)",
    "math": "hendrycks/math is_equiv + MetaMath eval_math.process_results",
    "gaokao-math-2023": "microsoft/unilm MWPBench eval_vllm/util.py is_correct",
    "deepmind-math": "hkust-nlp/dart-math EvaluatorMath (extract_ans + eq)",
    "college-math": "microsoft/unilm MWPBench eval_vllm/util.py is_correct",
    "olympiad-math": "OpenBMB/OlympiadBench AutoScoringJudge + dart-math extraction",
    "theoremqa": "TIGER-AI-Lab/TheoremQA utils.answer_clean + compare_answer_with_groundtruth "
                 "(latex2sympy2 1.9.1, vendored)",
    "openmathinstruct-2": "hendrycks/math is_equiv (APPROXIMATION - not NeMo-Skills math_grader)",
    "omi2-natural-100k": "hendrycks/math is_equiv (APPROXIMATION - not NeMo-Skills math_grader)",
    "omi2-natural-10k": "hendrycks/math is_equiv (APPROXIMATION - not NeMo-Skills math_grader)",
}


def availability(dataset: str) -> Tuple[bool, str]:
    """Can `dataset`'s official scorer run at full fidelity here? -> (ok, reason).

    Never guesses: it actually constructs the objects whose optional dependencies are in
    question, so a missing antlr4 is reported before a sweep runs, not after.
    """
    if dataset not in OFFICIAL_SCORERS:
        return False, f"no official scorer registered for {dataset!r}"
    if dataset == "olympiad-math":
        try:
            _get_olympiad_judge()
        except Exception as e:
            return False, (
                f"OlympiadBench judge unavailable ({type(e).__name__}: {str(e)[:80]}). It needs "
                "sympy's LaTeX parser: uv pip install 'antlr4-python3-runtime==4.11.1'"
            )
    if dataset == "deepmind-math":
        # DART's `EvaluatorMath` wraps every `parse_latex` call in try/except, so without the
        # parser it would not fail -- it would silently stop matching LaTeX answers.
        try:
            from sympy.parsing.latex import parse_latex
            assert float(parse_latex(r"\frac{1}{2}")) == 0.5  # returned unevaluated: compare value
        except Exception as e:
            return False, (
                f"DART-Math EvaluatorMath unavailable ({type(e).__name__}: {str(e)[:80]}). It "
                "needs sympy's LaTeX parser: uv pip install 'antlr4-python3-runtime==4.11.1'"
            )
    # TheoremQA needs nothing optional: its latex2sympy2 (and the antlr4 4.7.2 runtime that one
    # needs) is vendored in `third_party/official_evals/_latex2sympy2/`, so it is always at full
    # fidelity.
    return True, "ok"


def score(dataset: str, generated: str, row: Dict) -> Optional[bool]:
    """Score one generation with `dataset`'s official evaluator.

    Returns None when no official scorer is registered. Any exception raised inside an upstream
    evaluator is caught and scored as incorrect: upstream harnesses treat an unparseable
    prediction as wrong, and a crash mid-sweep would otherwise lose a whole GPU pass.
    """
    scorer = OFFICIAL_SCORERS.get(dataset)
    if scorer is None:
        return None
    try:
        return bool(scorer(generated, row))
    except Exception:
        return False


def fidelity_report(dataset: str) -> str:
    ok, reason = availability(dataset)
    status = "ok" if ok else "unavailable"
    return f"[official-eval] {dataset}: {status} — {SCORER_SOURCE.get(dataset, '?')}" + (
        "" if reason == "ok" else f"\n[official-eval]   {reason}"
    )
