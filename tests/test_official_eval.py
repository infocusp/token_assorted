"""The official per-benchmark scorers (`src/official_eval.py`) — Stage-2 eval's only scorer.

Two things are being protected here:

1. Each benchmark routes to its OWN upstream evaluator and reproduces the behaviour that
   evaluator is known for (degree/unit stripping, 4%-relative tolerance, bool/option/list
   handling, set matching).
2. A broken scorer can never masquerade as a real score: `evaluate()` refuses to run rather than
   report a plausible-looking `accuracy_official == 0.0`.
"""
import pytest

from src.official_eval import OFFICIAL_SCORERS, availability, score
from src.stage2_eval import BENCHMARKS


def _gen(answer: str) -> str:
    """A generation in the format this repo's Stage-2 model actually emits."""
    return f"Some reasoning here. The answer is: {answer}"


# ---------------------------------------------------------------------------------------
# Registry coverage
# ---------------------------------------------------------------------------------------
def test_every_benchmark_has_an_official_scorer():
    missing = set(BENCHMARKS) - set(OFFICIAL_SCORERS)
    assert not missing, f"benchmarks with no official scorer: {sorted(missing)}"


def test_availability_reports_a_reason_for_every_scorer():
    for ds in OFFICIAL_SCORERS:
        ok, reason = availability(ds)
        assert isinstance(ok, bool) and reason, ds


def test_unregistered_dataset_scores_none():
    assert score("not-a-benchmark", _gen("1"), {}) is None


# ---------------------------------------------------------------------------------------
# GSM8K — MetaMath extract_answer_number + float equality
# ---------------------------------------------------------------------------------------
@pytest.mark.parametrize("answer,gold,expected", [
    ("18", "#### 18", True),
    ("18.0", "#### 18", True),
    ("1,000", "#### 1000", True),
    ("18 apples", "#### 18", True),    # regex takes the number
    ("17", "#### 18", False),
])
def test_gsm8k_official(answer, gold, expected):
    assert score("gsm8k", _gen(answer), {"answer": f"reasoning\n{gold}"}) is expected


def test_gsm8k_official_accepts_a_unit_suffix():
    row = {"answer": "reasoning\n#### 18"}
    assert score("gsm8k", _gen("18 apples"), row) is True


# ---------------------------------------------------------------------------------------
# MATH — Hendrycks is_equiv (the 19-step _strip_string)
# ---------------------------------------------------------------------------------------
@pytest.mark.parametrize("answer,gold_sol,expected", [
    ("39", r"so \boxed{39^\circ}", True),            # ^\circ stripped
    ("40", r"so \boxed{\$40}", True),                # \$ stripped
    ("575", r"so \boxed{575\text{ students}}", True),  # units stripped
    (r"\dfrac{1}{3}", r"so \boxed{\frac{1}{3}}", True),  # dfrac -> frac
    ("7", r"so \boxed{39}", False),
])
def test_math_official(answer, gold_sol, expected):
    assert score("math", _gen(answer), {"problem": "q", "solution": gold_sol}) is expected


def test_math_official_strips_a_latex_dollar():
    assert score("math", _gen("40"), {"problem": "q", "solution": r"\boxed{\$40}"}) is True


# ---------------------------------------------------------------------------------------
# MWPBench suite — college-math / gaokao-math-2023
# ---------------------------------------------------------------------------------------
@pytest.mark.parametrize("dataset", ["college-math", "gaokao-math-2023"])
def test_mwpbench_official_basic_numeric(dataset):
    assert score(dataset, _gen("12"), {"answer": "$12$"}) is True
    assert score(dataset, _gen("13"), {"answer": "$12$"}) is False


def test_college_math_official_strips_left_right():
    row = {"answer": r"$2\left|-3\right|+2$"}
    assert score("college-math", _gen(r"2|-3|+2"), row) is True


def test_college_math_official_strips_degree_marks():
    assert score("college-math", _gen("29"), {"answer": r"$29^{\circ}$"}) is True


# ---------------------------------------------------------------------------------------
# DeepMind-Mathematics — DART-Math EvaluatorMath
# ---------------------------------------------------------------------------------------
# Every case below is a real MWPBench-vs-DART disagreement on saved Stage-2 generations.
def _dm(answer):
    return {"question": "Calculate the value.", "answer": answer}


needs_dart = pytest.mark.skipif(not availability("deepmind-math")[0],
                                reason=availability("deepmind-math")[1])


@needs_dart
@pytest.mark.parametrize("pred, gold", [
    ("12", "$12$"),
    (r"-\frac{1}{2}", "-0.5"),         # fraction <-> decimal (MWPBench missed these)
    ("-58/5", "-11.6"),
    ("11", "[11]"),                    # one-element list gold ("list the prime factors")
    ("-3-8+2+4", "-5"),                # unevaluated arithmetic, numerically equal
])
def test_deepmind_official_dart_equivalences(pred, gold):
    assert score("deepmind-math", _gen(pred), _dm(gold)) is True


@needs_dart
@pytest.mark.parametrize("pred, gold", [
    ("13", "12"),
    (r"\frac{93}{4}", "4"),            # MWPBench's last-number fallback credited these
    (r"-\frac{5}{2}", "2"),
    ("(b)1", "1"),                     # upstream keeps the option letter: faithfully wrong
])
def test_deepmind_official_dart_rejections(pred, gold):
    assert score("deepmind-math", _gen(pred), _dm(gold)) is False


def test_deepmind_reports_unavailable_without_sympy_latex_parser(monkeypatch):
    """DART swallows parse errors, so a missing antlr4 must surface here, not as silent misses."""
    import sympy.parsing.latex

    def _no_parser(*a, **k):
        raise ImportError("LaTeX parsing requires the antlr4 Python package")
    monkeypatch.setattr(sympy.parsing.latex, "parse_latex", _no_parser)
    ok, reason = availability("deepmind-math")
    assert ok is False and "antlr4" in reason


def test_dartmath_eval_excerpt_is_verbatim():
    """The BEGIN/END region of `dartmath_eval.py` is upstream `dart_math/eval.py` lines 288-1399
    @ 219b324, byte for byte."""
    import hashlib
    from pathlib import Path

    text = (Path(__file__).resolve().parent.parent / "third_party" / "official_evals"
            / "dartmath_eval.py").read_text(encoding="utf-8")
    begin = "# ---- BEGIN VERBATIM: dart_math/eval.py lines 288-1399 ----\n"
    body = text.split(begin, 1)[1].rsplit("# ---- END VERBATIM ----\n", 1)[0]
    assert hashlib.sha256(body.encode("utf-8")).hexdigest() == \
        "20022d88224f6e4314169189b41cfff58f9920b0b5b7c3cf14d1ade28fcf9f12"


# ---------------------------------------------------------------------------------------
# OlympiadBench — AutoScoringJudge (set matching + per-row tolerance)
# ---------------------------------------------------------------------------------------
def _olympiad_row(final_answer, error=None):
    return {"final_answer": final_answer, "error": error}


@pytest.mark.skipif(not availability("olympiad-math")[0],
                    reason=availability("olympiad-math")[1])
def test_olympiad_official_exact():
    assert score("olympiad-math", _gen("2"), _olympiad_row(["2"])) is True
    assert score("olympiad-math", _gen("3"), _olympiad_row(["2"])) is False


@pytest.mark.skipif(not availability("olympiad-math")[0],
                    reason=availability("olympiad-math")[1])
def test_olympiad_official_multi_answer_is_order_insensitive():
    """The judge matches comma-separated parts as a multiset, so order does not matter."""
    row = _olympiad_row(["2", "3"])
    assert score("olympiad-math", _gen("3, 2"), row) is True


@pytest.mark.skipif(not availability("olympiad-math")[0],
                    reason=availability("olympiad-math")[1])
def test_olympiad_official_uses_the_row_tolerance():
    """`error` is the per-problem tolerance the OlympiadBench harness supplies."""
    assert score("olympiad-math", _gen("1.05"), _olympiad_row(["1.0"], error="0.1")) is True
    assert score("olympiad-math", _gen("1.05"), _olympiad_row(["1.0"])) is False


# ---------------------------------------------------------------------------------------
# TheoremQA — type-aware matching
# ---------------------------------------------------------------------------------------
def _tqa(answer, kind):
    return {"Answer": answer, "Answer_type": kind}


def test_theoremqa_official_bool_mapping():
    """115 of the 800 golds are booleans; 'yes'/'no' must map onto them."""
    assert score("theoremqa", _gen("yes"), _tqa("True", "bool")) is True
    assert score("theoremqa", _gen("no"), _tqa("False", "bool")) is True
    assert score("theoremqa", _gen("yes"), _tqa("False", "bool")) is False


def test_theoremqa_official_four_percent_relative_tolerance():
    """eps = abs(gt) * 0.04 — a relative window, not an absolute one."""
    assert score("theoremqa", _gen("1.03"), _tqa("1.0", "float")) is True
    assert score("theoremqa", _gen("1.30"), _tqa("1.0", "float")) is False


def test_theoremqa_official_integers_are_rounded():
    assert score("theoremqa", _gen("5.2"), _tqa("5", "integer")) is True


def test_theoremqa_official_lists_are_order_insensitive():
    assert score("theoremqa", _gen("(1, 2)"), _tqa("[2, 1]", "list of integer")) is True
    assert score("theoremqa", _gen("(1, 3)"), _tqa("[2, 1]", "list of integer")) is False


def test_theoremqa_official_option_matching():
    assert score("theoremqa", _gen("(a)"), _tqa("(a)", "option")) is True
    assert score("theoremqa", _gen("(b)"), _tqa("(a)", "option")) is False


# ---------------------------------------------------------------------------------------
# Robustness: a scorer must never take down a sweep
# ---------------------------------------------------------------------------------------
@pytest.mark.parametrize("dataset", sorted(OFFICIAL_SCORERS))
def test_scorers_never_raise_on_junk(dataset):
    """Upstream evaluators are not defensive; `score` must absorb whatever they throw."""
    for row in ({}, {"answer": ""}, {"final_answer": []}, {"Answer": "x", "Answer_type": "float"}):
        assert score(dataset, "", row) in (True, False)
        assert score(dataset, "\\boxed{", row) in (True, False)


# ---------------------------------------------------------------------------------------
# A missing dependency must never masquerade as a real score
# ---------------------------------------------------------------------------------------
def test_unavailable_scorer_refuses_to_evaluate(monkeypatch):
    """`score()` grades an unparseable prediction as wrong, so a broken scorer would otherwise
    produce a perfectly plausible `accuracy_official == 0.0`. It is the only metric, so
    `evaluate()` must refuse to run at all — before spending any generation on it.

    Reuses the stub model/tokenizer from test_eval_registry so no GPU or weights are needed.
    """
    from tests.test_eval_registry import _StubModel, _StubTok, _stub_gold, _stub_rows
    import src.stage2_eval as ev

    # stage2_eval binds `availability` at import, so patch it where evaluate() looks it up
    monkeypatch.setattr(ev, "availability", lambda ds: (False, "simulated missing dependency"))

    with pytest.raises(RuntimeError, match="simulated missing dependency"):
        ev.evaluate(_StubModel(), _StubTok(), _stub_rows((5, 1, 9)), lambda r: r["q"],
                    _stub_gold, "cpu", max_new_tokens=4, batch_size=2, dataset="gsm8k")


# ---------------------------------------------------------------------------------------
# scripts/rescore_generations.py — the offline re-scoring path
# ---------------------------------------------------------------------------------------
def _load_rescore_module():
    import importlib.util
    import pathlib
    path = pathlib.Path(__file__).parent.parent / "scripts" / "rescore_generations.py"
    spec = importlib.util.spec_from_file_location("rescore_generations", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("name,expected", [
    ("eval_theoremqa_generations.jsonl", "theoremqa"),
    ("eval_college-math_n100_generations.jsonl", "college-math"),
    ("eval_gsm8k_n50_seed0_mnt1024_generations.jsonl", "gsm8k"),
])
def test_rescore_infers_dataset_from_filename(name, expected):
    assert _load_rescore_module()._infer_dataset(f"runs/adapter/{name}") == expected


def test_rescore_rejects_the_legacy_index_only_format(tmp_path):
    """The first version of the format stored `row` as the row INDEX. A key-presence check
    passes on it, every official scorer then fails on an int, and the run reports a
    plausible-looking accuracy_official=0.0. The guard must be type-based, so this must exit."""
    import json
    legacy = tmp_path / "eval_gsm8k_generations.jsonl"
    legacy.write_text(json.dumps({
        "row": 0, "gold": "18", "generated": "The answer is: 18",
        "pred_repo": "18", "correct_repo": True, "correct_official": True,
    }) + "\n")

    mod = _load_rescore_module()
    monkey_argv = ["rescore_generations.py", str(legacy)]
    import sys as _sys
    old = _sys.argv
    _sys.argv = monkey_argv
    try:
        with pytest.raises(SystemExit):
            mod.main()
    finally:
        _sys.argv = old


def test_rescore_matches_evaluate_on_the_current_format(tmp_path):
    """A self-contained record must re-score to exactly what `evaluate` reported inline."""
    import json
    f = tmp_path / "eval_gsm8k_generations.jsonl"
    with open(f, "w") as fh:
        for ans, gold in (("18", "18"), ("17", "18"), ("18 apples", "18")):
            fh.write(json.dumps({
                "row_idx": 0, "row": {"answer": f"reasoning\n#### {gold}"}, "gold": gold,
                "generated": f"The answer is: {ans}", "correct_official": None,
            }) + "\n")

    mod = _load_rescore_module()
    out = tmp_path / "m.json"
    import sys as _sys
    old = _sys.argv
    _sys.argv = ["rescore_generations.py", str(f), "--out", str(out)]
    try:
        mod.main()
    finally:
        _sys.argv = old

    m = json.loads(out.read_text())
    # official: "18" ok, "17" wrong, "18 apples" OK (regex takes the number) -> 2/3
    assert m["accuracy_official"] == pytest.approx(2 / 3)
    assert "accuracy" not in m        # the retired in-repo scale is never written
