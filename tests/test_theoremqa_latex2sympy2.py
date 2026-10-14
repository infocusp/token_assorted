"""TheoremQA's official scorer at full fidelity, via the vendored latex2sympy2.

`latex2sympy2` pins antlr4 4.7.2, so it cannot be installed alongside the 4.11 runtime the
OlympiadBench judge needs. Without it, the TheoremQA scorer silently marks LaTeX, comma-grouped and
bare-list answers wrong, and can mark a truncated generation right via ``eval('False') == 0.0``.
It is vendored on a private antlr4 runtime (`scripts/vendor_latex2sympy2.py`). These tests protect:

1. the vendored tree is byte-for-byte what the script generated (MANIFEST.json);
2. both TheoremQA modules bind to the vendored copy, and nothing leaks into the global module
   namespace (no `latex2sympy2`, no top-level `antlr4` served from the private runtime);
3. the answers that need latex2sympy2 — every case below is a real disagreement between the scorer
   without it and the upstream full-fidelity scorer, found on saved Stage-2 generations — score as
   upstream;
4. TheoremQA and OlympiadBench run at full fidelity in ONE process.
"""
import hashlib
import json
import sys
from pathlib import Path

import pytest

from src.official_eval import availability, score
from third_party.official_evals import theoremqa_number_utils, theoremqa_utils

VENDOR = Path(__file__).resolve().parent.parent / "third_party" / "official_evals" / "_latex2sympy2"
VENDOR_PKG = "third_party.official_evals._latex2sympy2"


def _gen(answer: str) -> str:
    return f"Some reasoning here. The answer is: {answer}"


def _tqa(answer: str, kind: str) -> dict:
    """A TheoremQA row as the HF mirror stores it: `Answer` is always a string."""
    return {"Answer": answer, "Answer_type": kind}


# ---------------------------------------------------------------------------------------
# 1. Integrity of the vendored tree
# ---------------------------------------------------------------------------------------
def test_vendored_tree_matches_manifest():
    manifest = json.loads((VENDOR / "MANIFEST.json").read_text())
    on_disk = {
        p.relative_to(VENDOR).as_posix() for p in VENDOR.rglob("*")
        if p.is_file() and "__pycache__" not in p.parts
    } - {"__init__.py", ".gitattributes", "MANIFEST.json"}
    assert on_disk == set(manifest["files"]), "files added/removed by hand — re-run the script"
    for rel, sha in manifest["files"].items():
        assert hashlib.sha256((VENDOR / rel).read_bytes()).hexdigest() == sha, \
            f"{rel} edited by hand — re-run scripts/vendor_latex2sympy2.py"


def test_manifest_pins_the_upstream_versions():
    src = json.loads((VENDOR / "MANIFEST.json").read_text())["sources"]
    assert src["latex2sympy2"]["version"] == "1.9.1"
    assert src["antlr4-python3-runtime"]["version"] == "4.7.2"


# ---------------------------------------------------------------------------------------
# 2. Wiring and isolation
# ---------------------------------------------------------------------------------------
def test_theoremqa_modules_bind_the_vendored_latex2sympy2():
    for mod in (theoremqa_number_utils, theoremqa_utils):
        assert mod.latex2sympy.__module__ == f"{VENDOR_PKG}.latex2sympy2", mod.__name__


def test_vendoring_does_not_leak_module_names():
    leaked = sys.modules.get("latex2sympy2")
    assert leaked is None or not getattr(leaked, "__name__", "").startswith(VENDOR_PKG)
    for name, mod in list(sys.modules.items()):
        if name == "antlr4" or name.startswith("antlr4."):
            assert "_latex2sympy2" not in (getattr(mod, "__file__", None) or ""), name
    assert any(n.startswith(f"{VENDOR_PKG}.antlr4") for n in sys.modules), \
        "the private runtime should be what latex2sympy2 loaded"


def test_theoremqa_is_always_full_fidelity():
    assert availability("theoremqa") == (True, "ok")


# ---------------------------------------------------------------------------------------
# 3. Answers that need latex2sympy2 (real saved-generation disagreements)
# ---------------------------------------------------------------------------------------
@pytest.mark.parametrize("answer, gold, kind", [
    ("362,880", "362880", "integer"),                      # comma-grouped integer
    (r"\frac{\pi}{2}", "1.570796", "float"),               # symbolic constant, 4% window
    (r"\frac{5}{6}", "0.8333", "float"),
    (r"\frac{1}{2}", "0.5", "float"),
    (r"\frac{1}{1}", "1.0", "float"),
    ("1,2,4,5,6", "[1, 2, 4, 5, 6]", "list of integer"),   # bare list -> latex2sympy tuple
])
def test_theoremqa_latex_answers_now_match(answer, gold, kind):
    assert score("theoremqa", _gen(answer), _tqa(gold, kind)) is True


@pytest.mark.parametrize("answer, gold, kind", [
    (r"\frac{1}{3}", "0.5", "float"),
    ("362,881", "362880", "integer"),
    ("1,2,4,5", "[1, 2, 4, 5, 6]", "list of integer"),
])
def test_theoremqa_latex_answers_still_reject_wrong_values(answer, gold, kind):
    assert score("theoremqa", _gen(answer), _tqa(gold, kind)) is False


def test_theoremqa_no_answer_is_not_credited_as_zero():
    """A truncated generation with no answer trigger: upstream extracts 'False' (from 'no' in
    the text). Without latex2sympy2, ``number_it`` falls back to ``eval('False') -> 0.0`` and
    credits a gold of 0.0; with it, 'False' is not a number, so it is (correctly) wrong."""
    truncated = ("The expression $(2n+1)$ cancels out, leaving us with no remaining terms "
                 r"and $\frac{2n")
    assert score("theoremqa", truncated, _tqa("0.0", "float")) is False


# ---------------------------------------------------------------------------------------
# 4. One process, both scorers at full fidelity
# ---------------------------------------------------------------------------------------
@pytest.mark.skipif(not availability("olympiad-math")[0],
                    reason=availability("olympiad-math")[1])
def test_olympiad_and_theoremqa_share_one_process():
    """sympy's parse_latex (antlr4 4.11) and latex2sympy2 (vendored antlr4 4.7.2), interleaved."""
    olympiad_row = {"final_answer": [r"$\frac{1}{2}$"], "error": None}
    for _ in range(2):
        assert score("theoremqa", _gen(r"\frac{1}{2}"), _tqa("0.5", "float")) is True
        assert score("olympiad-math", _gen(r"\frac{2}{4}"), olympiad_row) is True
        assert score("olympiad-math", _gen(r"\frac{1}{3}"), olympiad_row) is False
