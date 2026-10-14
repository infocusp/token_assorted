"""Vendored **official** benchmark evaluators (answer extraction + answer matching).

Every module in this package is upstream code, copied at a pinned commit. **Do not edit them.**
All adaptation lives in `src/official_eval.py`, which imports and composes these. See
`PROVENANCE.md` for the per-file upstream URL, commit SHA and license.

This `__init__` installs two import shims so the vendored files can stay byte-identical to
upstream despite being imported as a package rather than as top-level scripts:

1. `theoremqa_utils.py` does ``from number_utils import *`` -- a *top-level* import that only
   resolves when the file sits in its own repo root. We pre-register the vendored
   `theoremqa_number_utils` module under the name ``number_utils``.
2. Both TheoremQA files do ``from latex2sympy2 import latex2sympy``. That name resolves to the
   copy vendored in `_latex2sympy2/` (latex2sympy2 1.9.1 on a private antlr4 4.7.2 runtime), so
   TheoremQA runs at full fidelity in the same environment as the OlympiadBench judge, whose
   sympy ``parse_latex`` needs the installed antlr4 4.11. The vendored module is registered as
   ``latex2sympy2`` only while the two TheoremQA modules are imported, then the previous entry
   (normally none) is restored -- so an installed ``latex2sympy2`` can never be picked up here,
   and importing this package never shadows one for anybody else.
"""
import sys
import warnings

# Upstream writes `"\%"` etc. in non-raw strings, which Python 3.12+ flags. These are upstream's
# to fix, and we keep the files byte-identical, so silence the noise rather than edit them.
warnings.filterwarnings("ignore", category=SyntaxWarning, module=r"third_party\.official_evals.*")

from ._latex2sympy2 import latex2sympy2 as _latex2sympy2  # noqa: E402

_prev_latex2sympy2 = sys.modules.get("latex2sympy2")
sys.modules["latex2sympy2"] = _latex2sympy2
try:
    from . import theoremqa_number_utils as _theoremqa_number_utils  # noqa: E402

    # `theoremqa_utils.py` resolves `number_utils` as a top-level module name.
    sys.modules.setdefault("number_utils", _theoremqa_number_utils)
    from . import theoremqa_utils as _theoremqa_utils  # noqa: E402, F401
finally:
    if _prev_latex2sympy2 is None:
        del sys.modules["latex2sympy2"]
    else:
        sys.modules["latex2sympy2"] = _prev_latex2sympy2
