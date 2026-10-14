"""VERBATIM EXCERPT -- DO NOT EDIT. See PROVENANCE.md.

Upstream: https://github.com/hkust-nlp/dart-math  @ 219b32467b55f7c610195e148b56df3431e1a963
Path:     dart_math/eval.py   (lines 38-215: constants, extract_boxed, norm_str2bool, EvaluatorBase)
License:  MIT

Only the ANSWER-EXTRACTION half of the upstream module is excerpted here. The *matching* half
(EvaluatorMath, lines 288-1399) is excerpted separately in `dartmath_eval.py`, which builds on
this file and scores deepmind-math; OlympiadBench matching is done by the upstream OpenBMB judge
in `olympiadbench_judge.py` instead. Excerpted lines are byte-identical.

`from __future__ import annotations` is prepended (not part of the excerpt) so that the
`RespSampleBase` annotation on the unused `EvaluatorBase.eval` stays a lazy string instead of
requiring the upstream `.data` import. It changes no runtime behaviour of the excerpt.
"""
from __future__ import annotations

import re

STRIP_STRS = [
    ":",
    ".",
    "/",
    ",",
    "#",
    "?",
    "$",
    '"',
    "'",
    # "ки" is the delimeter for Math-Shepherd
    "к",
    "и",
    # LaTeX
    "\\(",
    "\\)",
    "\\[",
    "\\]",
]
NO_TRAILING_STRS = ["(", "[", "{", "\\"] + STRIP_STRS
NO_PRECEDING_PUNCS = ["!", ")", "]", "}", "\\\\"] + STRIP_STRS
# Answer prefixes
PRM800K_ANS_PRRFIX = "# Answer"
GSM8K_ANS_PREFIX = "####"


# %% ../nbs/04_eval.ipynb 0
def extract_boxed(resp: str) -> str:
    ans = resp.split("oxed")[-1]
    if len(ans) == 0:
        return ""
    elif ans[0] == "{":
        stack = 1
        a = ""
        for i_pre, c in enumerate(ans[1:]):
            if ans[i_pre] == "\\":
                a += c
                continue
            if c == "{":
                stack += 1
                a += c
            elif c == "}":
                stack -= 1
                if stack == 0:
                    break
                a += c
            else:
                a += c
    else:
        a = ans.split("$")[0].strip()
    return a


def norm_str2bool(s: str) -> bool | None:
    """Converts a string representation of a boolean value to its corresponding boolean value."""
    s = str(s).lower().strip().replace("noindent", "")
    if any(pos in s for pos in ["yes", "true"]):
        return True
    elif any(neg in s for neg in ["no", "false"]):
        return False
    else:
        return None


class EvaluatorBase:
    """Base class for evaluators.

    Parameters
    ----------
    strict_extract: bool, default: False
        Whether to extract answers strictly. If `False`, speculate the answer from the last number if needed.
    """

    def __init__(
        self,
        strict_extract: bool = False,
    ):
        self.strict_extract = strict_extract

    def eval(self, sample: RespSampleBase) -> bool:
        """Evaluate a sample based on comprehensive information."""
        ans = sample.ans if sample.ans is not None else self.extract_ans(sample.resp)
        correct = self.eq(sample.ref_ans, ans)
        return correct

    def eq(self, ref: str, ans: str) -> bool:
        """Check if reference answer and prediction answer are **literally** equal."""
        return ref == ans

    def extract_ans(self, resp_str: str) -> str:
        """Extract answer segment from complete `resp`."""
        ans = self.extract_explicit_ans(resp_str)
        if ans is not None:
            return ans
        elif not self.strict_extract:
            # Speculate with the last latex formula
            matches = re.findall(
                r"(?:\$|\\\(|\\\[)([^\$]+)(?:\$|\\\(|\\\[)", resp_str, re.DOTALL
            )
            if len(matches) > 0:
                return matches[-1]
            # Speculate with the last number
            matches = re.findall(r"-?\d*\.?\d+", resp_str.replace(",", ""))
            if len(matches) > 0:
                return matches[-1]
        return ""  # Empty str if no answer is found

    def extract_explicit_ans(self, resp_str: str) -> str:
        resp_str = self.clean_trailing(resp_str)
        # might be answer only
        if "herefore" in resp_str:
            resp_str = resp_str.split("herefore")[-1].strip()
        if GSM8K_ANS_PREFIX in resp_str:
            resp_str = resp_str.split(GSM8K_ANS_PREFIX)[-1].strip()
        if PRM800K_ANS_PRRFIX in resp_str:
            resp_str = resp_str.split(PRM800K_ANS_PRRFIX)[-1].strip()

        if "oxed{" in resp_str:
            resp = extract_boxed(resp_str)
        else:
            resp = resp_str

            # should be answer only
            if "is the ans" in resp:
                resp = re.split(r"(,|\.|\!\|?)", resp.split("is the ans")[-2].strip())[
                    -1
                ].strip()
            elif "is our ans" in resp:
                resp = re.split(r"(,|\.|\!\|?)", resp.split("is our ans")[-2].strip())[
                    -1
                ].strip()
            elif "answer is" in resp:
                resp = resp.split("answer is")[-1].strip()
            elif "answer:" in resp:
                resp = resp.split("answer:")[-1].strip()
            elif "answer :" in resp:
                resp = resp.split("answer :")[-1].strip()
            elif "statement" in resp:
                bool_resp = norm_str2bool(resp.split("is ")[-1].strip())
                if bool_resp is not None:
                    return str(bool_resp)
            else:
                return None

            if resp.startswith("$") and resp.endswith("$"):
                resp = resp[1:-1]

        return resp

    def clean(self, ans: str) -> str:
        """Clean the extracted answer."""

        ans = ans.strip()
        ans = self.clean_preceding(ans)
        ans = self.clean_trailing(ans)

        return ans

    def clean_preceding(
        self,
        s: str,  # The input string.
    ) -> str:  # The cleaned string with preceding punctuation marks removed.
        """Removes preceding punctuation marks from a string."""
        s = str(s).strip()
        while s != "" and s[0] in NO_PRECEDING_PUNCS:
            s = s[1:].strip()

        return s

    def clean_trailing(
        self,
        s: str,  # The input string.
    ) -> str:  # The cleaned string with trailing punctuation marks removed.
        """Removes trailing punctuation marks from a string."""
        s = str(s).strip()
        while s != "" and s[-1] in NO_TRAILING_STRS:
            s = s[:-1].strip()
        return s
