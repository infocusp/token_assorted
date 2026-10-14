"""VERBATIM EXCERPT -- DO NOT EDIT. See PROVENANCE.md.

Upstream: https://github.com/meta-math/MetaMath  @ 913a0b2349ba849dc926b66689cd6ae72dd7c6fc
Paths:    eval_gsm8k.py (`is_number`, `extract_answer_number`)
          eval_math.py  (`remove_boxed`, `process_results`)
License:  Apache-2.0

Why an excerpt rather than the whole modules: both upstream files do `from vllm import LLM`
at module scope, so importing them requires a GPU inference stack we do not use here. The
functions below are copied byte-for-byte from those files.

ONE documented substitution: upstream does `from fraction import Fraction` (the third-party
`fraction` package). We use the standard library's `fractions.Fraction`, which parses the same
"a/b" strings identically for this code path, to avoid adding an unmaintained dependency.

`process_results` is reproduced with its `util.is_equiv` call rebound to the vendored
Hendrycks `math_equivalence.is_equiv`. MetaMath's `util.py` is Hendrycks' `math_equivalence.py`
copied with the leading underscores removed from the helper names (`_strip_string` ->
`strip_string`); the bodies, including the 19 normalization steps and the `0.5 -> \\frac{1}{2}`
rule, are identical, so this rebinding is a no-op semantically.
"""
import re
from fractions import Fraction  # upstream: `from fraction import Fraction`

from .hendrycks_math_equivalence import is_equiv as _hendrycks_is_equiv


# ---- eval_gsm8k.py --------------------------------------------------------------------
def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
    return False


def extract_answer_number(completion):
    text = completion.split('The answer is: ')
    if len(text) > 1:
        extract_ans = text[-1].strip()
        match = re.search(r'[\-+]?\d*[\.,/]?\d+', extract_ans)
        if match:
            if '/' in match.group():
                denominator = match.group().split('/')[1]
                numerator = match.group().split('/')[0]
                if is_number(denominator) == True and is_number(numerator) == True:
                    if denominator == '0':
                        return round(float(numerator.replace(',', '')))
                    else:
                        frac = Fraction(match.group().replace(',', ''))
                        num_numerator = frac.numerator
                        num_denominator = frac.denominator
                        return round(float(num_numerator / num_denominator))
                else:
                    return None
            else:
                if float(match.group().replace(',', '')) == float('inf'):
                    return None
                return round(float(match.group().replace(',', '')))
        else:
            return None
    else:
        return None


# ---- eval_math.py ---------------------------------------------------------------------
def remove_boxed(s):
    left = "\\boxed{"
    try:
        assert s[:len(left)] == left
        assert s[-1] == "}"
        return s[len(left):-1]
    except:
        return None


def process_results(doc, completion, answer):
    split_ans = completion.split('The answer is: ')
    if len(split_ans) > 1:
        ans = split_ans[-1]
        extract_ans_temp = ans.split('.\n')[0]
        extract_ans_temp = extract_ans_temp.strip()
        if len(extract_ans_temp) > 0 and extract_ans_temp[-1] == '.':
            extract_ans = extract_ans_temp[0:-1]
        else:
            extract_ans = extract_ans_temp
        extract_ans = extract_ans.strip()
        if _hendrycks_is_equiv(extract_ans, answer):
            return True
        else:
            return False
    else:
        return False
