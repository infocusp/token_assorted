"""Stage-2 Sol-Only data mode.

Locks the shape produced by ``Stage2Dataset(..., sol_only=True)``: each example must be
``[BOS] + prompt + solution + [EOS]`` with NO latent block and NO CoT text — the paper's Sol-Only
baseline (Table 4.2). Contrasts it with the CoT-only path (``sol_only=False, m_max_set=[0]``), which
also has no latent block but MUST keep the full CoT text — proving sol_only genuinely drops the CoT.

Uses a tiny stub tokenizer (no network / no HF weights) so this stays CPU-only and fast.
"""
from types import SimpleNamespace

from src.stage2_data import LatentVocab, Stage2Dataset

# Disjoint id ranges so each span is identifiable in the assembled sequence.
_BOS, _EOS = 1, 2
_BO, _EO = 1000, 1001                       # latent delimiters
_CODE_BASE = 2000                           # code_id[c] = 2000 + c
_COT_BASE = 500                             # sentinel CoT-token ids (never produced by the stub tok)


class _FakeTok:
    """Maps a string to a deterministic list of base-vocab ids in [10, 110)."""

    def __call__(self, text, add_special_tokens=False):
        return SimpleNamespace(input_ids=[ord(ch) % 100 + 10 for ch in text])


def _vocab(codebook_size=8):
    return LatentVocab(
        bo_id=_BO, eo_id=_EO, bos_id=_BOS, eos_id=_EOS,
        code_id=[_CODE_BASE + c for c in range(codebook_size)],
        n_added=codebook_size + 2, base_vocab_size=200,
    )


def _record():
    return {
        "prompt": "question?",
        "answer": "The answer is: 42",
        "cot_token_ids": [_COT_BASE + k for k in range(64)],   # 64 sentinel CoT tokens
        "code_indices": [0, 1, 2, 3, 4, 5, 6, 7],
    }


def test_sol_only_shape():
    tok = _FakeTok()
    v = _vocab()
    ds = Stage2Dataset([_record()], tok, v, m_max_set=[256], chunk_length=16, seed=42, sol_only=True)
    item = ds[0]
    ids, labels = item["input_ids"], item["labels"]

    prompt_ids = tok("question?").input_ids
    answer_ids = tok("The answer is: 42").input_ids

    # Exact sequence: [BOS] + prompt + answer + [EOS] — nothing else.
    assert ids == [_BOS] + prompt_ids + answer_ids + [_EOS]

    # No latent block and no CoT text leaked in.
    assert _BO not in ids and _EO not in ids
    assert all(c not in ids for c in v.code_id)
    assert all((_COT_BASE + k) not in ids for k in range(64))

    # Labels: -100 over [BOS] + prompt, real over answer + EOS.
    n_mask = 1 + len(prompt_ids)
    assert labels[:n_mask] == [-100] * n_mask
    assert labels[n_mask:] == answer_ids + [_EOS]
    assert item["attention_mask"] == [1] * len(ids)


def test_cot_only_keeps_cot_but_sol_only_drops_it():
    """sol_only=False with m_max_set=[0] => no latent block but FULL CoT retained; sol_only=True
    with the same data drops that CoT entirely. This distinguishes the two 'no-latent' modes."""
    tok = _FakeTok()
    v = _vocab()
    rec = _record()

    cot = Stage2Dataset([rec], tok, v, m_max_set=[0], chunk_length=16, seed=42, sol_only=False)[0]
    sol = Stage2Dataset([rec], tok, v, m_max_set=[0], chunk_length=16, seed=42, sol_only=True)[0]

    # CoT-only keeps every sentinel CoT token; neither mode emits a latent block.
    assert all((_COT_BASE + k) in cot["input_ids"] for k in range(64))
    assert _BO not in cot["input_ids"]
    assert all((_COT_BASE + k) not in sol["input_ids"] for k in range(64))
    # Dropping the CoT makes the Sol-Only sequence strictly shorter.
    assert len(sol["input_ids"]) < len(cot["input_ids"])
