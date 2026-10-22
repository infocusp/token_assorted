"""Stage-2 dataset: build assorted latent+text training sequences from the Stage-1 artifact.

This is the read side of the Stage-1 -> Stage-2 hand-off. It consumes the JSONL produced by
``src/encode_dataset.py`` (records with ``prompt``, ``cot_token_ids``, ``code_indices``,
``answer``, ...) and, per the paper (§3.2 Eq.1 + §4.2), constructs a *modified* input

    X~ = [BOS] + P + [<boLatent> z_1..z_{m/r} <eoLatent>] + c_{m+1..} + S + [EOS]

where the first ``m`` CoT text tokens are replaced by their ``m/r`` latent codes and the rest
stay as text. ``m`` is drawn per example by the paper's randomized-replacement scheme.

Design choices (paper is underspecified — see config/stage2_llama1b.yaml comments):
  * **Vocabulary extension**: add ``<boLatent>``, ``<eoLatent>`` and ``<Latent-code-0..N-1>`` as
    ``additional_special_tokens`` (never split; skipped by ``add_special_tokens=False``).
  * **m -> codes (simple sequential)**: ``num_latent = m // L``; latent = ``code_indices[:num_latent]``;
    remaining text = ``cot_token_ids[m:]``. The codes are fresh symbols the LLM learns end-to-end,
    so exact token<->chunk alignment is unnecessary; this matches the paper's "32 CoT tokens -> 2
    codes" mental model. (When ``c_start % L != 0`` the leading code covers up to ``L-1`` prompt
    tokens rather than exactly ``m`` CoT tokens — an accepted, documented boundary imprecision.)
  * **Reuse ``cot_token_ids`` verbatim** for the remaining CoT text (same tokenizer as Stage 1 ->
    no re-tokenization drift). Only ``prompt`` and ``answer`` strings are (re-)tokenized here.
  * **Prompt masking**: labels are ``-100`` over ``[BOS] + P`` and real ids over the latent block +
    remaining CoT + answer + EOS. So the loss trains the model to *emit* the latent tokens and the
    two delimiters, plus the remaining reasoning and the solution.
  * **Sol-Only mode** (``sol_only=True``, config field ``sol_only``): reproduces the paper's Sol-Only
    baseline (Table 4.2). Both the latent block and *all* CoT text are dropped, so each example is
    ``[BOS] + P + S + [EOS]`` (P masked, S + EOS supervised). ``m_max_set``/``chunk_length`` are
    ignored in this mode. Default ``sol_only=False`` leaves the assorted behavior above unchanged.

``__getitem__`` returns python lists (not tensors) for ``input_ids``/``attention_mask``/``labels``
so llama-cookbook's ``ConcatDataset`` (sequence packing) can concatenate them.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from torch.utils.data import Dataset

from .data import _read_jsonl, deterministic_train_val_split


# --------------------------------------------------------------------------------------
# Tokenizer extension
# --------------------------------------------------------------------------------------
@dataclass
class LatentVocab:
    """Cached token ids for the extended vocabulary."""
    bo_id: int
    eo_id: int
    bos_id: int
    eos_id: int
    code_id: List[int]           # code_id[c] = tokenizer id of "<Latent-code-c>"
    n_added: int
    base_vocab_size: int         # len(tokenizer) BEFORE the additions (= first new id)


def build_extended_tokenizer(
    base_model: str,
    codebook_size: int,
    bo_token: str,
    eo_token: str,
    code_token_fmt: str,
):
    """Load the base tokenizer and add the 2 delimiters + ``codebook_size`` latent-code tokens.

    Returns ``(tokenizer, LatentVocab)``. The caller must ``model.resize_token_embeddings(len(tok))``.
    """
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(base_model)
    base_vocab_size = len(tok)

    # The base unsloth/Llama-3.2-1B ships a pad token; guard anyway (mirrors src/data.py).
    if tok.pad_token is None:
        tok.add_special_tokens({"pad_token": "[PAD]"})
        base_vocab_size = len(tok)

    new_tokens = [bo_token, eo_token] + [code_token_fmt.format(i) for i in range(codebook_size)]
    n_added = tok.add_special_tokens({"additional_special_tokens": new_tokens})

    # Each new token must map to exactly one, brand-new id (>= base_vocab_size).
    def _one_id(t: str) -> int:
        ids = tok.encode(t, add_special_tokens=False)
        assert len(ids) == 1, f"special token {t!r} did not map to a single id: {ids}"
        return ids[0]

    bo_id = _one_id(bo_token)
    eo_id = _one_id(eo_token)
    code_id = [_one_id(code_token_fmt.format(i)) for i in range(codebook_size)]
    assert min([bo_id, eo_id] + code_id) >= base_vocab_size, "new tokens collided with base vocab"

    eos_id = tok.eos_token_id
    bos_id = tok.bos_token_id if tok.bos_token_id is not None else eos_id
    vocab = LatentVocab(
        bo_id=bo_id, eo_id=eo_id, bos_id=bos_id, eos_id=eos_id,
        code_id=code_id, n_added=n_added, base_vocab_size=base_vocab_size,
    )
    return tok, vocab


# --------------------------------------------------------------------------------------
# Randomized replacement dataset
# --------------------------------------------------------------------------------------
class Stage2Dataset(Dataset):
    """Assorted latent+text sequences with per-example randomized latent replacement."""

    def __init__(
        self,
        records: List[Dict[str, Any]],
        tokenizer,
        vocab: LatentVocab,
        m_max_set: List[int],
        chunk_length: int,
        seed: int,
        sol_only: bool = False,
    ):
        self.records = records
        self.tok = tokenizer
        self.v = vocab
        self.m_max_set = list(m_max_set)
        self.L = int(chunk_length)
        self.seed = int(seed)
        self.sol_only = bool(sol_only)   # Sol-Only baseline: drop CoT + latent codes, keep P -> S only

    def __len__(self) -> int:
        return len(self.records)

    def _sample_m(self, rng: random.Random, num_codes: int, num_cot: int) -> int:
        """Paper §4.2: sample m_max in M, then m in {0, L, ..., m_max}, clamped by data."""
        m_max = rng.choice(self.m_max_set)
        # multiples of L, capped by m_max, #available codes, and #available CoT tokens.
        cap_chunks = min(m_max // self.L, num_codes, num_cot // self.L)
        if cap_chunks <= 0:
            return 0
        return self.L * rng.randint(0, cap_chunks)

    def __getitem__(self, i: int) -> Dict[str, List[int]]:
        rec = self.records[i]
        cot_ids: List[int] = rec["cot_token_ids"]
        codes: List[int] = rec["code_indices"]
        # per-example reproducible RNG (1-epoch run -> a fixed per-example draw is sufficient)
        rng = random.Random(self.seed * 1_000_003 + i)
        if self.sol_only:
            num_latent = 0
            remaining: List[int] = []                     # drop all CoT text + latent codes -> P -> S only
        else:
            m = self._sample_m(rng, len(codes), len(cot_ids))
            num_latent = m // self.L
            remaining = cot_ids[m:]                        # simple sequential rule

        prompt_ids = self.tok(rec["prompt"], add_special_tokens=False).input_ids
        answer_ids = self.tok(rec["answer"], add_special_tokens=False).input_ids

        ids: List[int] = [self.v.bos_id]
        labels: List[int] = [-100]                       # BOS masked
        ids += prompt_ids
        labels += [-100] * len(prompt_ids)               # prompt masked

        if num_latent > 0:
            block = [self.v.bo_id] + [self.v.code_id[c] for c in codes[:num_latent]] + [self.v.eo_id]
            ids += block
            labels += block                              # model must learn to EMIT the codes

        ids += remaining
        labels += remaining

        ids += answer_ids
        labels += answer_ids

        ids.append(self.v.eos_id)
        labels.append(self.v.eos_id)

        return {"input_ids": ids, "attention_mask": [1] * len(ids), "labels": labels}


def load_stage2_records(artifact: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """Read the Stage-1 artifact JSONL into memory (list of dict records)."""
    rows = _read_jsonl(artifact)
    if limit is not None:
        rows = rows[:limit]
    return rows


def build_stage2_datasets(
    artifact: str,
    tokenizer,
    vocab: LatentVocab,
    m_max_set: List[int],
    chunk_length: int,
    seed: int,
    n_val: int,
    limit: Optional[int] = None,
    sol_only: bool = False,
):
    """Return ``(train_ds, val_ds)`` with a deterministic tail-holdout split (Stage-1 convention)."""
    rows = load_stage2_records(artifact, limit=limit)
    train_rows, val_rows = deterministic_train_val_split(rows, n_val)
    def mk(rs):
        return Stage2Dataset(list(rs), tokenizer, vocab, m_max_set, chunk_length, seed,
                             sol_only=sol_only)

    return mk(train_rows), mk(val_rows)


def decode_example(item: Dict[str, List[int]], tokenizer) -> str:
    """Debug helper: render an assembled example back to text (keeps special tokens)."""
    return tokenizer.decode(item["input_ids"], skip_special_tokens=False)
