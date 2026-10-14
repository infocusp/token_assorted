"""MetaMathQA loading, P/C/S splitting, tokenization, chunking, and loaders.

Adapts the HF-`datasets` + tokenizer pipeline pattern from facebookresearch/coconut
(`dataset.py`). The P/C/S split is **custom**: coconut's `### ` line markers do not match
MetaMathQA, whose `response` ends with the substring ``The answer is: <value>`` (not always
on its own line, e.g. ``...correctly.The answer is: 5``; GSM rows also carry ``#### <val>``).

Each example is assembled as  X = P (+) C (+) S  (paper notation, §3.1), where:
  * P = `query` (the question / prompt),
  * S = trailing answer span starting at the last ``The answer is:`` occurrence,
  * C = everything in `response` before S (the chain-of-thought).

The VQ-VAE is *trained* to reconstruct the full X (matches the paper), but `encode_dataset.py`
compresses only the C span — so we record the C token span for every example.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

# Fast tokenizers + forked DataLoader workers print a parallelism warning and can deadlock;
# disable tokenizer-internal threading since we parallelize at the DataLoader level.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

ANSWER_MARKER = "The answer is:"


# --------------------------------------------------------------------------------------
# P / C / S splitting
# --------------------------------------------------------------------------------------
def split_cot_solution(response: str) -> Tuple[str, str]:
    """Split a MetaMathQA `response` into (C = chain-of-thought, S = solution/answer span).

    S is the trailing span beginning at the **last** ``The answer is:`` substring (robust to
    it being glued to the previous token). If the marker is absent, the whole response is C
    and S is empty.
    """
    idx = response.rfind(ANSWER_MARKER)
    if idx == -1:
        return response, ""
    return response[:idx], response[idx:]


# --------------------------------------------------------------------------------------
# Tokenizer
# --------------------------------------------------------------------------------------
def load_tokenizer(name: str):
    """Load an HF tokenizer and guarantee a pad token.

    gpt2 (smoke) has no pad token and Llama-3.2 (full run) ships none either; we add a
    dedicated ``[PAD]`` so padding never collides with a real token. The model must size its
    embedding to ``len(tokenizer)`` (see model.build_from_config).
    """
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.add_special_tokens({"pad_token": "[PAD]"})
    return tok


@dataclass
class TokenInfo:
    vocab_size: int
    pad_id: int
    eos_id: int
    bos_id: int


def token_info(tokenizer) -> TokenInfo:
    eos = tokenizer.eos_token_id
    if eos is None:  # extremely rare; fall back to pad
        eos = tokenizer.pad_token_id
    bos = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else eos
    return TokenInfo(
        vocab_size=len(tokenizer),
        pad_id=tokenizer.pad_token_id,
        eos_id=eos,
        bos_id=bos,
    )


# --------------------------------------------------------------------------------------
# Example assembly (tokenize -> concat -> pad/truncate -> C span)
# --------------------------------------------------------------------------------------
def assemble_example(
    query: str,
    response: str,
    tokenizer,
    info: TokenInfo,
    max_seq: int,
) -> Dict[str, Any]:
    """Build one fixed-length training example.

    Pieces are tokenized separately (``add_special_tokens=False``) and concatenated so the C
    span is tracked exactly; this can differ slightly from tokenizing the joined string at BPE
    boundaries (documented trade-off). An EOS is appended so the decoder can learn termination.
    Returns fixed-length ``input_ids``/``attention_mask`` plus the C token span and prompt length.
    """
    c_text, s_text = split_cot_solution(response)
    p_ids = tokenizer(query, add_special_tokens=False).input_ids
    c_ids = tokenizer(c_text, add_special_tokens=False).input_ids
    s_ids = tokenizer(s_text, add_special_tokens=False).input_ids

    ids = p_ids + c_ids + s_ids + [info.eos_id]
    c_start = len(p_ids)
    c_end = len(p_ids) + len(c_ids)

    ids = ids[:max_seq]
    seq_len = len(ids)
    c_start = min(c_start, seq_len)
    c_end = min(c_end, seq_len)
    prompt_len = min(len(p_ids), seq_len)

    attn = [1] * seq_len + [0] * (max_seq - seq_len)
    ids = ids + [info.pad_id] * (max_seq - seq_len)

    return {
        "input_ids": torch.tensor(ids, dtype=torch.long),
        "attention_mask": torch.tensor(attn, dtype=torch.long),
        "prompt_len": int(prompt_len),
        "c_start": int(c_start),
        "c_end": int(c_end),
    }


# --------------------------------------------------------------------------------------
# Datasets
# --------------------------------------------------------------------------------------
def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_rows(
    dataset_name: str,
    split: str,
    data_fixture: Optional[str],
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Return a list of raw rows with at least `query` and `response` keys.

    Uses the offline JSONL fixture if `data_fixture` is set (smoke/tests), otherwise downloads
    `meta-math/MetaMathQA` via HF `datasets` (cached on first use).
    """
    if data_fixture is not None:
        rows = _read_jsonl(data_fixture)
    else:
        from datasets import load_dataset

        ds = load_dataset(dataset_name, split=split)
        rows = ds  # HF Dataset supports integer indexing + len
    if limit is not None:
        rows = [rows[i] for i in range(min(limit, len(rows)))]
    return rows


class MetaMathDataset(Dataset):
    """Tokenizes MetaMathQA rows on the fly into fixed-length VQ-VAE examples."""

    def __init__(self, rows, tokenizer, info: TokenInfo, max_seq: int):
        self.rows = rows
        self.tokenizer = tokenizer
        self.info = info
        self.max_seq = max_seq

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> Dict[str, Any]:
        row = self.rows[i]
        ex = assemble_example(
            row["query"], row["response"], self.tokenizer, self.info, self.max_seq
        )
        ex["id"] = i
        return ex


def _subset(rows, start: int, end: int):
    """Return rows[start:end] as row-records for both list and HF `Dataset` inputs.

    NOTE: slicing an HF `datasets.Dataset` (``ds[start:end]``) returns a *dict of columns*,
    not a subset of rows — so its ``len`` is the column count, which silently collapses the
    dataset (e.g. 395k rows -> 4). Use ``.select(range(...))`` for Datasets to keep row
    semantics (and the memory-mapped arrow backing); plain lists slice correctly.
    """
    if hasattr(rows, "select"):  # HF Dataset
        return rows.select(range(start, end))
    return rows[start:end]


def deterministic_train_val_split(rows, n_val: int):
    """Hold out the last `n_val` rows for validation (no HF val split exists)."""
    n = len(rows)
    n_val = min(n_val, max(0, n - 1))
    if n_val == 0:
        return rows, []
    return _subset(rows, 0, n - n_val), _subset(rows, n - n_val, n)


def build_loaders(cfg, tokenizer, info: TokenInfo, limit: Optional[int] = None):
    """Build train/val DataLoaders. `limit` truncates the dataset (quick runs)."""
    from .utils import seed_worker

    rows = load_rows(cfg.dataset_name, cfg.dataset_split, cfg.data_fixture, limit=limit)
    train_rows, val_rows = deterministic_train_val_split(rows, cfg.n_val)

    train_ds = MetaMathDataset(train_rows, tokenizer, info, cfg.max_seq)
    val_ds = MetaMathDataset(val_rows, tokenizer, info, cfg.max_seq)

    g = torch.Generator()
    g.manual_seed(cfg.seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        worker_init_fn=seed_worker,
        generator=g,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        drop_last=False,
    )
    return train_loader, val_loader
