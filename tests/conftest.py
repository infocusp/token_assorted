"""Shared pytest fixtures: a non-gated gpt2 tokenizer and a tiny model+batch builder.

These tests need network access once to fetch the gpt2 tokenizer (cached thereafter).
"""
import sys
from pathlib import Path

import pytest
import torch
from torch.utils.data import default_collate

# Ensure the repo root is importable (`src`, `third_party`) regardless of pytest's cwd.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.make_fixture import make_rows  # noqa: E402
from src.data import assemble_example, load_tokenizer, token_info  # noqa: E402
from src.model import TokenVQVAE  # noqa: E402
from src.utils import SMOKE_OVERRIDES, Config, apply_overrides  # noqa: E402


@pytest.fixture(scope="session")
def tokenizer():
    return load_tokenizer("gpt2")


@pytest.fixture
def setup(tokenizer):
    """Return a builder: decoder_mode -> (cfg, info, model, batch, tokenizer)."""

    def _make(
        decoder_mode: str = "ar",
        n: int = 8,
        seed: int = 0,
        upsample_mode: str = "repeat",
        transformer_impl: str = "searchformer",
    ):
        cfg = apply_overrides(
            Config(),
            {
                **SMOKE_OVERRIDES,
                "decoder_mode": decoder_mode,
                "upsample_mode": upsample_mode,
                "transformer_impl": transformer_impl,
            },
        )
        info = token_info(tokenizer)
        rows = make_rows(n, seed=seed)
        exs = [
            assemble_example(r["query"], r["response"], tokenizer, info, cfg.max_seq)
            for r in rows
        ]
        batch = default_collate(exs)
        torch.manual_seed(seed)
        model = TokenVQVAE.build_from_config(cfg, info)
        return cfg, info, model, batch, tokenizer

    return _make
