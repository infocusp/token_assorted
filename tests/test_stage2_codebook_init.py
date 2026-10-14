"""Stage-2 ``new_token_init: codebook`` — the Stage-1 codebook -> LLM-width bridge.

The whole point of this init is that the width mismatch (Stage-1 d=512 -> Llama 2048/3072) is
crossed by a SEMI-ORTHOGONAL projection, which preserves inner products exactly. These tests lock
that property, since a silently non-isometric W would still "work" (distinct rows, no crash) while
destroying the codebook geometry the init exists to transfer.

Pure CPU/tensor tests: the codebook loader and the chunk-mean target pass are monkeypatched, so
nothing here needs HF weights, the artifact, or the Stage-1 checkpoint.
"""
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from src import stage2_finetune as s2

N_CODES, D_STAGE1, D_LLM, BASE_VOCAB = 32, 8, 64, 100


def _vocab():
    # Real layout is [bo, eo, code_0..code_n]; only the row indices matter here.
    return SimpleNamespace(
        base_vocab_size=BASE_VOCAB,
        code_id=[BASE_VOCAB + 2 + c for c in range(N_CODES)],
    )


def _emb():
    """Base rows + the global-mean init that build_model_and_tokenizer applies before dispatch."""
    torch.manual_seed(0)
    e = torch.randn(BASE_VOCAB + 2 + N_CODES, D_LLM) * 0.02
    e[BASE_VOCAB:] = e[:BASE_VOCAB].mean(dim=0, keepdim=True)
    return e


def _cfg(rotation="procrustes", scale=1.0):
    return {"codebook_size": N_CODES, "codebook_init_rotation": rotation,
            "new_token_init_scale": scale, "seed": 0, "artifact": "unused.jsonl"}


def _patch(monkeypatch, codebook, targets, seen):
    monkeypatch.setattr(s2, "_load_stage1_codebook", lambda cfg, n: (codebook, "fake-ckpt.pt"))
    monkeypatch.setattr(s2, "_chunk_mean_targets", lambda base, vocab, cfg: (targets, seen, 123))


def _targets_from(codebook, rank=None, noise=0.1):
    """Chunk-mean-like targets: a random rotation of the codebook plus noise, optionally forced
    to low rank to emulate the degenerate real ones (measured pair-cos 0.60)."""
    torch.manual_seed(1)
    T = codebook @ torch.randn(codebook.shape[1], D_LLM) + noise * torch.randn(N_CODES, D_LLM)
    if rank is not None:
        T = torch.randn(N_CODES, rank) @ torch.randn(rank, D_LLM)
    return T


@pytest.mark.parametrize("rotation", ["procrustes", "random"])
def test_projection_is_isometric(monkeypatch, rotation):
    """Pairwise cosines of the written rows must equal the codebook's, exactly."""
    torch.manual_seed(2)
    cb = torch.randn(N_CODES, D_STAGE1) * torch.rand(N_CODES, 1) * 50  # wild EMA-like norms
    seen = torch.ones(N_CODES, dtype=torch.bool)
    _patch(monkeypatch, cb, _targets_from(cb), seen)

    emb, vocab = _emb(), _vocab()
    with torch.no_grad():
        s2._codebook_init(emb, vocab, _cfg(rotation))
    rows = emb[torch.tensor(vocab.code_id)]

    g_cb = F.normalize(cb, dim=-1) @ F.normalize(cb, dim=-1).T
    g_rows = F.normalize(rows, dim=-1) @ F.normalize(rows, dim=-1).T
    assert torch.allclose(g_cb, g_rows, atol=1e-5), "projection distorted the codebook geometry"

    # Direction-only: the codebook's 50x norm spread must NOT leak into the embedding rows.
    target = emb[:BASE_VOCAB].norm(dim=-1).mean()
    assert torch.allclose(rows.norm(dim=-1), target.expand(N_CODES), atol=1e-4)


def test_isometry_survives_rank_deficient_targets(monkeypatch):
    """Degenerate chunk-mean targets (the real ones are only cos 0.60 apart) may cost alignment
    but can never cost the isometry — svd() returns orthonormal factors at any rank."""
    torch.manual_seed(3)
    cb = torch.randn(N_CODES, D_STAGE1)
    seen = torch.ones(N_CODES, dtype=torch.bool)
    _patch(monkeypatch, cb, _targets_from(cb, rank=1), seen)

    emb, vocab = _emb(), _vocab()
    with torch.no_grad():
        s2._codebook_init(emb, vocab, _cfg("procrustes"))
    rows = emb[torch.tensor(vocab.code_id)]

    g_cb = F.normalize(cb, dim=-1) @ F.normalize(cb, dim=-1).T
    g_rows = F.normalize(rows, dim=-1) @ F.normalize(rows, dim=-1).T
    assert torch.allclose(g_cb, g_rows, atol=1e-5)


def test_procrustes_aligns_to_targets_and_random_does_not(monkeypatch):
    """The only thing the rotation choice buys is placement near the chunk-mean targets — the
    difference the two experiment arms are meant to measure."""
    torch.manual_seed(4)
    cb = torch.randn(N_CODES, D_STAGE1)
    seen = torch.ones(N_CODES, dtype=torch.bool)
    T = _targets_from(cb, noise=0.05)
    mu = _emb()[:BASE_VOCAB].mean(dim=0)
    Tn = F.normalize(T - mu, dim=-1)

    align = {}
    for rotation in ("procrustes", "random"):
        _patch(monkeypatch, cb, T, seen)
        emb, vocab = _emb(), _vocab()
        with torch.no_grad():
            s2._codebook_init(emb, vocab, _cfg(rotation))
        rows = F.normalize(emb[torch.tensor(vocab.code_id)], dim=-1)
        align[rotation] = float((rows * Tn).sum(-1).mean())

    assert align["procrustes"] > 0.5, f"procrustes failed to align: {align}"
    assert align["procrustes"] > align["random"] + 0.4, f"rotation arms not separated: {align}"


def test_unseen_codes_still_get_a_row(monkeypatch):
    """chunk_mean leaves codes absent from train at the global mean (81/1024 in the real
    artifact); codebook init has a vector for every code and must use it."""
    torch.manual_seed(5)
    cb = torch.randn(N_CODES, D_STAGE1)
    seen = torch.ones(N_CODES, dtype=torch.bool)
    seen[-5:] = False                                   # last 5 codes never appear in train
    T = _targets_from(cb)
    T[~seen] = 0.0
    _patch(monkeypatch, cb, T, seen)

    emb, vocab = _emb(), _vocab()
    mean_row = emb[BASE_VOCAB].clone()                  # the pre-dispatch global-mean init
    with torch.no_grad():
        s2._codebook_init(emb, vocab, _cfg("procrustes"))
    rows = emb[torch.tensor(vocab.code_id)]

    assert not torch.allclose(rows[-5:], mean_row.expand(5, D_LLM), atol=1e-6)
    assert torch.unique(rows, dim=0).shape[0] == N_CODES, "code rows are not all distinct"
    # Delimiters (bo/eo) are NOT touched — they keep the global-mean init.
    assert torch.allclose(emb[BASE_VOCAB], mean_row) and torch.allclose(emb[BASE_VOCAB + 1], mean_row)


def test_bad_rotation_rejected(monkeypatch):
    torch.manual_seed(6)
    cb = torch.randn(N_CODES, D_STAGE1)
    _patch(monkeypatch, cb, _targets_from(cb), torch.ones(N_CODES, dtype=torch.bool))
    with pytest.raises(ValueError, match="codebook_init_rotation"):
        with torch.no_grad():
            s2._codebook_init(_emb(), _vocab(), _cfg("orthogonal-ish"))


def test_wider_codebook_than_model_rejected(monkeypatch):
    """No isometry exists downward — fail loudly instead of silently discarding directions."""
    torch.manual_seed(7)
    cb = torch.randn(N_CODES, D_LLM + 8)
    _patch(monkeypatch, cb, _targets_from(cb), torch.ones(N_CODES, dtype=torch.bool))
    with pytest.raises(ValueError, match="exceeds the LLM width"):
        with torch.no_grad():
            s2._codebook_init(_emb(), _vocab(), _cfg("procrustes"))
