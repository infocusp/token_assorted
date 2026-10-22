"""encode/decode wiring: the C-span -> code-count relationship and a round-trip that runs."""
import argparse
import json
import math

import pytest

from scripts.make_fixture import make_rows
from src.data import assemble_example
from src.encode_dataset import build_meta, build_record, cot_code_span


def test_cot_code_span_aligned():
    # aligned C span -> exactly ceil(len / r) codes
    assert cot_code_span(0, 16, r=8, n_codes=8) == (0, 2)   # 16 tokens / 8 = 2
    assert cot_code_span(8, 24, r=8, n_codes=8) == (1, 3)   # 16 tokens -> 2 codes
    assert cot_code_span(0, 0, r=8, n_codes=8) == (0, 0)    # empty C -> 0 codes


def test_cot_code_span_approx_ceil():
    # unaligned spans give ceil(len/r) or +1 code (boundary chunk); always >= ceil
    for c_start, c_end, r in [(3, 20, 8), (5, 17, 4), (1, 31, 16)]:
        k0, k1 = cot_code_span(c_start, c_end, r, n_codes=1000)
        n_codes = k1 - k0
        ceil = math.ceil((c_end - c_start) / r)
        assert ceil <= n_codes <= ceil + 1


@pytest.mark.parametrize("impl", ["searchformer", "vanilla"])
@pytest.mark.parametrize("mode", ["ar", "parallel"])
def test_encode_decode_runs(setup, mode, impl):
    """Covers both decode() branches for both block families. The AR branch is the only place a
    sequence grows step by step, so it is what exercises the vanilla impl's position-table bound."""
    cfg, info, model, batch, tok = setup(mode, transformer_impl=impl)
    model.eval()
    indices, chunk_valid, quant = model.encode(batch["input_ids"], batch["attention_mask"])
    assert indices.shape == (batch["input_ids"].shape[0], cfg.n_codes)
    assert quant.shape == (batch["input_ids"].shape[0], cfg.n_codes, cfg.d_model)

    # decode the first example's codes back to token ids (untrained -> content not meaningful)
    codes = indices[:1, : max(1, int(chunk_valid[0].sum()))]
    out_ids = model.decode(codes, max_len=cfg.max_seq)
    assert out_ids.ndim == 2 and out_ids.shape[0] == 1
    text = tok.decode(out_ids[0].tolist(), skip_special_tokens=True)
    assert isinstance(text, str)


# ------------------------------------------------------------------------------------------
# build_record / build_meta: the Stage-2 hand-off schema
# ------------------------------------------------------------------------------------------
RECORD_KEYS = {"id", "prompt", "c_start", "num_cot_tokens", "cot_token_ids",
               "num_codes", "code_indices", "answer"}


def _build_records(setup, target="cot", n=8):
    """Records + the per-example pieces they were built from (rows match conftest's seed=0)."""
    cfg, info, model, batch, tok = setup("ar", n=n)
    model.eval()
    indices, chunk_valid, _ = model.encode(batch["input_ids"], batch["attention_mask"])
    rows = make_rows(n, seed=0)
    out = []
    for j, rw in enumerate(rows):
        ex = {
            "input_ids": batch["input_ids"][j],
            "attention_mask": batch["attention_mask"][j],
            "c_start": int(batch["c_start"][j]),
            "c_end": int(batch["c_end"][j]),
        }
        rec = build_record(j, rw["query"], rw["response"], ex, indices[j], chunk_valid[j],
                           target, cfg.compression_rate, cfg.n_codes)
        out.append((rec, ex, indices[j], chunk_valid[j]))
    return cfg, info, tok, out


def test_build_record_cot_schema(setup):
    _, _, _, built = _build_records(setup)
    for rec, _, _, _ in built:
        assert set(rec.keys()) == RECORD_KEYS
        assert json.loads(json.dumps(rec)) == rec  # plain python types only


def test_cot_ids_match_input_slice(setup):
    _, _, _, built = _build_records(setup)
    for rec, ex, _, _ in built:
        c_start, c_end = ex["c_start"], ex["c_end"]
        assert rec["c_start"] == c_start
        assert rec["cot_token_ids"] == ex["input_ids"][c_start:c_end].tolist()
        assert len(rec["cot_token_ids"]) == rec["num_cot_tokens"] == c_end - c_start


def test_code_span_alignment(setup):
    cfg, _, _, built = _build_records(setup)
    r = cfg.compression_rate
    for rec, ex, indices_row, valid_row in built:
        c_start, c_end = ex["c_start"], ex["c_end"]
        assert c_end > c_start  # every fixture row has a nonempty C
        k0, k1 = cot_code_span(c_start, c_end, r, cfg.n_codes)
        assert rec["c_start"] // r == k0
        assert rec["code_indices"] == indices_row[k0:k1][valid_row[k0:k1]].tolist()
        assert rec["num_codes"] == len(rec["code_indices"])
        assert all(0 <= c < cfg.codebook_size for c in rec["code_indices"])


def test_empty_cot_record(setup):
    cfg, info, model, _, tok = setup("ar", n=2)
    model.eval()
    cases = [
        {"query": "What is 2 + 2?", "response": "The answer is: 4"},  # empty C
        {"query": "word " * (2 * cfg.max_seq), "response": "We add.\nThe answer is: 4"},  # P fills max_seq
    ]
    for rw in cases:
        ex = assemble_example(rw["query"], rw["response"], tok, info, cfg.max_seq)
        indices, chunk_valid, _ = model.encode(ex["input_ids"][None], ex["attention_mask"][None])
        rec = build_record(0, rw["query"], rw["response"], ex, indices[0], chunk_valid[0],
                           "cot", cfg.compression_rate, cfg.n_codes)
        assert ex["c_start"] == ex["c_end"]
        assert rec["cot_token_ids"] == [] and rec["num_cot_tokens"] == 0
        assert rec["code_indices"] == [] and rec["num_codes"] == 0


def test_full_mode_semantics(setup):
    _, _, _, built = _build_records(setup, target="full")
    for rec, ex, indices_row, valid_row in built:
        # full mode: num_cot_tokens counts the whole real span, not the C slice ...
        assert rec["num_cot_tokens"] == int(ex["attention_mask"].sum())
        assert rec["code_indices"] == indices_row[valid_row].tolist()
        # ... while cot_token_ids stays the C slice
        assert rec["cot_token_ids"] == ex["input_ids"][ex["c_start"]:ex["c_end"]].tolist()


def test_build_meta(setup):
    cfg, info, _, _, tok = setup("ar", n=2)
    args = argparse.Namespace(checkpoint="runs/x/best.pt", split="train",
                              data_fixture=None, limit=None)
    meta = build_meta(args, cfg, info, tok, "cot", "meta-math/MetaMathQA", n_written=2)
    assert {"schema_version", "created_at", "checkpoint", "tokenizer_name", "dataset_name",
            "split", "compress_target", "max_seq", "compression_rate", "chunk_length",
            "n_codes", "codebook_size", "pad_id", "eos_id", "n_records"} <= set(meta.keys())
    assert meta["tokenizer_name"] == tok.name_or_path
    assert meta["compression_rate"] == cfg.compression_rate
    assert meta["max_seq"] == cfg.max_seq
    assert meta["codebook_size"] == cfg.codebook_size
    assert meta["n_records"] == 2
    json.dumps(meta)  # serializable
