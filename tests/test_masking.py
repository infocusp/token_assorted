"""Padding-mask correctness.

With right-padding + a causal encoder, real-token outputs must not depend on what sits in the
pad region: pad-position encoder outputs are zeroed before down-sampling, pad chunks are masked
out of the quantizer, and CE ignores pad positions. So changing the pad tokens must leave the
codes for valid chunks and the logits at real positions unchanged.

Run against both `transformer_impl` block families: since neither passes an attention mask on the
self-attention paths, these tests are what certify that the vanilla encoder is genuinely causal. A
bidirectional stack would let a real position attend a corrupted pad position and fail here.
"""
import pytest
import torch

IMPLS = ["searchformer", "vanilla"]


def _corrupt_pad_region(batch, info):
    """Return a copy of batch with pad-region token ids replaced by a different valid token."""
    ids = batch["input_ids"].clone()
    pad = batch["attention_mask"] == 0
    # replace pad slots with some non-pad token id (0); attention_mask still marks them pad
    ids[pad] = (info.pad_id + 1) % info.vocab_size
    new = dict(batch)
    new["input_ids"] = ids
    return new


@pytest.mark.parametrize("impl", IMPLS)
def test_xbar_invariant_to_pad(setup, impl):
    """Pre-quantization latents on valid chunks must be EXACTLY invariant to pad content.

    This is the sensitive version of the two tests below, and the one that actually detects an
    attention leak. Measured: a causal encoder gives max|dX̄| = 0.0, while a deliberately
    non-causal one gives ~1.3e-2 — a real leak that nonetheless flips no code index (the codebook
    is coarse enough to absorb it) and so is invisible to a codes-equality or logits assertion.
    Asserting on X̄ closes that gap: it is the last point before the quantizer discretizes the leak
    away.
    """
    cfg, info, model, batch, _ = setup("ar", transformer_impl=impl)
    model.eval()
    corrupt = _corrupt_pad_region(batch, info)
    with torch.no_grad():
        _, xbar_a, valid = model._encode_to_latents(batch["input_ids"], batch["attention_mask"])
        _, xbar_b, _ = model._encode_to_latents(corrupt["input_ids"], corrupt["attention_mask"])
    assert valid.any(), "fixture produced no valid chunks — test would be vacuous"
    delta = (xbar_a - xbar_b)[valid].abs().max().item()
    assert delta == 0.0, (
        f"pre-quantization latents leaked pad information: max|dX̄|={delta:.3e} on valid chunks. "
        "Some attention path is reading pad positions."
    )


@pytest.mark.parametrize("impl", IMPLS)
def test_valid_codes_invariant_to_pad(setup, impl):
    cfg, info, model, batch, _ = setup("ar", transformer_impl=impl)
    model.eval()
    idx_a, valid_a, _ = model.encode(batch["input_ids"], batch["attention_mask"])
    corrupt = _corrupt_pad_region(batch, info)
    idx_b, valid_b, _ = model.encode(corrupt["input_ids"], corrupt["attention_mask"])

    assert torch.equal(valid_a, valid_b)
    # indices on valid chunks must be identical despite different pad tokens
    assert torch.equal(idx_a[valid_a], idx_b[valid_b])


@pytest.mark.parametrize("impl", IMPLS)
def test_real_position_logits_invariant_to_pad(setup, impl):
    cfg, info, model, batch, _ = setup("ar", transformer_impl=impl)
    model.eval()
    with torch.no_grad():
        out_a = model(batch["input_ids"], batch["attention_mask"], batch["prompt_len"])
        corrupt = _corrupt_pad_region(batch, info)
        out_b = model(corrupt["input_ids"], corrupt["attention_mask"], corrupt["prompt_len"])
    real = batch["attention_mask"].bool()
    la = out_a.logits[real]
    lb = out_b.logits[real]
    assert torch.allclose(la, lb, atol=1e-4), "real-position logits leaked information from pad tokens"


@pytest.mark.parametrize("impl", IMPLS)
def test_conv_transpose_logits_invariant_to_pad(setup, impl):
    """The learned ConvTranspose1d up-sampler (kernel=stride=r, no overlap) must preserve the
    pad-invariance of real-position logits: each output position depends on exactly one chunk, and
    under right-padding a real position's chunk (and everything the causal decoder attends to) is
    real, so corrupting the pad region cannot change real-position logits."""
    cfg, info, model, batch, _ = setup(
        "parallel", upsample_mode="conv_transpose", transformer_impl=impl
    )
    model.eval()
    with torch.no_grad():
        out_a = model(batch["input_ids"], batch["attention_mask"], batch["prompt_len"])
        corrupt = _corrupt_pad_region(batch, info)
        out_b = model(corrupt["input_ids"], corrupt["attention_mask"], corrupt["prompt_len"])
    real = batch["attention_mask"].bool()
    assert torch.allclose(out_a.logits[real], out_b.logits[real], atol=1e-4), (
        "conv_transpose real-position logits leaked information from pad tokens"
    )
