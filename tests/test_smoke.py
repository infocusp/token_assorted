"""End-to-end smoke tests (acceptance criterion §12.2).

Builds the model, runs one forward+backward on a few examples, and asserts: finite loss, correct
shapes, the straight-through gradient reaches the encoder, and a short training loop decreases the
reconstruction loss. Covers both decoder modes and both transformer_impl block families.
"""
import math

import pytest
import torch

from src.losses import compute_loss_and_metrics

# Every wiring assertion below must hold for both block families. The searchformer arm is the
# validated recipe behind existing checkpoints; the vanilla arm is src/vanilla_blocks.py. Note the
# `attention.wq` gradient probes work unchanged for both because the vanilla blocks deliberately
# mirror the vendored submodule names.
IMPLS = ["searchformer", "vanilla"]


@pytest.mark.parametrize("impl", IMPLS)
@pytest.mark.parametrize("mode", ["ar", "parallel"])
def test_forward_finite_and_shapes(setup, mode, impl):
    cfg, info, model, batch, _ = setup(mode, transformer_impl=impl)
    out = model(batch["input_ids"], batch["attention_mask"], batch["prompt_len"])
    B, T = batch["input_ids"].shape
    assert out.logits.shape == (B, T, info.vocab_size)
    assert out.indices.shape == (B, cfg.n_codes)
    assert out.chunk_valid.shape == (B, cfg.n_codes)
    loss, metrics = compute_loss_and_metrics(out, batch["input_ids"], info.pad_id, cfg.codebook_size)
    assert math.isfinite(float(loss.detach()))
    assert math.isfinite(float(metrics["recon_ce"]))
    assert math.isfinite(float(metrics["vq_loss"]))


@pytest.mark.parametrize("impl", IMPLS)
@pytest.mark.parametrize("mode", ["ar", "parallel"])
def test_ste_gradient_reaches_encoder(setup, mode, impl):
    """The VQ straight-through estimator must propagate gradient back through f_enc."""
    cfg, info, model, batch, _ = setup(mode, transformer_impl=impl)
    model.train()
    out = model(batch["input_ids"], batch["attention_mask"], batch["prompt_len"])
    loss, _ = compute_loss_and_metrics(out, batch["input_ids"], info.pad_id, cfg.codebook_size)
    loss.backward()

    # first encoder block's query projection + the down-sampler must receive gradient
    enc_grad = model.enc_layers[0].attention.wq.weight.grad
    down_grad = model.down.weight.grad
    assert enc_grad is not None and enc_grad.norm().item() > 0, "no grad reached the encoder"
    assert down_grad is not None and down_grad.norm().item() > 0, "no grad reached the down-sampler"


@pytest.mark.parametrize("impl", IMPLS)
@pytest.mark.parametrize("mode", ["ar", "parallel"])
def test_short_training_decreases_recon(setup, mode, impl):
    """Overfit a few examples for a few dozen steps; reconstruction CE must drop."""
    cfg, info, model, batch, _ = setup(mode, transformer_impl=impl)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    recons = []
    for _ in range(60):
        opt.zero_grad(set_to_none=True)
        out = model(batch["input_ids"], batch["attention_mask"], batch["prompt_len"])
        loss, metrics = compute_loss_and_metrics(out, batch["input_ids"], info.pad_id, cfg.codebook_size)
        loss.backward()
        opt.step()
        recons.append(float(metrics["recon_ce"]))
    early = sum(recons[:5]) / 5
    late = sum(recons[-5:]) / 5
    assert late < early - 0.3, f"recon_ce did not decrease enough: early={early:.3f} late={late:.3f}"


@pytest.mark.parametrize("impl", IMPLS)
def test_codebook_not_collapsed_after_training(setup, impl):
    cfg, info, model, batch, _ = setup("ar", transformer_impl=impl)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    last = None
    for _ in range(60):
        opt.zero_grad(set_to_none=True)
        out = model(batch["input_ids"], batch["attention_mask"], batch["prompt_len"])
        loss, last = compute_loss_and_metrics(out, batch["input_ids"], info.pad_id, cfg.codebook_size)
        loss.backward()
        opt.step()
    assert last["codebook_active_frac"] > 0.1, "codebook appears collapsed"


@pytest.mark.parametrize("impl", IMPLS)
def test_conv_transpose_parallel_wiring(setup, impl):
    """Learned up-sampler (upsample_mode=conv_transpose): builds a ConvTranspose1d, produces the
    same (B, T, V) logits shape as repeat_interleave, and the straight-through gradient reaches it."""
    cfg, info, model, batch, _ = setup(
        "parallel", upsample_mode="conv_transpose", transformer_impl=impl
    )
    assert isinstance(model.up, torch.nn.ConvTranspose1d)
    model.train()
    out = model(batch["input_ids"], batch["attention_mask"], batch["prompt_len"])
    B, T = batch["input_ids"].shape
    assert out.logits.shape == (B, T, info.vocab_size)
    loss, _ = compute_loss_and_metrics(out, batch["input_ids"], info.pad_id, cfg.codebook_size)
    loss.backward()
    up_grad = model.up.weight.grad
    assert up_grad is not None and up_grad.norm().item() > 0, "no grad reached the learned up-sampler"


@pytest.mark.parametrize("impl", IMPLS)
def test_hybrid_decoder_wiring(setup, impl):
    """Hybrid decoder: one model holds BOTH decoder stacks. Each active_mode produces (B,T,V) logits
    and a finite loss, and its own decoder stack receives gradient while the other stays untouched."""
    cfg, info, model, batch, _ = setup(
        "hybrid", upsample_mode="conv_transpose", transformer_impl=impl
    )
    # both stacks + the learned up-sampler coexist; default active_mode is the deployment decoder
    assert hasattr(model, "ar_dec_layers") and hasattr(model, "par_dec_layers")
    assert isinstance(model.up, torch.nn.ConvTranspose1d)
    assert model.active_mode == "parallel"
    B, T = batch["input_ids"].shape

    for mode, active, idle in [
        ("ar", model.ar_dec_layers, model.par_dec_layers),
        ("parallel", model.par_dec_layers, model.ar_dec_layers),
    ]:
        model.zero_grad(set_to_none=True)
        model.active_mode = mode
        model.train()
        out = model(batch["input_ids"], batch["attention_mask"], batch["prompt_len"])
        assert out.logits.shape == (B, T, info.vocab_size)
        loss, metrics = compute_loss_and_metrics(out, batch["input_ids"], info.pad_id, cfg.codebook_size)
        assert math.isfinite(float(loss.detach())) and math.isfinite(float(metrics["recon_ce"]))
        loss.backward()
        active_grad = active[0].attention.wq.weight.grad
        idle_grad = idle[0].attention.wq.weight.grad
        assert active_grad is not None and active_grad.norm().item() > 0, f"{mode}: active decoder got no grad"
        assert idle_grad is None, f"{mode}: idle decoder unexpectedly received grad"


def test_vanilla_pos_embed_init_matches_tok_embed(setup):
    """Regression guard: the vanilla arm's `pos_embed` must be initialized at `tok_embed`'s std.

    `_add_pos` does `h = tok + pos` at BOTH the encoder and the decoder stack input, so
    `nn.Embedding`'s N(0, 1) default left position ~50x larger in norm than content — and
    `tie_head` blocks the obvious escape, since growing `tok_embed` to compensate also rescales
    every logit through the tied head. Run 660 (ema-parallel-llama-learnedup-vanilla) trained a
    full 100k steps with `pos_embed` coord std still 0.9947 and scored val recon_acc 0.343 against
    the searchformer arm's 0.475. Assert the scales AND the resulting signal ratio, not merely that
    the table exists.
    """
    _, _, model, batch, _ = setup("parallel", transformer_impl="vanilla")
    assert model.pos_embed is not None and model.rope is None

    tok_std = model.tok_embed.weight.std().item()
    pos_std = model.pos_embed.weight.std().item()
    assert pos_std == pytest.approx(tok_std, rel=0.25), (
        f"pos_embed std {pos_std:.4f} != tok_embed std {tok_std:.4f} — an un-initialized "
        "nn.Embedding (N(0,1)) drowns content at both stack inputs; see run 660."
    )

    # The invariant that actually bites: content must not be drowned by position at the stack
    # input. This is the quantity xbar_rms failed to reveal (the pre-Conv1d LayerNorm hides it).
    tok = model.tok_embed(batch["input_ids"])
    pos = model.pos_embed.weight[: tok.shape[1]]
    ratio = (tok.norm(dim=-1).mean() / pos.norm(dim=-1).mean()).item()
    assert 0.5 < ratio < 2.0, (
        f"content/position norm ratio at the encoder input is {ratio:.3f}; run 660 shipped 0.045"
    )


def test_searchformer_has_no_pos_embed(setup):
    """The searchformer arm must carry RoPE and NO learned position table: a `pos_embed` there
    would add 264 untrained rows to every existing checkpoint's state_dict and break strict loads."""
    _, _, model, _, _ = setup("parallel", transformer_impl="searchformer")
    assert model.pos_embed is None and model.rope is not None
