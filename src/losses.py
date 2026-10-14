"""Loss assembly for the Stage-1 VQ-VAE (paper Eq. 2).

    L(X) = L_recon  +  || sg[X̄] - q(X̄) ||^2  +  beta * || X̄ - sg[q(X̄)] ||^2

L_recon is the token-level cross-entropy of the decoder logits vs. the original tokens. The two
VQ terms are provided by ``vector-quantize-pytorch`` and returned as ``out.vq_loss``:
  * with EMA (default, ``vq_ema=True``): the codebook term ``||sg[X̄]-q(X̄)||^2`` is realized by
    EMA updates inside the quantizer, and ``out.vq_loss`` is exactly ``beta*commitment``. We must
    NOT add a codebook MSE ourselves (it would double-count).
  * without EMA (``vq_ema=False``): ``out.vq_loss`` is the literal Eq. 2 VQ + commitment loss,
    computed in ``TokenVQVAE._eq2_vq_loss`` (weight-1 codebook term + ``beta``*commitment), NOT the
    library's merged term. The codebook is a learnable Parameter trained by that gradient.

Either way the total is simply ``recon_ce + out.vq_loss``.
"""
from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F

from .model import VQVAEOutput
from .utils import codebook_metrics, reconstruction_accuracy


def compute_loss_and_metrics(
    out: VQVAEOutput,
    target_ids: torch.Tensor,
    pad_id: int,
    codebook_size: int,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Return (total_loss, metrics). Reconstruction CE ignores pad positions."""
    B, T, V = out.logits.shape
    recon_ce = F.cross_entropy(
        out.logits.reshape(-1, V), target_ids.reshape(-1), ignore_index=pad_id
    )
    total = recon_ce + out.vq_loss

    with torch.no_grad():
        cb = codebook_metrics(out.indices, codebook_size, mask=out.chunk_valid)
        acc = reconstruction_accuracy(out.logits, target_ids, ignore_index=pad_id)

    metrics = {
        "loss": total.detach(),
        "recon_ce": recon_ce.detach(),
        "vq_loss": out.vq_loss.detach(),
        "xbar_rms": out.xbar_rms.detach(),
        "recon_acc": acc,
        "codebook_active_frac": cb["active_fraction"],
        "codebook_perplexity": cb["perplexity"],
    }
    return total, metrics
