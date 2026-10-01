"""Born-style squared-overlap readout + cross-entropy (master plan Part 4.3).

  a(o,n) = <W_E[o], psi[n]> = sum_k conj(W_E[o,k]) psi[n,k]      (complex)
  l(o|n) = |a(o,n)|^2 / T_o                                       (real >= 0)
  p(o|n) = softmax_o l(o|n)
  L_CE   = - sum_n ln p(o_{n+1} | n)

This is driver D1 (a bilinear contrastive readout over a learned embedding) with
one extra non-vanishing nonlinearity (the square) that lifts the hypothesis class
past linear-in-embeddings.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def born_logits(psi: Tensor, W_E: Tensor, T_o: Tensor) -> Tensor:
    """l(o|n) = |<W_E[o], psi[n]>|^2 / T_o.

    psi: (B, T, d) complex; W_E: (V, d) complex; T_o: scalar real > 0.
    Returns logits: (B, T, V) real.
    """
    # a[b,n,o] = sum_k conj(W_E[o,k]) psi[b,n,k]
    a = torch.einsum("btk,vk->btv", psi, W_E.conj())          # (B, T, V) complex
    mag2 = a.real**2 + a.imag**2                              # (B, T, V) real
    return mag2 / T_o.to(mag2.dtype)


def linear_logits(psi: Tensor, W_E: Tensor, T_o: Tensor) -> Tensor:
    """Linear (non-Born) readout l(o|n) = Re<W_E[o], psi[n]> / T_o.

    Used ONLY for the expressivity ablation (master plan Part 12): with the
    intensity nonlinearity disabled, this makes the whole model linear in the
    embeddings, which provably cannot represent composition tasks (e.g. modular
    addition). The real model uses born_logits.
    """
    a = torch.einsum("btk,vk->btv", psi, W_E.conj())          # (B, T, V) complex
    return a.real / T_o.to(a.real.dtype)


def ce_loss(logits: Tensor, targets: Tensor) -> Tensor:
    """Mean next-byte cross-entropy.

    logits:  (B, T, V) real  -- l(.|n) for n=0..T-1
    targets: (B, T) long byte ids
    Predicts target[n+1] from logits[n], i.e. positions 0..T-2.
    Returns scalar mean CE in nats per predicted byte.
    """
    B, T, V = logits.shape
    pred = logits[:, :-1, :].reshape(-1, V)        # (B*(T-1), V)
    tgt = targets[:, 1:].reshape(-1)               # (B*(T-1),)
    return F.cross_entropy(pred, tgt, reduction="mean")


def bits_per_byte(ce_nats: Tensor | float) -> float:
    """Convert mean CE (nats/byte) to bits/byte."""
    import math
    val = float(ce_nats)
    return val / math.log(2.0)


__all__ = ["born_logits", "ce_loss", "bits_per_byte"]
