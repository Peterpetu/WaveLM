"""The WaveLM wave step (master plan Part 4.2 / spec/wave_lm/02_dynamics.md).

One internal computation step tau -> tau+1 on a complex field psi of shape
(B, T, d). As built (see spec/wave_lm/04_as_built.md), the step is FIVE ops:

  (1) dilated depthwise causal convolution (info across POSITIONS):
        p[n,k] = beta * psi[n,k] + sum_{j=1}^{Delta} K[j,k] * psi[n - j*D_tau, k]
  (2) [optional] mode coupling (info across CHANNELS): dense p <- M p, or low-rank
        residual p <- p + U(Vᴴ p).
  (3) per-channel saturating nonlinearity (load-bearing; changes direction):
        q[n,k] = p[n,k] / (1 + g * |p[n,k]|^2)
  (4) per-channel phase rotation (distinct carriers -> interference):
        r[n,k] = exp(i*omega_k) * q[n,k]
  (5) energy normalisation (fixed per-position energy ||psi[n]|| = sqrt(d)):
        psi'[n,k] = sqrt(d) * r[n,k] / ||r[n]||

All operations are pure functions of (psi, constrained-params, tau); the model
(wave/model.py) owns the parameters and supplies their constrained values.

Stability: op (5) fixes ||psi'[n]|| = sqrt(d) exactly for every tau, n -- this
(empirically forced, master plan Part 5 revised) cures the depth vanishing/blow-up
gradient and makes the Born readout a true normalised-state probability. The
historical pre-normalisation bound ||q[n]|| <= sqrt(d)/(2 sqrt(g)) from op (3)
still holds but is dominated by op (5).
"""

from __future__ import annotations

import math

import torch
from torch import Tensor


def causal_dilated_conv(psi: Tensor, K: Tensor, beta: Tensor, dilation: int,
                        self_gate: Tensor | None = None) -> Tensor:
    """p[n,k] = beta*g[n,k]*psi[n,k] + sum_{j=1}^{Delta} K[j-1,k] * psi[n - j*dilation, k].

    psi:       (B, T, d) complex
    K:         (Delta, d) complex   (tap j-1 carries offset j*dilation)
    beta:      scalar real tensor (self-retention)
    dilation:  positive int D_tau
    self_gate: optional (B, T, d) REAL tensor g[n,k] in (0,1) multiplying the
               self-retention term ONLY (data-dependent selectivity, doc 12). None
               (default) => g==1 => bit-identical to the historical conv.

    Returns p: (B, T, d) complex. Causal: position n only reads n' < n
    (zero-padded at the start), so the model stays autoregressive.
    """
    import torch.nn.functional as F
    B, T, d = psi.shape
    Delta = K.shape[0]
    self_term = beta.to(psi.dtype) * psi
    if self_gate is not None:
        self_term = self_gate.to(psi.dtype) * self_term
    p = self_term
    for j in range(1, Delta + 1):
        s = j * dilation
        if s >= T:
            break
        # contribution of position n-s to position n, zero-padded at the front
        # (causal). Pad on the time axis only: (d_lo,d_hi, t_lo,t_hi).
        contrib = K[j - 1].view(1, 1, d) * psi[:, : T - s, :]    # (B, T-s, d)
        p = p + F.pad(contrib, (0, 0, s, 0))
    return p


def saturating_nonlinearity(p: Tensor, g: Tensor) -> Tensor:
    """Per-channel saturation q[n,k] = p[n,k] / (1 + g * |p[n,k]|^2).

    p: (B, T, d) complex; g: scalar real tensor > 0.

    Per-CHANNEL (not mean) so it compresses high-magnitude channels more than
    low-magnitude ones, i.e. it changes the *direction* of psi[n] -- this is the
    load-bearing nonlinearity that survives the unit-norm projection in wave_step
    (a pure scalar rescale would be cancelled by normalisation). Smooth, bounded
    (|q| <= 1/(2 sqrt(g))), derivative finite everywhere (1+g|p|^2 >= 1).
    """
    u = p.real**2 + p.imag**2                                  # (B, T, d) real
    denom = (1.0 + g.to(u.dtype) * u)                          # (B, T, d) real
    return p / denom.to(p.dtype)


def channel_mix(p: Tensor, M: Tensor) -> Tensor:
    """Dense mode coupling p[n] <- M @ p[n] (per position), M complex (d, d).

    The conv mixes information across POSITIONS within a channel; this mixes
    across CHANNELS (modes) at a position -- the waveguide mode-coupling analogue.
    Without it, channels evolve as independent 1-D filters that interact only at
    the Born readout; with it, modes exchange information during evolution, which
    is needed for rich (many-feature) composition. O(B*T*d^2) per application.
    """
    return torch.einsum("btk,kj->btj", p, M)


def channel_mix_lowrank(p: Tensor, U: Tensor, V: Tensor) -> Tensor:
    """Low-rank residual mode coupling p <- p + U (Vᴴ p), U,V complex (d, r).

    Equivalent to p @ (I + V* Uᵀ)... implemented as the cheap residual
    A·p with A = U Vᴴ (rank r). Cost O(B*T*d*r) << dense O(B*T*d^2). Lets us add a
    lot of (untied, per-mixing-point) channel-mixing capacity within budget.
    Init U,V ~ 0 so the block starts as identity and learns the coupling.
    """
    z = torch.einsum("btk,kr->btr", p, V.conj())     # Vᴴ p   -> (B,T,r)
    Ap = torch.einsum("btr,kr->btk", z, U)           # U z    -> (B,T,d)
    return p + Ap


def selective_gate(psi: Tensor, W_in: Tensor, W_out: Tensor, bias: Tensor) -> Tensor:
    """Data-dependent per-channel retention gate (the "hidden lever" probe, doc 12).

        u[n,k] = |psi[n,k]|^2                                  (B,T,d) real
        s[n]   = sigmoid( (u[n] @ W_in) @ W_out + bias )       (B,T,d) real, in (0,1)

    W_in (d, r_sel), W_out (r_sel, d), bias (d,) -- all REAL. Low-rank (r_sel) so the
    gate costs O(B*T*d*r_sel). Makes the conv self-retention input-dependent (the
    selectivity that an otherwise linear-time-invariant WaveLM lacks). Recomputed each
    internal step from the current field, so the gate adapts as the wave evolves.
    """
    u = psi.real**2 + psi.imag**2                    # (B,T,d) real intensity features
    h = torch.einsum("btk,kr->btr", u, W_in)         # (B,T,r_sel)
    logits = torch.einsum("btr,rk->btk", h, W_out) + bias
    return torch.sigmoid(logits)                     # (B,T,d) in (0,1)


def phase_rotation(q: Tensor, omega: Tensor) -> Tensor:
    """psi'[n,k] = exp(i*omega_k) * q[n,k].

    q: (B, T, d) complex; omega: (d,) real. Magnitude-preserving.
    """
    d = q.shape[-1]
    rot = torch.exp(1j * omega.to(torch.float64 if q.dtype == torch.complex128 else torch.float32))
    rot = rot.to(q.dtype).view(1, 1, d)
    return rot * q


def normalize_field(psi: Tensor, eps: float = 1e-8) -> Tensor:
    """Project each position to fixed energy: ||psi[n]|| = sqrt(d) (per-channel RMS=1).

    Keeps the field energy constant across the n_sub internal steps -> no decay,
    no vanishing gradient at depth (master plan Part 5, revised). The sqrt(d)
    target (rather than unit norm) keeps per-channel magnitudes O(1), so the Born
    readout |<W_E[o],psi>|^2 produces O(1)-scale logits at init -> a non-vanishing
    learning signal from step 1 (softmax handles probability normalisation, so the
    overall scale is free; sqrt(d) is the well-conditioned choice).
    """
    d = psi.shape[-1]
    norm = (psi.real**2 + psi.imag**2).sum(dim=-1, keepdim=True).sqrt()   # (B,T,1)
    target = math.sqrt(d)
    return psi * target / (norm + eps).to(psi.dtype)


def wave_step(
    psi: Tensor,
    K: Tensor,
    beta: Tensor,
    g: Tensor,
    omega: Tensor,
    dilation: int,
    M: Tensor | None = None,
    mix_lowrank: tuple | None = None,
    normalize: bool = True,
    self_gate: Tensor | None = None,
) -> Tensor:
    """One full wave step: conv -> [mode-coupling] -> per-channel saturation ->
    rotation -> normalize.

    psi: (B, T, d) complex. Mode coupling: pass either a dense `M (d,d)` or a
    low-rank `mix_lowrank=(U, V)` with U,V (d,r). `self_gate` (optional, (B,T,d)
    real in (0,1)) makes the conv self-retention data-dependent (doc 12 selective
    probe); None => identity. Returns (B,T,d) (per-position RMS=1 when normalize).
    """
    p = causal_dilated_conv(psi, K, beta, dilation, self_gate=self_gate)
    if M is not None:
        p = channel_mix(p, M)
    elif mix_lowrank is not None:
        p = channel_mix_lowrank(p, mix_lowrank[0], mix_lowrank[1])
    q = saturating_nonlinearity(p, g)
    r = phase_rotation(q, omega)
    return normalize_field(r) if normalize else r


__all__ = [
    "causal_dilated_conv",
    "saturating_nonlinearity",
    "channel_mix",
    "channel_mix_lowrank",
    "selective_gate",
    "phase_rotation",
    "normalize_field",
    "wave_step",
]
