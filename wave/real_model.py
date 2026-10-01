"""RealWaveModel -- the REAL-VALUED structural twin of WaveLM (driver study, doc 12).

The fair complex-vs-real test. It mirrors WaveLM's five-stage macro-structure but in
REAL arithmetic, with NO phase (a real scalar has no phase to rotate):

    x^(0)[n] = W_E[o_n]                                          (real embedding)
    repeat n_sub times:
      (1) real dilated causal conv   p[n,k] = beta x[n,k] + sum_j K[j,k] x[n-jD,k]
      (2) [optional] real mode coupling  (dense M, or low-rank residual p + U(V^T p))
      (3) real radial saturation     q = p / (1 + g * p^2)
      (4) [NO phase rotation -- the complex e^{i*omega} has no real analogue]
      (5) real RMS normalisation     x^+ [n] = sqrt(d) * q[n] / ||q[n]||
    readout (squared, like Born)     a(o,n) = <W_E[o], x[n]> ,  l = a^2 / T_o

It is built to **matched real degrees of freedom**: a complex WaveLM parameter is 2
real DOF, so a complex model of width d is compared against a RealWaveModel of width
~2d (the driver matches DOF exactly by searching width). Everything else -- the
depthwise dilated conv, the channel mixing, the radial saturation, the RMSNorm, and
the SQUARED readout -- is identical in form, so the only thing removed is the
complex/phase (interference) structure. If the real twin matches or beats complex
WaveLM at matched DOF, the complex "wave" structure is not the performance driver.

This deliberately reuses WaveConfig for shape/schedule (interpreting cfg.d as the
real width and reusing cfg.dilation / cfg.L_cyc / the receptive-field assertion); the
complex-only flags (ablate_phase, selective, untie_embedding, mup) are ignored here.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .config import WaveConfig


def _inv_softplus(y: float) -> float:
    """theta such that softplus(theta) = y, for y > 0."""
    return math.log(math.expm1(y))


def _real_causal_dilated_conv(x: Tensor, K: Tensor, beta: Tensor, dilation: int) -> Tensor:
    """p[n,k] = beta*x[n,k] + sum_{j=1}^{Delta} K[j-1,k]*x[n - j*dilation, k] (real, causal)."""
    B, T, d = x.shape
    Delta = K.shape[0]
    p = beta * x
    for j in range(1, Delta + 1):
        s = j * dilation
        if s >= T:
            break
        contrib = K[j - 1].view(1, 1, d) * x[:, : T - s, :]
        p = p + F.pad(contrib, (0, 0, s, 0))
    return p


def _real_saturation(p: Tensor, g: Tensor) -> Tensor:
    """Real radial saturation q = p/(1 + g*p^2) (the real form of p/(1+g|p|^2))."""
    return p / (1.0 + g * (p * p))


def _real_channel_mix_lowrank(p: Tensor, U: Tensor, V: Tensor) -> Tensor:
    """Real low-rank residual mode coupling p <- p + U (V^T p), U,V (d,r) real."""
    z = torch.einsum("btk,kr->btr", p, V)
    Ap = torch.einsum("btr,kr->btk", z, U)
    return p + Ap


def _real_normalize(x: Tensor, eps: float = 1e-8) -> Tensor:
    """RMS normalise each position to fixed energy ||x[n]|| = sqrt(d) (real RMSNorm)."""
    d = x.shape[-1]
    norm = (x * x).sum(dim=-1, keepdim=True).sqrt()
    return x * math.sqrt(d) / (norm + eps)


def _real_squared_readout(x: Tensor, W_E: Tensor, T_o: Tensor) -> Tensor:
    """l(o|n) = <W_E[o], x[n]>^2 / T_o (the real analogue of the Born |.|^2 readout)."""
    a = torch.einsum("btk,vk->btv", x, W_E)        # (B,T,V) real
    return (a * a) / T_o


class RealWaveModel(nn.Module):
    def __init__(self, cfg: WaveConfig):
        super().__init__()
        self.cfg = cfg
        V, d, Delta = cfg.V, cfg.d, cfg.delta
        rdtype = torch.float64 if cfg.real_dtype == "float64" else torch.float32
        self._rdtype = rdtype

        gen = torch.Generator().manual_seed(cfg.seed)

        # ---- W_E real (V, d): var 1/d so E||W_E[o]||^2 = d*(1/d) = 1 ----
        self.W_E = nn.Parameter(
            (torch.randn(V, d, generator=gen) / math.sqrt(d)).to(rdtype))

        # ---- K real (Delta, d): geometric-decay magnitudes, random SIGN (no phase) ----
        # |K[j]| = (1-beta)*gamma^j / sum gamma^j; sign +/-1 uniform. Guarantees
        # beta + sum_j |K[j,k]| = 1 at init (the marginally-stable propagation medium).
        gamma = cfg.gamma_K
        geom = torch.tensor([gamma ** j for j in range(1, Delta + 1)])
        mags = (1.0 - cfg.beta_init) * geom / geom.sum()        # (Delta,)
        signs = (torch.randint(0, 2, (Delta, d), generator=gen) * 2 - 1).to(rdtype)
        self.K = nn.Parameter(mags.view(Delta, 1).to(rdtype) * signs)

        # ---- scalar maps (same as WaveModel; no omega) ----
        self.theta_beta = nn.Parameter(torch.tensor(
            math.log(cfg.beta_init / (1.0 - cfg.beta_init)), dtype=rdtype))
        self.theta_g = nn.Parameter(torch.tensor(_inv_softplus(cfg.g_init), dtype=rdtype))
        self.theta_To = nn.Parameter(torch.tensor(_inv_softplus(cfg.T_o_init), dtype=rdtype))

        # ---- optional real mode coupling (mirror WaveModel: dense or low-rank) ----
        self.M = None
        self.mix_U = None
        self.mix_V = None
        if cfg.mix_channels:
            n_points = (cfg.n_sub + cfg.mix_every - 1) // cfg.mix_every
            if cfg.mix_rank == 0:
                eye = torch.eye(d, dtype=rdtype)
                noise = torch.randn(d, d, generator=gen).to(rdtype)
                self.M = nn.Parameter(eye + (0.02 / math.sqrt(d)) * noise)
            else:
                r = cfg.mix_rank
                s = 1.0 / math.sqrt(r)
                u = (torch.randn(n_points, d, r, generator=gen) * s).to(rdtype)
                self.mix_U = nn.Parameter(u)
                self.mix_V = nn.Parameter(torch.zeros(n_points, d, r, dtype=rdtype))

    # ------------------------------------------------------------------
    def beta(self) -> Tensor:
        return torch.sigmoid(self.theta_beta)

    def g(self) -> Tensor:
        return F.softplus(self.theta_g)

    def T_o(self) -> Tensor:
        return F.softplus(self.theta_To)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, byte_ids: Tensor, use_checkpoint=None):
        """byte_ids: (B, T) long in [0,256). Returns (logits (B,T,V) real, x (B,T,d)).

        `use_checkpoint` is accepted for harness compatibility and ignored (the real
        model is cheap; no activation checkpointing needed at the studied scales).
        """
        cfg = self.cfg
        beta, g = self.beta(), self.g()
        x = self.W_E[byte_ids]                                  # (B, T, d) real
        for tau in range(cfg.n_sub):
            p = _real_causal_dilated_conv(x, self.K, beta, cfg.dilation(tau))
            if cfg.mix_channels and tau % cfg.mix_every == 0:
                pt = tau // cfg.mix_every
                if self.M is not None:
                    p = torch.einsum("btk,kj->btj", p, self.M)
                else:
                    p = _real_channel_mix_lowrank(p, self.mix_U[pt], self.mix_V[pt])
            q = _real_saturation(p, g)
            x = _real_normalize(q)
        logits = _real_squared_readout(x, self.W_E, self.T_o())
        return logits, x


__all__ = ["RealWaveModel"]
