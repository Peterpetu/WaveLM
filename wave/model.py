"""WaveModel: the complete WaveLM (master plan Parts 3-9).

Parameters (stored unconstrained, mapped via Part 7):
  W_E      complex (V, d)         byte embedding (also the readout basis)
  K        complex (Delta, d)     causal conv kernel
  theta_omega real (d,)           omega = pi * tanh(theta_omega)  in (-pi, pi)
  theta_beta  real ()             beta  = sigmoid(theta_beta)     in (0, 1)
  theta_g     real ()             g     = softplus(theta_g)       > 0
  theta_To    real ()             T_o   = softplus(theta_To)      > 0

forward(bytes) -> (logits, psi):
  psi = W_E[bytes]; for tau in 0..n_sub-1: psi = wave_step(...); logits = born.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.checkpoint import checkpoint

from .config import WaveConfig
from .field import wave_step, selective_gate
from .readout import born_logits, linear_logits


def _inv_softplus(y: float) -> float:
    """theta such that softplus(theta) = y, for y > 0."""
    # softplus(x) = log(1+e^x) = y  ->  x = log(e^y - 1)
    return math.log(math.expm1(y))


class WaveModel(nn.Module):
    def __init__(self, cfg: WaveConfig):
        super().__init__()
        self.cfg = cfg
        V, d, Delta = cfg.V, cfg.d, cfg.delta
        rdtype = torch.float64 if cfg.real_dtype == "float64" else torch.float32
        cdtype = torch.complex128 if rdtype == torch.float64 else torch.complex64
        self._cdtype = cdtype
        self._rdtype = rdtype

        gen = torch.Generator().manual_seed(cfg.seed)

        # ---- W_E: complex (V, d), x,y ~ N(0, 1/(2d)) so E||W_E[o]||^2 = 1 ----
        scale = math.sqrt(1.0 / (2.0 * d))
        we_real = torch.randn(V, d, generator=gen) * scale
        we_imag = torch.randn(V, d, generator=gen) * scale
        self.W_E = nn.Parameter(torch.complex(we_real, we_imag).to(cdtype))

        # ---- optional untied readout basis W_out (exact muP, opt-plan item 3) ----
        # Off by default: the Born readout reuses W_E (the predecessor's bilinear
        # contrastive readout). On: a separate (V,d) readout basis, so input fan-in
        # (1) and output fan-in (d) get exact, not geometric-mean, muP lr's.
        self.W_out = None
        if cfg.untie_embedding:
            wo_real = torch.randn(V, d, generator=gen) * scale
            wo_imag = torch.randn(V, d, generator=gen) * scale
            self.W_out = nn.Parameter(torch.complex(wo_real, wo_imag).to(cdtype))

        # ---- K: complex (Delta, d). |K[j]| = (1-beta)*gamma^j / sum gamma^j;
        #         phase ~ U(0, 2pi). Guarantees beta + sum_j|K[j]| = 1 at init. ----
        gamma = cfg.gamma_K
        geom = torch.tensor([gamma ** j for j in range(1, Delta + 1)])
        mags = (1.0 - cfg.beta_init) * geom / geom.sum()        # (Delta,)
        if cfg.untie_steps:
            # One independent conv kernel per internal step (capacity item 4),
            # shape (n_sub, Delta, d). Each step's kernel has the SAME magnitudes
            # (so beta + sum_j|K[tau,j,k]| = 1 holds per step) but fresh phases.
            phases = torch.rand(cfg.n_sub, Delta, d, generator=gen) * (2.0 * math.pi)
            K_init = (mags.view(1, Delta, 1) * torch.exp(1j * phases)).to(cdtype)
        else:
            phases = torch.rand(Delta, d, generator=gen) * (2.0 * math.pi)
            K_init = (mags.view(Delta, 1) * torch.exp(1j * phases)).to(cdtype)
        self.K = nn.Parameter(K_init)

        # ---- omega via pi*tanh(theta); init uniform spread in (-pi, pi) ----
        k_idx = torch.arange(d, dtype=rdtype)
        omega_target = math.pi * (2.0 * (k_idx + 0.5) / d - 1.0)   # in (-pi, pi)
        theta_omega = torch.atanh((omega_target / math.pi).clamp(-0.999999, 0.999999))
        self.theta_omega = nn.Parameter(theta_omega.to(rdtype))

        # ---- scalar maps ----
        self.theta_beta = nn.Parameter(torch.tensor(
            math.log(cfg.beta_init / (1.0 - cfg.beta_init)), dtype=rdtype))   # logit(beta)
        self.theta_g = nn.Parameter(torch.tensor(_inv_softplus(cfg.g_init), dtype=rdtype))
        self.theta_To = nn.Parameter(torch.tensor(_inv_softplus(cfg.T_o_init), dtype=rdtype))

        # ---- optional mode coupling ----
        # dense (mix_rank=0): a single (d,d) matrix reused at every mixing point;
        # low-rank (mix_rank=r>0): untied residual U Vᴴ per mixing point.
        self.M = None
        self.mix_U = None
        self.mix_V = None
        if cfg.mix_channels:
            n_points = (cfg.n_sub + cfg.mix_every - 1) // cfg.mix_every
            if cfg.mix_rank == 0:
                eye = torch.eye(d, dtype=cdtype)
                noise = (torch.randn(d, d, generator=gen)
                         + 1j * torch.randn(d, d, generator=gen))
                self.M = nn.Parameter(eye + (0.02 / math.sqrt(d)) * noise.to(cdtype))
            else:
                r = cfg.mix_rank
                # LoRA-style init: U small random, V = 0 so the residual U(Vᴴp) is
                # EXACTLY zero at init. The block starts as identity (preserves a
                # warm-started base perfectly) and learns mode coupling gradually
                # (V receives gradient via U; once V != 0, U also learns).
                # muP wants Var(U) ~ 1/r (U contracts over fan-in r); default keeps
                # 1/d for bit-reproducibility of pre-muP runs.
                s = 1.0 / math.sqrt(r if cfg.mup else d)
                u = (torch.randn(n_points, d, r, generator=gen)
                     + 1j * torch.randn(n_points, d, r, generator=gen)) * s
                self.mix_U = nn.Parameter(u.to(cdtype))
                self.mix_V = nn.Parameter(torch.zeros(n_points, d, r, dtype=cdtype))

        # ---- optional selective (data-dependent) gate (doc 12 probe) ----
        # Makes the conv self-retention input-dependent: s[n]=sigmoid(|psi[n]|^2 @
        # W_sel_in @ W_sel_out + b_sel). LoRA-style init (W_sel_out=0) so the gate is
        # a uniform ~constant s0=0.95 at init -> the model starts ~= the non-selective
        # one and LEARNS selectivity (W_sel_out gets gradient first, then W_sel_in).
        self.W_sel_in = None
        self.W_sel_out = None
        self.W_sel_bias = None
        if cfg.selective:
            r_sel = cfg.selective_rank
            self.W_sel_in = nn.Parameter(
                torch.randn(d, r_sel, generator=gen, dtype=rdtype) / math.sqrt(d))
            self.W_sel_out = nn.Parameter(torch.zeros(r_sel, d, dtype=rdtype))
            s0 = 0.95
            self.W_sel_bias = nn.Parameter(
                torch.full((d,), math.log(s0 / (1.0 - s0)), dtype=rdtype))

    # ------------------------------------------------------------------
    # Constrained-parameter accessors (Part 7 maps)
    # ------------------------------------------------------------------
    def omega(self) -> Tensor:
        return math.pi * torch.tanh(self.theta_omega)

    def beta(self) -> Tensor:
        return torch.sigmoid(self.theta_beta)

    def g(self) -> Tensor:
        return F.softplus(self.theta_g)

    def T_o(self) -> Tensor:
        return F.softplus(self.theta_To)

    # ------------------------------------------------------------------
    # Evolution
    # ------------------------------------------------------------------
    def _evolve(self, psi: Tensor, use_checkpoint: bool, linear: bool = False) -> Tensor:
        """Run all n_sub wave steps, optionally activation-checkpointed in chunks.

        linear=True forces g=0 (intensity nonlinearity off) for the expressivity
        ablation; combined with linear_logits this makes the model linear in the
        embeddings (master plan Part 12).
        """
        cfg = self.cfg
        K, beta = self.K, self.beta()
        # Phase ablation (doc 12): omega=0 makes e^{i*omega} the identity.
        omega = torch.zeros_like(self.theta_omega) if cfg.ablate_phase else self.omega()
        # Saturation off if legacy `linear` (coupled gate) OR `ablate_saturation`
        # (decoupled: keeps normalisation + Born). g=0 makes q=p (identity).
        sat_off = linear or cfg.ablate_saturation
        g = self.theta_g.new_zeros(()) if sat_off else self.g()
        step_size = max(1, cfg.checkpoint_every)

        normalize = not linear   # the legacy linear ablation must stay linear (no norm)
        mixing = cfg.mix_channels and (not linear)
        selective = cfg.selective and (not linear)

        def run_chunk(psi_in: Tensor, lo: int, hi: int) -> Tensor:
            out = psi_in
            for tau in range(lo, hi):
                K_tau = self.K[tau] if cfg.untie_steps else K   # per-step kernel (item 4)
                M_tau, lr_tau = None, None
                if mixing and tau % cfg.mix_every == 0:
                    pt = tau // cfg.mix_every
                    if self.M is not None:
                        M_tau = self.M
                    else:
                        lr_tau = (self.mix_U[pt], self.mix_V[pt])
                # data-dependent gate recomputed from the CURRENT field each step
                sg = (selective_gate(out, self.W_sel_in, self.W_sel_out, self.W_sel_bias)
                      if selective else None)
                out = wave_step(out, K_tau, beta, g, omega, cfg.dilation(tau),
                                M=M_tau, mix_lowrank=lr_tau, normalize=normalize,
                                self_gate=sg)
            return out

        tau = 0
        while tau < cfg.n_sub:
            hi = min(tau + step_size, cfg.n_sub)
            if use_checkpoint and psi.requires_grad:
                # Bind lo/hi via default args to avoid late-binding in the closure.
                psi = checkpoint(
                    lambda x, lo=tau, h=hi: run_chunk(x, lo, h),
                    psi, use_reentrant=False,
                )
            else:
                psi = run_chunk(psi, tau, hi)
            tau = hi
        return psi

    def forward(
        self,
        byte_ids: Tensor,
        use_checkpoint: Optional[bool] = None,
        linear: bool = False,
    ) -> Tuple[Tensor, Tensor]:
        """byte_ids: (B, T) long in [0,256). Returns (logits (B,T,V), psi (B,T,d)).

        linear=True selects the linear-in-embeddings ablation (Part 12): g=0 and a
        linear readout. The real model uses linear=False (Born readout).
        """
        if use_checkpoint is None:
            use_checkpoint = self.training
        psi0 = self.W_E[byte_ids]                       # (B, T, d) complex (input basis)
        psi = self._evolve(psi0, use_checkpoint=use_checkpoint, linear=linear)
        W_read = self.W_out if self.W_out is not None else self.W_E   # readout basis
        # Born square off if legacy `linear` OR `ablate_born` (decoupled: keeps g>0 +
        # normalisation, swaps |.|^2 for the linear readout Re<W_E,psi>/T_o).
        if linear or self.cfg.ablate_born:
            logits = linear_logits(psi, W_read, self.T_o())
        else:
            logits = born_logits(psi, W_read, self.T_o())
        return logits, psi

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def omega_variance(self) -> float:
        with torch.no_grad():
            w = self.omega()
            return float(((w - w.mean()) ** 2).mean().item())

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    # ------------------------------------------------------------------
    # muP per-role optimiser groups (optimisation plan item 3)
    # ------------------------------------------------------------------
    def mup_param_groups(self, eta0: float, weight_decay: float) -> list:
        """Per-role AdamW groups with lr multipliers `lr_mult = 1/fan_in` (relative
        to the reference width d0 and rank r0).

        The cosine schedule scales every group's lr by the same base factor, so the
        muP *ratios* are preserved and the optimal base lr eta0 transfers across
        width d (and rank r). Fan-in classification (derived in
        spec/wave_lm/07_optimization_scaling_plan.md Section 3):
          W_E (tied)  -> geo-mean of input(1), output(d)  -> sqrt(d0/d)
          W_in,W_out  -> 1 , d0/d                          (when untie_embedding)
          K           -> Delta = O(1)                      -> 1
          M (dense)   -> d                                 -> d0/d
          mix_U       -> r                                 -> r0/r
          mix_V       -> d                                 -> d0/d
          omega,scalars -> 1                               -> 1   (no weight decay)
        """
        cfg = self.cfg
        d0, r0, d = float(cfg.mup_d0), float(cfg.mup_r0), float(cfg.d)
        r = float(cfg.mix_rank) if (cfg.mix_rank and cfg.mix_rank > 0) else 1.0
        groups: list = []

        def add(name, params, mult, wd):
            ps = [p for p in params if (p is not None and p.requires_grad)]
            if ps:
                groups.append({"name": name, "params": ps, "lr": eta0 * mult,
                               "lr_mult": mult, "weight_decay": wd})

        if cfg.untie_embedding and self.W_out is not None:
            add("W_in", [self.W_E], 1.0, weight_decay)        # input lookup: fan-in 1
            add("W_out", [self.W_out], d0 / d, weight_decay)  # readout: fan-in d
        else:
            add("W_E", [self.W_E], math.sqrt(d0 / d), weight_decay)
        add("K", [self.K], 1.0, weight_decay)
        if self.M is not None:
            add("M", [self.M], d0 / d, weight_decay)
        if self.mix_U is not None:
            add("mix_U", [self.mix_U], r0 / r, weight_decay)
            add("mix_V", [self.mix_V], d0 / d, weight_decay)
        if self.W_sel_in is not None:
            # gate matrices contract over d-dim features -> fan-in ~ d -> d0/d
            add("W_sel", [self.W_sel_in, self.W_sel_out], d0 / d, weight_decay)
        scalar_params = [self.theta_omega, self.theta_beta, self.theta_g, self.theta_To]
        if self.W_sel_bias is not None:
            scalar_params.append(self.W_sel_bias)   # bias: no weight decay, lr_mult 1
        add("scalars", scalar_params, 1.0, 0.0)

        # Every trainable parameter is covered exactly once (no silent drops/dups).
        covered = sum(len(g["params"]) for g in groups)
        n_params = sum(1 for p in self.parameters() if p.requires_grad)
        assert covered == n_params, f"muP groups cover {covered} of {n_params} params"
        return groups


__all__ = ["WaveModel"]
