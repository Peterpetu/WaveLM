"""WaveLM configuration.

See spec/wave_lm/01_objects.md (Part 3 of the master plan). Every constant has a
justification recorded there; this module only carries the values and the
consistency checks. No hyperparameter is silently defaulted — each field maps to
a symbol in the specification.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal


@dataclass
class WaveConfig:
    # ---- alphabet ----
    # Byte-level: the alphabet is the 256 raw byte values. There is NO learned
    # tokenizer / vocabulary (master plan Part 1.4).
    V: int = 256

    # ---- field topology ----
    d: int = 256              # complex channel count
    T: int = 2048             # sequence length in bytes
    n_sub: int = 24           # internal computation steps per forward pass
    delta: int = 2            # conv taps per step (offsets 1..delta, dilated)

    # ---- dilation schedule ----
    # D_tau = 2^(tau mod L_cyc), L_cyc = ceil(log2(T)). Receptive field over one
    # cycle is delta*(2^L_cyc - 1) >= T (master plan Part 8). L_cyc is derived in
    # __post_init__ from T (not user-set) to keep the receptive-field theorem exact.
    L_cyc: int = field(init=False)

    # ---- nonlinearity / readout initial scalar values (pre-constraint-map) ----
    # Stored unconstrained; mapped via Part 7. These are the *target constrained*
    # values; WaveModel inverts the maps at construction.
    beta_init: float = 0.5    # self-retention in [0,1)
    g_init: float = 1.0       # interference strength > 0
    T_o_init: float = 1.0     # readout temperature > 0
    gamma_K: float = 0.6      # geometric tap-magnitude decay for K init

    # ---- mode coupling (channel mixing); off by default ----
    mix_channels: bool = False    # apply learned complex mode-coupling
    mix_every: int = 4            # apply it every `mix_every` internal steps
    mix_rank: int = 0             # 0 = dense (d,d); >0 = low-rank residual (untied per point)

    # ---- spectral-diversity conditional term (master plan Part 9.1) ----
    lambda_omega0: float = 0.01   # coefficient applied WHEN activated (else 0)
    spec_collapse_rho: float = 0.25   # activate if Var(omega) < rho * Var0
    spec_collapse_patience: int = 2   # consecutive validations below threshold
    spec_release_frac: float = 0.5    # deactivate when Var(omega) > frac * Var0

    # ---- optimisation (master plan Part 9.2) ----
    lr_peak: float = 3e-4
    lr_floor: float = 3e-5
    warmup_frac: float = 0.05
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    weight_decay: float = 0.01
    adam_eps: float = 1e-8
    grad_clip: float = 1.0        # global L2 grad-norm clip threshold
    checkpoint_every: int = 5     # ~ ceil(sqrt(n_sub)); activation checkpointing

    # ---- muP width-scaled learning rate (optimisation plan item 3) ----
    # When True, the optimiser uses per-role lr multipliers (1/fan_in relative to
    # the reference d0,r0) so the optimal base lr (lr_peak) transfers across width.
    # When False, the historical single-lr decay/no_decay grouping is used.
    mup: bool = False
    mup_d0: int = 256             # reference width at which lr_peak is tuned
    mup_r0: int = 32              # reference mode-coupling rank
    untie_embedding: bool = False # exact muP: separate input W_E and readout W_out

    # ---- capacity: untie the conv kernel across internal steps (item 4) ----
    # When True, K has shape (n_sub, delta, d) -- a distinct propagation filter per
    # internal step (adds (n_sub-1)*delta*d complex params). When False, K is shared
    # across steps (shape (delta, d), the historical default).
    untie_steps: bool = False

    # ---- ablation / analysis switches (driver-attribution study, doc 12) ----
    # All default False => forward is BIT-IDENTICAL to the pre-study model. Each
    # isolates one component so its bits-per-byte contribution can be measured.
    #   ablate_phase      : force omega = 0 (phase rotation e^{i*omega} becomes the
    #                       identity). Tests whether the per-channel CARRIER is
    #                       load-bearing or decorative (omega barely trains; STATUS
    #                       sec 3c). Keeps saturation + Born + normalisation.
    #   ablate_saturation : force g = 0 (saturation q = p/(1+g|p|^2) becomes identity)
    #                       while KEEPING normalisation AND the Born readout. Isolates
    #                       the |p|^2-product nonlinearity from the Born square.
    #   ablate_born       : use the linear readout Re<W_E,psi>/T_o while KEEPING g>0
    #                       AND normalisation. Isolates the Born square from the
    #                       saturation. (ablate_saturation x ablate_born = the 2x2 that
    #                       the single coupled `linear=True` switch cannot resolve.)
    # NOTE: the legacy `linear=True` forward arg (composition gate, Part 12) is the
    # fully-coupled case g=0 + linear readout + NO normalisation; it is preserved.
    ablate_phase: bool = False
    ablate_saturation: bool = False
    ablate_born: bool = False

    # ---- selective (data-dependent) gate: the "hidden lever" probe (doc 12) ----
    # When True, the conv self-retention term becomes INPUT-DEPENDENT:
    #   p[n,k] = beta * s[n,k] * psi[n,k] + sum_j K[j,k] psi[n - j*D, k],
    #   s[n] = sigmoid( (|psi[n]|^2 @ W_sel_in) @ W_sel_out + b_sel )  in (0,1)^d.
    # This makes the dynamics data-dependent (Mamba-style selectivity) -- the property
    # the otherwise linear-time-invariant WaveLM lacks. Off by default (bit-identical).
    # selective_rank is the low-rank width r_sel of the gate (adds 2*d*r_sel + d real
    # DOF). Recomputed every internal step from the current field (genuinely dynamic).
    selective: bool = False
    selective_rank: int = 16

    # ---- training run shape ----
    total_steps: int = 30_000
    batch_size: int = 16
    seed: int = 1729

    # ---- dtype ----
    # Complex64 (fp32 real/imag) throughout; the model is tiny so fp32 is afford-
    # able and avoids complex-bf16 gaps (master plan Part 9.2).
    real_dtype: Literal["float32", "float64"] = "float32"

    def __post_init__(self) -> None:
        assert self.V == 256, "WaveLM is byte-level; V must be 256."
        assert self.d >= 2, "need at least 2 channels"
        assert self.T >= 2, "need at least 2 positions for one CE term"
        assert self.n_sub >= 1
        assert self.delta >= 1
        assert 0.0 <= self.beta_init < 1.0
        assert self.g_init > 0.0
        assert self.T_o_init > 0.0
        assert 0.0 < self.gamma_K < 1.0
        assert self.mup_d0 >= 2 and self.mup_r0 >= 1
        if self.selective:
            assert self.selective_rank >= 1, "selective gate needs rank >= 1"
        # Derived dilation cycle length: ceil(log2(T)).
        self.L_cyc = max(1, math.ceil(math.log2(self.T)))
        # Receptive-field guarantee (Part 8): one cycle reaches delta*(2^L_cyc - 1).
        rf_one_cycle = self.delta * ((1 << self.L_cyc) - 1)
        assert rf_one_cycle >= self.T, (
            f"receptive field {rf_one_cycle} < T {self.T}; "
            f"increase delta or n_sub coverage"
        )

    def dilation(self, tau: int) -> int:
        """D_tau = 2^(tau mod L_cyc)."""
        return 1 << (tau % self.L_cyc)


def wave_micro_config() -> WaveConfig:
    """Tiny config for unit tests (fast, FP64-friendly)."""
    return WaveConfig(V=256, d=8, T=16, n_sub=6, delta=2, batch_size=2,
                      total_steps=50, real_dtype="float64")


def wave_base_config() -> WaveConfig:
    """Reference base model for TinyStories byte-level training."""
    return WaveConfig()


__all__ = ["WaveConfig", "wave_micro_config", "wave_base_config"]
