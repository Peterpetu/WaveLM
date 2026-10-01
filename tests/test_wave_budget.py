"""Budget / capacity tests (optimisation plan item 4).

  - test_param_count_formula   : model.param_count() == budget complex-elems formula;
                                  budget params_dof == real-DOF (spec §4.1). Reconciles
                                  the TWO conventions (fixes the spec's equate-error).
  - test_flops_formula          : budget MACs == §4.1 C(...) with ceiling P_pts,
                                  low-rank mixing, and the conv beta self-term.
  - test_untie_steps_shapes_and_grad     : (n_sub,Δ,d) K, per-step init invariant,
                                            FP64 grad on K[tau], loss decreases.
  - test_untie_steps_stream_equals_parallel : streaming stays bit-exact with untie_steps.
"""

import math
import os
import sys

import torch

from wave import WaveModel, WaveConfig, ce_loss
from wave.infer import WaveStreamer

try:
    from tools.budget import budget
except ImportError:                            # tools/ not importable as a package
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "tools"))
    from budget import budget


def _complex_elems(V, d, n_sub, delta, mix, mix_every, mix_rank, untie_steps, untie_emb):
    """Independent re-derivation of the complex-parameter count (count complex as 1)."""
    P = ((n_sub + mix_every - 1) // mix_every) if mix else 0      # ceiling
    p_WE = V * d * (2 if untie_emb else 1)
    p_K = (n_sub if untie_steps else 1) * delta * d
    p_mix = 0 if not mix else (d * d if mix_rank == 0 else 2 * P * d * mix_rank)
    return p_WE + p_K + p_mix


def _C_macs(V, d, T, n_sub, delta, mix, mix_every, mix_rank):
    """Spec §4.1 C(...): per-sequence complex MACs (conv self-term + taps, mixing, readout)."""
    P = ((n_sub + mix_every - 1) // mix_every) if mix else 0
    conv = n_sub * T * (d + delta * d)
    if not mix:
        mix_macs = 0
    elif mix_rank == 0:
        mix_macs = P * T * d * d
    else:
        mix_macs = P * (2 * T * d * mix_rank)
    readout = T * V * d
    return conv + mix_macs + readout


_CONFIGS = [
    dict(d=64, T=32, n_sub=8, delta=2, mix_channels=True, mix_rank=16, mix_every=2),
    dict(d=48, T=32, n_sub=12, delta=2, mix_channels=True, mix_rank=8, mix_every=5),  # ceiling matters
    dict(d=32, T=16, n_sub=6, delta=3, mix_channels=False),
    dict(d=40, T=16, n_sub=6, delta=2, mix_channels=True, mix_rank=0, mix_every=3),    # dense M
    dict(d=40, T=16, n_sub=6, delta=2, untie_steps=True),
    dict(d=40, T=16, n_sub=6, delta=2, untie_embedding=True),
]


def test_param_count_formula():
    """model.param_count() (complex-as-1) and budget's two conventions agree with an
    independent formula, for tied / low-rank / dense / no-mix / untie_steps / untie_emb."""
    for c in _CONFIGS:
        cfg = WaveConfig(V=256, seed=0, **c)
        m = WaveModel(cfg)
        b = budget(256, cfg.d, cfg.T, cfg.n_sub, cfg.delta, 1, cfg.mix_channels,
                   cfg.mix_every, cfg.checkpoint_every, mix_rank=cfg.mix_rank,
                   untie_steps=cfg.untie_steps, untie_embedding=cfg.untie_embedding)
        ce = _complex_elems(256, cfg.d, cfg.n_sub, cfg.delta, cfg.mix_channels,
                            cfg.mix_every, cfg.mix_rank, cfg.untie_steps, cfg.untie_embedding)
        expect_count = ce + cfg.d + 3              # + omega(d) + beta,g,T_o (real)
        expect_dof = 2 * ce + cfg.d + 3
        assert m.param_count() == expect_count, (c, m.param_count(), expect_count)
        assert b["params_complex_elems"] == expect_count, (c, b["params_complex_elems"])
        assert b["params_dof"] == expect_dof, (c, b["params_dof"], expect_dof)
        # the two conventions differ by exactly 2x on the complex part
        assert b["params_dof"] - b["params_complex_elems"] == ce


def test_flops_formula():
    """budget MACs (batch=1) equal the §4.1 C formula -- including the ceiling P_pts
    and the conv beta self-term that the old budget.py omitted."""
    for c in _CONFIGS:
        cfg = WaveConfig(V=256, seed=0, **c)
        b = budget(256, cfg.d, cfg.T, cfg.n_sub, cfg.delta, 1, cfg.mix_channels,
                   cfg.mix_every, cfg.checkpoint_every, mix_rank=cfg.mix_rank,
                   untie_steps=cfg.untie_steps, untie_embedding=cfg.untie_embedding)
        expect = _C_macs(256, cfg.d, cfg.T, cfg.n_sub, cfg.delta, cfg.mix_channels,
                         cfg.mix_every, cfg.mix_rank)
        assert b["fwd_macs"] == expect, (c, b["fwd_macs"], expect)
    # ceiling check: n_sub=12, mix_every=5 -> 3 mixing points (not floor 2)
    b = budget(256, 48, 32, 12, 2, 1, True, 5, 5, mix_rank=8)
    assert b["P_pts"] == 3


def test_untie_steps_shapes_and_grad():
    """untie_steps gives a per-step kernel (n_sub,Δ,d) that keeps the init energy
    invariant, has correct gradients (FP64 finite diff), and trains."""
    cfg = WaveConfig(V=256, d=4, T=8, n_sub=4, delta=2, untie_steps=True,
                     real_dtype="float64", seed=3)
    m = WaveModel(cfg)
    assert tuple(m.K.shape) == (cfg.n_sub, cfg.delta, cfg.d)

    # per-step init invariant: beta + sum_j |K[tau,j,k]| = 1
    beta = float(m.beta())
    sumK = m.K.abs().sum(dim=1)                    # (n_sub, d)
    assert torch.allclose(beta + sumK, torch.ones_like(sumK), atol=1e-5)

    # FP64 finite-difference gradient check on per-step kernel entries
    b = torch.randint(0, 256, (2, cfg.T))
    logits, _ = m(b, use_checkpoint=False)
    m.zero_grad()
    ce_loss(logits, b).backward()
    eps = 1e-6
    for idx in [(0, 0, 0), (cfg.n_sub - 1, 1, cfg.d - 1)]:
        with torch.no_grad():
            orig = m.K[idx].item()
            m.K[idx] = orig + eps
            lp = float(ce_loss(m(b, use_checkpoint=False)[0], b))
            m.K[idx] = orig - eps
            lm = float(ce_loss(m(b, use_checkpoint=False)[0], b))
            m.K[idx] = orig + 1j * eps
            lpi = float(ce_loss(m(b, use_checkpoint=False)[0], b))
            m.K[idx] = orig - 1j * eps
            lmi = float(ce_loss(m(b, use_checkpoint=False)[0], b))
            m.K[idx] = orig
        dre = (lp - lm) / (2 * eps)
        dim = (lpi - lmi) / (2 * eps)
        g = m.K.grad[idx]
        rre = abs(dre - g.real.item()) / (abs(dre) + abs(g.real.item()) + 1e-12)
        rim = abs(dim - g.imag.item()) / (abs(dim) + abs(g.imag.item()) + 1e-12)
        assert rre < 1e-5 and rim < 1e-5, (idx, dre, dim, g, rre, rim)

    # 30 steps decreases loss
    opt = torch.optim.AdamW(m.parameters(), lr=1e-2)
    losses = []
    for _ in range(30):
        opt.zero_grad()
        loss = ce_loss(m(b, use_checkpoint=False)[0], b)
        loss.backward()
        opt.step()
        losses.append(float(loss))
    assert losses[-1] < losses[0], (losses[0], losses[-1])


def test_untie_steps_stream_equals_parallel():
    """The streamer indexes K[tau] under untie_steps and must stay bit-identical."""
    cfg = WaveConfig(V=256, d=8, T=20, n_sub=6, delta=2, untie_steps=True,
                     mix_channels=True, mix_rank=4, mix_every=2,
                     real_dtype="float64", seed=4)
    m = WaveModel(cfg).eval()
    b = torch.randint(0, 256, (1, cfg.T))
    with torch.no_grad():
        par, _ = m(b, use_checkpoint=False)
    s = WaveStreamer(m, batch_size=1, device="cpu")
    for n in range(cfg.T):
        lg = s.step(b[:, n])
        assert torch.allclose(lg[0], par[0, n], atol=1e-9, rtol=1e-7), \
            (n, (lg[0] - par[0, n]).abs().max().item())
