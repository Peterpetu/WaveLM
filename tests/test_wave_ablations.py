"""Ablation-instrument tests (driver-attribution study, doc 12).

These gate the analysis instruments added for the performance-driver study. The
guiding principle: each switch must be proven CORRECT by an equivalence/limit
argument (not merely "it runs"), the new parameters must pass an FP64
finite-difference gradient check, and the defaults must leave the model unchanged.

  - test_default_flags_no_extra_params     : defaults add no params, forward finite
  - test_phase_ablation_equals_zero_omega   : ablate_phase == forcing omega=0 (exact)
  - test_saturation_ablation_g_zero_limit   : ablate_saturation == the g->0 limit;
                                              keeps norm+Born (differs from legacy linear)
  - test_born_ablation_is_linear_readout    : ablate_born == linear readout on the
                                              SAME (g>0, normalised) field
  - test_saturation_born_are_independent     : the 2x2 {sat}x{born} are 4 distinct maps
  - test_selective_off_identity_on_changes   : selective=False adds nothing; =True wires
  - test_selective_grad_finite_diff          : FP64 grads for W_sel_* AND complex K
                                              (the self_gate backward path)
  - test_real_model_grad_finite_diff         : FP64 grads for RealWaveModel params
  - test_real_model_trains_and_matches_dof    : 30-step loss decrease + matched real-DOF
"""

import math

import torch

from wave import (WaveModel, RealWaveModel, WaveConfig, ce_loss,
                  born_logits, linear_logits)


# ----------------------------------------------------------------------------
# shared FP64 finite-difference gradient checkers (same convention as
# tests/test_wave_model.py)
# ----------------------------------------------------------------------------
def _check_real(loss_fn, param, idx, name, eps=1e-6, tol=1e-5):
    with torch.no_grad():
        orig = param[idx].item()
        param[idx] = orig + eps
        lp = loss_fn().item()
        param[idx] = orig - eps
        lm = loss_fn().item()
        param[idx] = orig
    num = (lp - lm) / (2 * eps)
    ana = param.grad[idx].item()
    rel = abs(num - ana) / (abs(num) + abs(ana) + 1e-12)
    assert rel < tol, f"{name}{idx}: num={num} ana={ana} rel={rel}"


def _check_complex(loss_fn, param, idx, name, eps=1e-6, tol=1e-5):
    with torch.no_grad():
        orig = param[idx].item()
        param[idx] = orig + eps
        lp = loss_fn().item()
        param[idx] = orig - eps
        lm = loss_fn().item()
        param[idx] = orig
        dre = (lp - lm) / (2 * eps)
        param[idx] = orig + 1j * eps
        lp = loss_fn().item()
        param[idx] = orig - 1j * eps
        lm = loss_fn().item()
        param[idx] = orig
        dim = (lp - lm) / (2 * eps)
    g = param.grad[idx]
    rre = abs(dre - g.real.item()) / (abs(dre) + abs(g.real.item()) + 1e-12)
    rim = abs(dim - g.imag.item()) / (abs(dim) + abs(g.imag.item()) + 1e-12)
    assert rre < tol and rim < tol, f"{name}{idx}: dre={dre} dim={dim} g={g}"


_BASE = dict(V=256, d=8, T=8, n_sub=4, delta=2, real_dtype="float64", seed=3)


def _batch(n=2, T=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 256, (n, T), generator=g)


# ----------------------------------------------------------------------------
# defaults unchanged
# ----------------------------------------------------------------------------
def test_default_flags_no_extra_params():
    """All ablation flags default off => no selective params, forward is finite."""
    m = WaveModel(WaveConfig(**_BASE))
    assert m.W_sel_in is None and m.W_sel_out is None and m.W_sel_bias is None
    assert not m.cfg.ablate_phase and not m.cfg.ablate_saturation and not m.cfg.ablate_born
    logits, _ = m(_batch(), use_checkpoint=False)
    assert torch.isfinite(logits).all()


# ----------------------------------------------------------------------------
# phase ablation: ablate_phase <=> omega forced to 0
# ----------------------------------------------------------------------------
def test_phase_ablation_equals_zero_omega():
    """ablate_phase=True must be BIT-IDENTICAL to the same model with omega=0
    (theta_omega=0 => omega=pi*tanh(0)=0). Same seed => all other params identical."""
    m_abl = WaveModel(WaveConfig(ablate_phase=True, **_BASE))
    m_ref = WaveModel(WaveConfig(ablate_phase=False, **_BASE))
    with torch.no_grad():
        m_ref.theta_omega.zero_()                 # omega = 0 in the reference
    b = _batch()
    la, _ = m_abl(b, use_checkpoint=False)
    lr, _ = m_ref(b, use_checkpoint=False)
    assert torch.equal(la, lr), "ablate_phase != forcing omega=0"
    # and the (non-trivial, spread) default omega actually does something:
    m_def = WaveModel(WaveConfig(**_BASE))
    ld, _ = m_def(b, use_checkpoint=False)
    assert not torch.equal(ld, la), "default phase rotation had no effect (suspicious)"


# ----------------------------------------------------------------------------
# saturation ablation: ablate_saturation <=> g -> 0, keeping norm + Born
# ----------------------------------------------------------------------------
def test_saturation_ablation_g_zero_limit():
    """ablate_saturation=True (g=0 exactly) matches the g->0 limit (g_init=1e-8),
    and -- unlike the legacy linear=True -- KEEPS normalisation and the Born readout."""
    m_abl = WaveModel(WaveConfig(ablate_saturation=True, g_init=1.0, **_BASE))
    m_lim = WaveModel(WaveConfig(ablate_saturation=False, g_init=1e-8, **_BASE))
    b = _batch()
    la, _ = m_abl(b, use_checkpoint=False)
    ll, _ = m_lim(b, use_checkpoint=False)
    assert torch.allclose(la, ll, atol=1e-5), "ablate_saturation is not the g->0 limit"
    # legacy linear=True also sets g=0 but drops norm and squares -> must differ:
    l_linear, _ = m_abl(b, use_checkpoint=False, linear=True)
    assert not torch.allclose(la, l_linear), \
        "ablate_saturation should keep norm+Born (differ from legacy linear)"


# ----------------------------------------------------------------------------
# born ablation: ablate_born <=> linear readout on the SAME evolved field
# ----------------------------------------------------------------------------
def test_born_ablation_is_linear_readout():
    """ablate_born=True keeps g>0 + normalisation and only swaps |.|^2 for the linear
    readout. The evolved field psi is unchanged, so logits == linear_logits(psi,...)
    and != born_logits(psi,...)."""
    m = WaveModel(WaveConfig(ablate_born=True, **_BASE))
    b = _batch()
    logits, psi = m(b, use_checkpoint=False)
    expect_lin = linear_logits(psi, m.W_E, m.T_o())
    expect_born = born_logits(psi, m.W_E, m.T_o())
    assert torch.equal(logits, expect_lin), "ablate_born did not use the linear readout"
    assert not torch.equal(logits, expect_born), "ablate_born still returned Born logits"


def test_saturation_born_are_independent():
    """The 2x2 {saturation on/off} x {Born on/off} must be FOUR distinct maps -- the
    decoupling the single legacy `linear` switch could not provide."""
    b = _batch()
    outs = {}
    for sat in (False, True):
        for born in (False, True):
            m = WaveModel(WaveConfig(ablate_saturation=sat, ablate_born=born, **_BASE))
            outs[(sat, born)] = m(b, use_checkpoint=False)[0]
    keys = list(outs)
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            assert not torch.equal(outs[keys[i]], outs[keys[j]]), \
                f"{keys[i]} and {keys[j]} gave identical logits (not independent)"


# ----------------------------------------------------------------------------
# selective gate
# ----------------------------------------------------------------------------
def test_selective_off_identity_on_changes():
    """selective=False creates no gate params; selective=True creates them and wires
    a (finite) data-dependent path that changes the output."""
    m_off = WaveModel(WaveConfig(selective=False, **_BASE))
    assert m_off.W_sel_in is None
    m_on = WaveModel(WaveConfig(selective=True, selective_rank=4, **_BASE))
    assert m_on.W_sel_in is not None and m_on.W_sel_out is not None
    b = _batch()
    lon, _ = m_on(b, use_checkpoint=False)
    assert torch.isfinite(lon).all()
    # W_sel_out=0 at init => gate is a uniform constant s0; force it non-constant so
    # the selective path is exercised, then it must differ from the non-selective model.
    with torch.no_grad():
        m_on.W_sel_out.normal_(0.0, 0.3)
        m_on.W_sel_in.normal_(0.0, 0.5)
    lon2, _ = m_on(b, use_checkpoint=False)
    loff, _ = m_off(b, use_checkpoint=False)
    assert not torch.equal(lon2, loff), "selective path had no effect once activated"


def test_selective_grad_finite_diff():
    """FP64 finite-difference gradient check for the selective gate parameters AND a
    complex K element (the self_gate backward path through the complex conv)."""
    cfg = WaveConfig(selective=True, selective_rank=4, **_BASE)
    m = WaveModel(cfg)
    # move W_sel_out off zero so gradients flow to W_sel_in (LoRA init has W_out=0)
    with torch.no_grad():
        m.W_sel_out.normal_(0.0, 0.2)
        m.W_sel_in.normal_(0.0, 0.5)
        m.W_sel_bias.normal_(0.0, 0.5)
    b = _batch()

    def loss():
        logits, _ = m(b, use_checkpoint=False)
        return ce_loss(logits, b)

    m.zero_grad(); loss().backward()
    _check_real(loss, m.W_sel_in, (0, 0), "W_sel_in")
    _check_real(loss, m.W_sel_in, (cfg.d - 1, cfg.selective_rank - 1), "W_sel_in")
    _check_real(loss, m.W_sel_out, (0, 0), "W_sel_out")
    _check_real(loss, m.W_sel_out, (cfg.selective_rank - 1, cfg.d - 1), "W_sel_out")
    _check_real(loss, m.W_sel_bias, (0,), "W_sel_bias")
    _check_complex(loss, m.K, (0, 0), "K")          # gate path backprops to complex K


# ----------------------------------------------------------------------------
# real-valued structural twin
# ----------------------------------------------------------------------------
def test_real_model_grad_finite_diff():
    """FP64 finite-difference gradient check for all RealWaveModel parameter roles."""
    cfg = WaveConfig(V=256, d=8, T=8, n_sub=3, delta=2, mix_channels=True,
                     mix_rank=4, mix_every=2, real_dtype="float64", seed=4)
    m = RealWaveModel(cfg)
    with torch.no_grad():
        m.mix_V.normal_(0.0, 0.2)                   # LoRA init V=0 -> move so U gets grad
    b = _batch(seed=1)

    def loss():
        logits, _ = m(b)
        return ce_loss(logits, b)

    m.zero_grad(); loss().backward()
    present = int(b[0, 0].item())
    _check_real(loss, m.W_E, (present, 0), "W_E")
    _check_real(loss, m.W_E, (present, cfg.d - 1), "W_E")
    _check_real(loss, m.K, (0, 0), "K")
    _check_real(loss, m.K, (1, cfg.d - 1), "K")
    _check_real(loss, m.theta_beta, (), "theta_beta")
    _check_real(loss, m.theta_g, (), "theta_g")
    _check_real(loss, m.theta_To, (), "theta_To")
    _check_real(loss, m.mix_U, (0, 0, 0), "mix_U")
    _check_real(loss, m.mix_V, (0, 0, 0), "mix_V")


def test_real_model_trains_and_matches_dof():
    """RealWaveModel trains (loss decreases, finite) and a width-2d real twin is
    real-DOF-MATCHED to a width-d complex model -- the comparison basis. Doubling the
    real width exactly compensates for the complex->real x2 DOF halving, so the ratio
    is ~1.0 (the only gap is the d carrier-frequency params the real model lacks)."""
    # 30-step train on a fixed batch
    torch.manual_seed(0)
    cfg = WaveConfig(V=256, d=16, T=16, n_sub=4, delta=2, mix_channels=True,
                     mix_rank=8, mix_every=2, real_dtype="float32", seed=4)
    m = RealWaveModel(cfg)
    b = _batch(n=4, T=16, seed=2)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    first = last = None
    for _ in range(30):
        logits, _ = m(b)
        L = ce_loss(logits, b)
        assert torch.isfinite(L)
        opt.zero_grad(); L.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        first = float(L) if first is None else first
        last = float(L)
    assert last < first, f"RealWaveModel CE did not decrease: {first} -> {last}"

    # matched-DOF: real twin at width 2d is DOF-matched (ratio ~1.0) to complex at d.
    def real_dof(model):
        return sum(p.numel() * (2 if p.is_complex() else 1)
                   for p in model.parameters() if p.requires_grad)
    cplx = WaveModel(WaveConfig(V=256, d=32, T=16, n_sub=4, delta=2, mix_channels=True,
                                mix_rank=8, mix_every=2, real_dtype="float32", seed=4))
    real = RealWaveModel(WaveConfig(V=256, d=64, T=16, n_sub=4, delta=2, mix_channels=True,
                                    mix_rank=8, mix_every=2, real_dtype="float32", seed=4))
    ratio = real_dof(real) / real_dof(cplx)
    assert 0.9 < ratio < 1.1, f"width-2d real twin not DOF-matched to complex-at-d: {ratio}"
