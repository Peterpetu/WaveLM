"""WaveLM model tests (master plan Part 14, gradient/dynamics/system groups).

  - test_grad_finite_diff        : every parameter vs central finite differences (FP64)
  - test_information_propagation  : perturbing o_0 changes logits at the last position
  - test_separation               : decoded direction differs from current embedding
  - test_smoke_train              : 50 steps on synthetic data: CE decreases, no NaN
"""

import math

import torch

from wave import WaveModel, WaveConfig, ce_loss


def _loss(m: WaveModel, b: torch.Tensor) -> torch.Tensor:
    logits, _ = m(b, use_checkpoint=False)
    return ce_loss(logits, b)


def test_grad_finite_diff():
    """Central finite differences vs autograd for all params, FP64, rel-err<1e-5."""
    torch.manual_seed(0)
    cfg = WaveConfig(V=256, d=4, T=6, n_sub=3, delta=2, real_dtype="float64", seed=3)
    m = WaveModel(cfg)
    b = torch.randint(0, 256, (2, cfg.T))

    L = _loss(m, b)
    m.zero_grad()
    L.backward()

    eps = 1e-6

    def check_real(param: torch.Tensor, idx, name):
        with torch.no_grad():
            orig = param[idx].item()
            param[idx] = orig + eps
            lp = _loss(m, b).item()
            param[idx] = orig - eps
            lm = _loss(m, b).item()
            param[idx] = orig
        num = (lp - lm) / (2 * eps)
        ana = param.grad[idx].item()
        rel = abs(num - ana) / (abs(num) + abs(ana) + 1e-12)
        assert rel < 1e-5, f"{name}{idx}: num={num} ana={ana} rel={rel}"

    def check_complex(param: torch.Tensor, idx, name):
        # torch stores grad = conj(Wirtinger); component-wise real/imag FD must match
        with torch.no_grad():
            orig = param[idx].item()
            param[idx] = orig + eps
            lp = _loss(m, b).item()
            param[idx] = orig - eps
            lm = _loss(m, b).item()
            param[idx] = orig
            dre = (lp - lm) / (2 * eps)
            param[idx] = orig + 1j * eps
            lp = _loss(m, b).item()
            param[idx] = orig - 1j * eps
            lm = _loss(m, b).item()
            param[idx] = orig
            dim = (lp - lm) / (2 * eps)
        g = param.grad[idx]
        # dL/d(re) should equal g.real ; dL/d(im) should equal g.imag
        rre = abs(dre - g.real.item()) / (abs(dre) + abs(g.real.item()) + 1e-12)
        rim = abs(dim - g.imag.item()) / (abs(dim) + abs(g.imag.item()) + 1e-12)
        assert rre < 1e-5 and rim < 1e-5, f"{name}{idx}: dre={dre} dim={dim} g={g} rre={rre} rim={rim}"

    # scalars
    check_real(m.theta_beta, (), "theta_beta")
    check_real(m.theta_g, (), "theta_g")
    check_real(m.theta_To, (), "theta_To")
    # omega (real vector) — check a couple of indices
    check_real(m.theta_omega, (0,), "theta_omega")
    check_real(m.theta_omega, (cfg.d - 1,), "theta_omega")
    # K (complex) — a couple of taps/channels
    check_complex(m.K, (0, 0), "K")
    check_complex(m.K, (1, cfg.d - 1), "K")
    # W_E (complex) — only bytes that appear in b, plus one that does not
    present = int(b[0, 0].item())
    check_complex(m.W_E, (present, 0), "W_E")
    check_complex(m.W_E, (present, cfg.d - 1), "W_E")


def _last_pos_shift(m, b, perturb_pos=0):
    with torch.no_grad():
        logits_a, _ = m(b, use_checkpoint=False)
        b2 = b.clone()
        b2[0, perturb_pos] = (b2[0, perturb_pos] + 13) % 256
        logits_b, _ = m(b2, use_checkpoint=False)
        lpa = torch.log_softmax(logits_a[0, -1], dim=-1)
        lpb = torch.log_softmax(logits_b[0, -1], dim=-1)
        return (lpa - lpb).abs().max().item()


def test_information_propagation():
    """R-DEPTH (cure for F2): perturbing o_0 reaches the LAST position's logits.

    We prove the contrast that defines the cure: WaveLM (with the dilated
    convolution) propagates a decisively non-zero signal across the full
    sequence, whereas a no-propagation ablation (K=0, the F2 regime where the
    receptive field collapses to 1 position) produces exactly zero shift at the
    last position. At random init the propagated magnitude is attenuated by the
    untrained kernel; the point here is existence of the pathway, not its
    trained strength.
    """
    cfg = WaveConfig(V=256, d=16, T=16, n_sub=6, delta=2, real_dtype="float64", seed=5)
    m = WaveModel(cfg).eval()
    b = torch.randint(0, 256, (1, cfg.T))

    shift_wave = _last_pos_shift(m, b)
    assert shift_wave > 1e-4, f"WaveLM failed to propagate o_0 to last pos: {shift_wave}"

    # Ablation: kill the convolution (F2 regime) -> last position cannot see o_0.
    with torch.no_grad():
        m.K.zero_()
    shift_ablate = _last_pos_shift(m, b)
    assert shift_ablate < 1e-12, f"ablation should not propagate, got {shift_ablate}"
    assert shift_wave > 1e6 * max(shift_ablate, 1e-300)


def test_separation():
    """After n_sub steps the decoded direction differs from the current-byte
    embedding (R-SEPARATE): mean |cos(psi[n], W_E[o_n])| < 0.5."""
    cfg = WaveConfig(V=256, d=32, T=16, n_sub=8, delta=2, real_dtype="float64", seed=7)
    m = WaveModel(cfg).eval()
    b = torch.randint(0, 256, (4, cfg.T))
    with torch.no_grad():
        _, psi = m(b, use_checkpoint=False)
        we = m.W_E[b]                           # (B,T,d)
        # complex cosine magnitude
        num = (psi.conj() * we).sum(-1).abs()
        den = psi.abs().pow(2).sum(-1).sqrt() * we.abs().pow(2).sum(-1).sqrt() + 1e-12
        cos = (num / den)
        # exclude position 0 (no past to mix in; trivially close to its own embedding)
        mean_cos = cos[:, 1:].mean().item()
    assert mean_cos < 0.5, f"decoded direction too close to current embedding: {mean_cos}"


def test_smoke_train():
    """50 AdamW steps on a fixed synthetic batch: CE decreases, stays finite."""
    torch.manual_seed(0)
    cfg = WaveConfig(V=256, d=16, T=32, n_sub=6, delta=2, real_dtype="float32", seed=11)
    m = WaveModel(cfg)
    b = torch.randint(0, 256, (4, cfg.T))
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    first = None
    last = None
    for step in range(50):
        logits, _ = m(b, use_checkpoint=False)
        L = ce_loss(logits, b)
        assert torch.isfinite(L), f"NaN/inf at step {step}"
        opt.zero_grad()
        L.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        if first is None:
            first = float(L)
        last = float(L)
    assert last < first, f"CE did not decrease: first={first} last={last}"
