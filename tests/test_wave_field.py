"""WaveLM field/identity tests (master plan Part 14, identity group).

Covers:
  - test_wave_step_norm_bound       : Part 5 stability theorem
  - test_causality                  : future tokens cannot affect past logits
  - test_init_norm_invariants       : beta + sum|K| = 1; E||W_E[o]||^2 = 1
  - test_limiter_derivative         : intensity-nonlinearity Jacobian vs finite diff
"""

import math

import torch

from wave import WaveModel, wave_micro_config
from wave.field import saturating_nonlinearity, wave_step


def test_wave_step_norm_bound():
    """Norm-preserving dynamics: ||psi^(tau)[n]|| == sqrt(d) for all tau >= 1
    (revised Part 5 -- fixed field energy, so it cannot decay or blow up at
    depth; the sqrt(d) target keeps per-channel magnitudes O(1))."""
    cfg = wave_micro_config()
    m = WaveModel(cfg)
    target = math.sqrt(cfg.d)
    b = torch.randint(0, 256, (4, cfg.T))
    psi = m.W_E[b]
    K, beta, gg, omega = m.K, m.beta(), m.g(), m.omega()
    for tau in range(cfg.n_sub):
        psi = wave_step(psi, K, beta, gg, omega, cfg.dilation(tau))
        per_pos = (psi.real**2 + psi.imag**2).sum(-1).sqrt()
        assert torch.allclose(per_pos, torch.full_like(per_pos, target), atol=1e-4), (
            f"step {tau}: norm != sqrt(d), max dev "
            f"{(per_pos - target).abs().max().item()}")


def test_causality():
    """Changing a future byte leaves earlier-position logits bit-identical."""
    cfg = wave_micro_config()
    m = WaveModel(cfg).eval()
    b = torch.randint(0, 256, (2, cfg.T))
    with torch.no_grad():
        logits_a, _ = m(b, use_checkpoint=False)
        b2 = b.clone()
        # perturb the LAST position only
        b2[:, -1] = (b2[:, -1] + 7) % 256
        logits_b, _ = m(b2, use_checkpoint=False)
    # all positions strictly before T-1 must be identical
    assert torch.equal(logits_a[:, :-1, :], logits_b[:, :-1, :])


def test_init_norm_invariants():
    """beta + sum_j|K[j,k]| = 1 (per channel) and E||W_E[o]||^2 ~ 1 at init."""
    cfg = wave_micro_config()
    m = WaveModel(cfg)
    beta = float(m.beta())
    sumK = m.K.abs().sum(dim=0)             # (d,)
    total = beta + sumK
    assert torch.allclose(total, torch.ones_like(total), atol=1e-5), total
    we_energy = (m.W_E.real**2 + m.W_E.imag**2).sum(-1).mean().item()
    assert abs(we_energy - 1.0) < 0.1, we_energy


def test_limiter_derivative():
    """Autograd Jacobian of the per-channel saturating nonlinearity matches finite
    differences. Verifies Part 6.2(b) numerically via real/imag perturbations (FP64).
    """
    torch.manual_seed(0)
    d = 5
    p = torch.randn(1, 1, d, dtype=torch.complex128, requires_grad=True)
    g = torch.tensor(0.7, dtype=torch.float64)

    def f(pp):
        return saturating_nonlinearity(pp, g)

    # per-channel: output channel c depends only on input channel c. Perturb and
    # read the SAME channel (c=1); real part of output.
    eps = 1e-6
    q = f(p)
    out = q[0, 0, 1].real
    out.backward()
    grad = p.grad.clone()                    # conjugate cogradient stored by torch

    def num_deriv(real_part: bool):
        p_pos = p.detach().clone()
        p_neg = p.detach().clone()
        if real_part:
            p_pos[0, 0, 1] += eps
            p_neg[0, 0, 1] -= eps
        else:
            p_pos[0, 0, 1] += 1j * eps
            p_neg[0, 0, 1] -= 1j * eps
        return (f(p_pos)[0, 0, 1].real - f(p_neg)[0, 0, 1].real) / (2 * eps)

    d_dre = num_deriv(True)
    d_dim = num_deriv(False)
    # torch stores grad = conj(Wirtinger): real/imag components match the FD of the
    # real/imag input perturbations.
    assert abs(grad[0, 0, 1].real.item() - d_dre) < 1e-4, (grad[0, 0, 1].real.item(), d_dre)
    assert abs(grad[0, 0, 1].imag.item() - d_dim) < 1e-4, (grad[0, 0, 1].imag.item(), d_dim)
