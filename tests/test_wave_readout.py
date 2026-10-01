"""WaveLM readout/reduction tests (master plan Part 14).

  - test_readout_normalisation : sum_o p(o|n) = 1
  - test_reduce_to_bigram      : K=0, n_sub=1, g->0 reduces to squared-bilinear bigram
"""

import math

import torch
import torch.nn.functional as F

from wave import WaveModel, WaveConfig
from wave.readout import born_logits


def test_readout_normalisation():
    cfg = WaveConfig(V=256, d=8, T=16, n_sub=4, delta=2, real_dtype="float64")
    m = WaveModel(cfg).eval()
    b = torch.randint(0, 256, (3, cfg.T))
    with torch.no_grad():
        logits, _ = m(b, use_checkpoint=False)
        p = F.softmax(logits, dim=-1)
    s = p.sum(-1)
    assert torch.allclose(s, torch.ones_like(s), atol=1e-9), s.min().item()


def test_reduce_to_bigram():
    """With no propagation (K=0), one step, and g->0, the model logit is exactly
    the closed-form squared-bilinear bigram l(o|n)=beta^2|<W_E[o],e^{iw}W_E[o_n]>|^2/T_o,
    a function of o_n only. (Master plan Part 12.)
    """
    cfg = WaveConfig(V=256, d=6, T=8, n_sub=1, delta=2, real_dtype="float64")
    m = WaveModel(cfg).eval()
    with torch.no_grad():
        m.K.zero_()                         # drop the convolution sum
        m.theta_g.fill_(-60.0)              # g = softplus(-60) ~ 0  -> q = beta*psi
    b = torch.randint(0, 256, (2, cfg.T))
    with torch.no_grad():
        logits, _ = m(b, use_checkpoint=False)

        # closed form: with K=0, g->0, n_sub=1, the step is
        #   psi[n] = normalize( exp(i w) * beta * W_E[o_n] )
        # (the per-position unit-norm projection cancels the beta scalar and |W_E|).
        omega = m.omega()
        T_o = m.T_o()
        rot = torch.exp(1j * omega).to(m.W_E.dtype)        # (d,)
        r = rot.view(1, 1, -1) * m.W_E[b]
        norm = (r.real**2 + r.imag**2).sum(-1, keepdim=True).sqrt()
        psi_closed = r * math.sqrt(cfg.d) / (norm + 1e-8)
        closed = born_logits(psi_closed, m.W_E, T_o)

    assert torch.allclose(logits, closed, atol=1e-8, rtol=1e-6), (
        (logits - closed).abs().max().item()
    )
    # and it depends only on o_n: two positions with the same byte get equal logits
    # (construct a sequence with a repeated byte)
    b2 = torch.zeros(1, cfg.T, dtype=torch.long)
    b2[0, 0] = 5
    b2[0, 3] = 5
    with torch.no_grad():
        lg, _ = m(b2, use_checkpoint=False)
    assert torch.allclose(lg[0, 0], lg[0, 3], atol=1e-8)
