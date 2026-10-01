"""Tests for the optional mode-coupling (channel-mixing) variant."""

import torch

from wave import WaveModel, WaveConfig, ce_loss
from wave.infer import WaveStreamer


def _cfg(**kw):
    base = dict(V=256, d=6, T=12, n_sub=6, delta=2, real_dtype="float64",
                seed=3, mix_channels=True, mix_every=2)
    base.update(kw)
    return WaveConfig(**base)


def test_mixing_grad_finite_diff_on_M():
    cfg = _cfg()
    m = WaveModel(cfg)
    assert m.M is not None and m.M.shape == (cfg.d, cfg.d)
    b = torch.randint(0, 256, (2, cfg.T))

    def loss():
        logits, _ = m(b, use_checkpoint=False)
        return ce_loss(logits, b)

    L = loss()
    m.zero_grad()
    L.backward()
    eps = 1e-6
    idx = (0, 1)
    with torch.no_grad():
        orig = m.M[idx].item()
        m.M[idx] = orig + eps
        lp = loss().item()
        m.M[idx] = orig - eps
        lm = loss().item()
        m.M[idx] = orig + 1j * eps
        lp_i = loss().item()
        m.M[idx] = orig - 1j * eps
        lm_i = loss().item()
        m.M[idx] = orig
    dre = (lp - lm) / (2 * eps)
    dim = (lp_i - lm_i) / (2 * eps)
    g = m.M.grad[idx]
    rre = abs(dre - g.real.item()) / (abs(dre) + abs(g.real.item()) + 1e-12)
    rim = abs(dim - g.imag.item()) / (abs(dim) + abs(g.imag.item()) + 1e-12)
    assert rre < 1e-5 and rim < 1e-5, (dre, dim, g, rre, rim)


def test_mixing_stream_equals_parallel():
    cfg = _cfg()
    m = WaveModel(cfg).eval()
    b = torch.randint(0, 256, (1, cfg.T))
    with torch.no_grad():
        par, _ = m(b, use_checkpoint=False)
    s = WaveStreamer(m, batch_size=1, device="cpu")
    for n in range(cfg.T):
        lg = s.step(b[:, n])
        assert torch.allclose(lg[0], par[0, n], atol=1e-9, rtol=1e-7), n


def test_lowrank_mixing_stream_equals_parallel():
    cfg = WaveConfig(V=256, d=8, T=12, n_sub=6, delta=2, real_dtype="float64",
                     seed=3, mix_channels=True, mix_every=2, mix_rank=3)
    m = WaveModel(cfg).eval()
    assert m.mix_U is not None and m.M is None
    b = torch.randint(0, 256, (1, cfg.T))
    with torch.no_grad():
        par, _ = m(b, use_checkpoint=False)
    s = WaveStreamer(m, batch_size=1, device="cpu")
    for n in range(cfg.T):
        lg = s.step(b[:, n])
        assert torch.allclose(lg[0], par[0, n], atol=1e-9, rtol=1e-7), n


def test_lowrank_mixing_grad_and_learns():
    torch.manual_seed(0)
    cfg = WaveConfig(V=256, d=16, T=24, n_sub=6, delta=2, real_dtype="float32",
                     seed=3, mix_channels=True, mix_every=2, mix_rank=4)
    m = WaveModel(cfg)
    b = torch.randint(0, 256, (4, cfg.T))
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    first = last = None
    for _ in range(40):
        logits, _ = m(b, use_checkpoint=False)
        L = ce_loss(logits, b)
        opt.zero_grad(); L.backward()
        # LoRA-style init (V=0): mix_V receives gradient from step 1; mix_U follows
        # once V becomes nonzero. The residual is exactly 0 at init (identity start).
        assert m.mix_V.grad is not None and m.mix_V.grad.abs().sum() > 0
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        first = float(L) if first is None else first
        last = float(L)
    assert last < first


def test_mixing_learns():
    torch.manual_seed(0)
    cfg = _cfg(d=16, T=24, n_sub=6, real_dtype="float32")
    m = WaveModel(cfg)
    b = torch.randint(0, 256, (4, cfg.T))
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    first = last = None
    for _ in range(40):
        logits, _ = m(b, use_checkpoint=False)
        L = ce_loss(logits, b)
        opt.zero_grad(); L.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        first = float(L) if first is None else first
        last = float(L)
    assert last < first
