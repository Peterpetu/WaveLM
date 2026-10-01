"""muP width-scaled learning-rate tests (optimisation plan item 3).

  - test_mup_multipliers              : per-role lr_mult equals the 1/fan-in table
  - test_mup_forward_scale_invariant   : init logit variance ~ width-invariant
  - test_mup_update_scale_invariant    : 1-step |Δlogit| ~ width-invariant (the muP property)
"""

import math

import torch

from wave import WaveModel, WaveConfig, ce_loss

WIDTHS = [256, 512, 1024, 2048]


def test_mup_multipliers():
    """lr_mult equals 1/fan-in relative to (d0,r0), for tied and untied embedding."""
    d0, r0 = 256, 32
    for d, r in [(256, 32), (512, 64), (1024, 32), (384, 48)]:
        cfg = WaveConfig(V=256, d=d, T=16, n_sub=4, delta=2,
                         mix_channels=True, mix_rank=r, mix_every=2,
                         mup=True, mup_d0=d0, mup_r0=r0, real_dtype="float64", seed=0)
        g = {grp["name"]: grp for grp in WaveModel(cfg).mup_param_groups(1.0, 0.01)}
        assert abs(g["W_E"]["lr_mult"] - math.sqrt(d0 / d)) < 1e-12
        assert abs(g["K"]["lr_mult"] - 1.0) < 1e-12
        assert "M" not in g                                # low-rank uses mix_U/mix_V
        assert abs(g["mix_U"]["lr_mult"] - r0 / r) < 1e-12
        assert abs(g["mix_V"]["lr_mult"] - d0 / d) < 1e-12
        assert abs(g["scalars"]["lr_mult"] - 1.0) < 1e-12
        assert g["scalars"]["weight_decay"] == 0.0
        assert g["W_E"]["weight_decay"] == 0.01

    # dense mixing (mix_rank=0) -> an M group with fan-in d
    cfg = WaveConfig(V=256, d=512, T=16, n_sub=4, delta=2, mix_channels=True,
                     mix_rank=0, mup=True, real_dtype="float64", seed=0)
    g = {grp["name"]: grp for grp in WaveModel(cfg).mup_param_groups(1.0, 0.01)}
    assert abs(g["M"]["lr_mult"] - 256 / 512) < 1e-12

    # untied embedding -> exact muP: W_in fan-in 1, W_out fan-in d
    cfg = WaveConfig(V=256, d=512, T=16, n_sub=4, delta=2, mup=True,
                     untie_embedding=True, real_dtype="float64", seed=0)
    g = {grp["name"]: grp for grp in WaveModel(cfg).mup_param_groups(1.0, 0.01)}
    assert abs(g["W_in"]["lr_mult"] - 1.0) < 1e-12
    assert abs(g["W_out"]["lr_mult"] - 256 / 512) < 1e-12


def test_mup_forward_scale_invariant():
    """At init the Born |<W_E,psi>|^2 logits have ~width-invariant scale (E|W_E|^2=1/d
    and ||psi[n]||=sqrt(d) give an O(1) overlap for all d). This is the assumption the
    muP lr rule relies on."""
    variances = {}
    for d in WIDTHS:
        cfg = WaveConfig(V=256, d=d, T=16, n_sub=4, delta=2, mup=True,
                         real_dtype="float32", seed=0)
        m = WaveModel(cfg).eval()
        b = torch.randint(0, 256, (4, cfg.T), generator=torch.Generator().manual_seed(0))
        with torch.no_grad():
            logits, _ = m(b, use_checkpoint=False)
        variances[d] = float(logits.var())
    vals = list(variances.values())
    ratio = max(vals) / min(vals)
    assert ratio < 2.0, f"logit variance not width-invariant: {variances} ratio={ratio:.2f}"


def test_mup_update_scale_invariant():
    """After one Adam step at the muP lr, the mean |Δlogit| is ~width-invariant --
    the property the per-group 1/fan-in rule is designed to produce (vs a single
    global lr, whose |Δlogit| grows with width)."""
    deltas = {}
    eta0 = 1e-2
    for d in WIDTHS:
        cfg = WaveConfig(V=256, d=d, T=16, n_sub=4, delta=2, mup=True,
                         real_dtype="float32", seed=0)
        m = WaveModel(cfg)
        b = torch.randint(0, 256, (4, cfg.T), generator=torch.Generator().manual_seed(1))
        with torch.no_grad():
            logits0, _ = m(b, use_checkpoint=False)
        opt = torch.optim.AdamW(m.mup_param_groups(eta0, 0.0),
                                betas=(0.9, 0.95), eps=1e-8)
        logits, _ = m(b, use_checkpoint=False)
        ce_loss(logits, b).backward()
        opt.step()
        with torch.no_grad():
            logits1, _ = m(b, use_checkpoint=False)
        deltas[d] = float((logits1 - logits0).abs().mean())
    vals = list(deltas.values())
    ratio = max(vals) / min(vals)
    assert ratio < 3.0, f"update scale not width-invariant: {deltas} ratio={ratio:.2f}"


def test_untie_embedding_stream_equals_parallel():
    """untie_embedding routes the Born readout through W_out; the streamer must stay
    bit-identical to the parallel forward (the readout-basis change is the only diff)."""
    from wave.infer import WaveStreamer
    cfg = WaveConfig(V=256, d=8, T=20, n_sub=6, delta=2, untie_embedding=True,
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
