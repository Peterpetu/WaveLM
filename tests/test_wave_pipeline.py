"""WaveLM pipeline tests (master plan Part 14, system group + Part 9 guard)."""

import os
import tempfile

import torch

from wave import WaveModel, WaveConfig
from wave.data import ByteData, decode_bytes, encode_str
from wave.infer import WaveStreamer, generate
from wave.train import SpectralGuard, spectral_repulsion, lr_cosine


def test_stream_equals_parallel():
    """Streaming logits are bit-identical to the parallel forward (FP64)."""
    cfg = WaveConfig(V=256, d=8, T=20, n_sub=6, delta=2, real_dtype="float64", seed=4)
    m = WaveModel(cfg).eval()
    b = torch.randint(0, 256, (1, cfg.T))
    with torch.no_grad():
        par, _ = m(b, use_checkpoint=False)            # (1, T, V)
    s = WaveStreamer(m, batch_size=1, device="cpu")
    for n in range(cfg.T):
        lg = s.step(b[:, n])                            # predicts position n+1
        # compare to parallel logits at position n
        assert torch.allclose(lg[0], par[0, n], atol=1e-9, rtol=1e-7), (
            n, (lg[0] - par[0, n]).abs().max().item())


def test_byte_packing_roundtrip():
    text = "Once upon a time, a little dog ran. \xe9é done.\n" * 50
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "t.txt")
        with open(path, "wb") as f:
            f.write(text.encode("utf-8"))
        with ByteData(path, T=16) as data:
            batch = data.random_batch(4, torch.Generator().manual_seed(0))
            assert batch.shape == (4, 16)
            assert batch.min() >= 0 and batch.max() < 256
            raw = open(path, "rb").read()
            # a sequential window must equal the raw bytes
            first = next(data.sequential_batches(1, max_batches=1))
            assert bytes(first[0].tolist()) == raw[:16]
    # encode/decode roundtrip
    assert decode_bytes(encode_str("hello é")) == "hello é"


def test_spectral_diversity_only_when_collapsing():
    """lambda_omega stays 0 while Var(omega) is healthy; activates only after the
    variance collapses for `patience` validations; releases on recovery."""
    g = SpectralGuard(var0=1.0, lambda0=0.01, rho=0.25, patience=2, release_frac=0.5)
    assert g.update(0.9) == 0.0          # healthy
    assert g.update(0.2) == 0.0          # 1st below (need 2)
    assert g.update(0.2) == 0.01         # 2nd consecutive below -> activate
    assert g.active
    assert g.update(0.3) == 0.01         # still below release -> stays on
    assert g.update(0.6) == 0.0          # recovered above release -> off
    assert not g.active
    # repulsion is more negative when spread, ~0 when collapsed
    spread = spectral_repulsion(torch.linspace(-3.0, 3.0, 32))
    collapsed = spectral_repulsion(torch.full((32,), 0.5))
    assert spread < -0.5 and abs(collapsed) < 1e-6


def test_lr_schedule_shape():
    total, peak, floor = 1000, 3e-4, 3e-5
    assert lr_cosine(0, total, 0.05, peak, floor) == 0.0
    assert abs(lr_cosine(50, total, 0.05, peak, floor) - peak) < 1e-9   # end of warmup
    assert lr_cosine(1000, total, 0.05, peak, floor) <= floor + 1e-9    # end


def test_generation_runs_and_varies():
    """At random init the model is near-uniform, so sampling must produce variety
    (no immediate collapse to a single byte). The trained-model non-degeneracy
    check is performed in the Phase F analysis."""
    cfg = WaveConfig(V=256, d=16, T=64, n_sub=4, delta=2, real_dtype="float32", seed=9)
    m = WaveModel(cfg)
    out = generate(m, prompt=encode_str("Once "), max_new=40, temperature=1.0,
                   device="cpu", seed=0)
    new = out[5:]
    uniq = len(set(new)) / len(new)
    assert uniq > 0.5, f"degenerate generation, unique ratio {uniq}"
