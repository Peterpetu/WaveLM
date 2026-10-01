"""WaveLM expressivity go/no-go (master plan Part 12, Phase D).

The decisive test of the whole design. Task: modular addition
    c = (a + b) mod K ,   with a, b ~ Uniform{0..K-1} drawn INDEPENDENTLY per example.
Each example is the 3-symbol sequence [a, b, c]; we score ONLY the prediction of
c from the field at position 1 (which has seen a and b). Because every example is
i.i.d., there is no global sequence to memorise — the only way to predict c is to
*compose* the two observed symbols. Modular addition is linearly inseparable (it
needs a product/Fourier feature of a and b), yet is natural for WaveLM's complex
phase carriers + |.|^2 readout.

Claim (Part 12):
  - Full WaveLM drives CE far below the bigram floor H(c|b)=ln K (it composes).
  - The linear-in-embeddings ablation (g=0 + linear readout) cannot beat the
    bigram floor (it cannot form the product feature).

If WaveLM cannot beat the bigram ceiling here, the design is wrong: STOP before
any TinyStories run.
"""

import math

import torch

from wave import WaveModel, WaveConfig, ce_loss


K_SYM = 5
BIGRAM_FLOOR = math.log(K_SYM)   # H(c|b) = H(a) = ln K, since a ⟂ b


def _make_triples(n: int, gen: torch.Generator) -> torch.Tensor:
    a = torch.randint(0, K_SYM, (n,), generator=gen)
    b = torch.randint(0, K_SYM, (n,), generator=gen)
    c = (a + b) % K_SYM
    return torch.stack([a, b, c], dim=1)        # (n, 3)


def _ce_on_c(model: WaveModel, x: torch.Tensor, linear: bool) -> torch.Tensor:
    """CE of predicting c (position 2) from the field at position 1."""
    logits, _ = model(x, use_checkpoint=False, linear=linear)   # (n, 3, V)
    pred = logits[:, 1, :]                                       # predict x[:,2]
    return torch.nn.functional.cross_entropy(pred, x[:, 2])


def _train(model, steps, lr, linear, gen):
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    for _ in range(steps):
        x = _make_triples(256, gen)
        L = _ce_on_c(model, x, linear)
        opt.zero_grad()
        L.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    return float(L)


def _eval(model, linear, gen):
    model.eval()
    with torch.no_grad():
        x = _make_triples(2048, gen)
        return float(_ce_on_c(model, x, linear))


def test_wavelm_fits_trigram_and_linear_cannot():
    gen = torch.Generator().manual_seed(1)
    cfg = WaveConfig(V=256, d=48, T=3, n_sub=8, delta=2, real_dtype="float32", seed=2)

    full = WaveModel(cfg)
    _train(full, steps=600, lr=5e-3, linear=False, gen=gen)
    ce_full = _eval(full, linear=False, gen=gen)

    lin = WaveModel(cfg)
    _train(lin, steps=600, lr=5e-3, linear=True, gen=gen)
    ce_lin = _eval(lin, linear=True, gen=gen)

    print(f"\n[expressivity] bigram_floor={BIGRAM_FLOOR:.3f} "
          f"CE_full={ce_full:.3f} CE_linear={ce_lin:.3f}")

    # Full WaveLM composes -> CE well below the bigram floor.
    assert ce_full < 0.5 * BIGRAM_FLOOR, (
        f"WaveLM did not beat bigram floor: {ce_full:.3f} vs {BIGRAM_FLOOR:.3f}")
    # Linear ablation cannot form the product feature -> stuck at/above floor.
    assert ce_lin > 0.85 * BIGRAM_FLOOR, (
        f"linear ablation unexpectedly composed: {ce_lin:.3f} vs {BIGRAM_FLOOR:.3f}")
