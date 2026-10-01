"""Correctness gates for the byte-GPT comparison baseline.

A buggy baseline (too weak, non-causal, or mis-initialised) would invalidate the
WaveLM-vs-Transformer comparison, so we lock these before any training run.

  - test_byte_gpt_causality        : future bytes never affect earlier logits
  - test_byte_gpt_init_bpb_uniform  : random init ~ uniform (bpb ~ 8.0), not degenerate
  - test_byte_gpt_overfits_batch    : it can drive CE -> ~0 on a tiny batch (it learns)
  - test_byte_gpt_weight_tying       : output head shares the embedding (param accounting)
"""

import math

import torch

from baselines.byte_gpt import GPTConfig, ByteGPT
from wave.readout import ce_loss, bits_per_byte


def test_byte_gpt_causality():
    """Changing a future byte must leave earlier-position logits bit-identical."""
    torch.manual_seed(0)
    m = ByteGPT(GPTConfig(d_model=32, n_layer=2, n_head=2, T=16)).eval()
    b = torch.randint(0, 256, (1, 16))
    with torch.no_grad():
        lg0 = m(b)
        b2 = b.clone()
        b2[0, -1] = (b2[0, -1] + 7) % 256          # perturb only the LAST byte
        lg1 = m(b2)
    # positions 0..T-2 must be unchanged (they cannot see the last byte)
    assert torch.allclose(lg0[:, :-1], lg1[:, :-1], atol=1e-6, rtol=0), \
        (lg0[:, :-1] - lg1[:, :-1]).abs().max().item()
    # the last position SHOULD change (sanity that the perturbation matters)
    assert not torch.allclose(lg0[:, -1], lg1[:, -1], atol=1e-6)


def test_byte_gpt_init_bpb_uniform():
    """At random init the model is ~uniform over 256 bytes -> bpb ~ 8.0 (= log2 256)."""
    torch.manual_seed(0)
    m = ByteGPT(GPTConfig(d_model=64, n_layer=2, n_head=2, T=64)).eval()
    b = torch.randint(0, 256, (4, 64))
    with torch.no_grad():
        bpb = bits_per_byte(ce_loss(m(b), b))
    assert abs(bpb - 8.0) < 0.6, f"init bpb {bpb} not near uniform 8.0"


def test_byte_gpt_overfits_batch():
    """A small GPT must memorise a tiny fixed batch (CE -> ~0): proves it learns."""
    torch.manual_seed(0)
    m = ByteGPT(GPTConfig(d_model=64, n_layer=2, n_head=2, T=16)).train()
    b = torch.randint(0, 256, (4, 16))
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=0.0)
    first = None
    for step in range(400):
        opt.zero_grad()
        loss = ce_loss(m(b), b)
        loss.backward()
        opt.step()
        if first is None:
            first = float(loss)
    final = float(loss)
    assert final < 0.2, f"GPT failed to overfit a tiny batch: {first:.3f} -> {final:.3f}"


def test_byte_gpt_weight_tying_and_param_count():
    """Output head is tied to the embedding; param_count counts shared weights once."""
    cfg = GPTConfig(d_model=128, n_layer=3, n_head=4, T=128)
    m = ByteGPT(cfg)
    assert m.head.weight is m.tok.weight, "output head must be weight-tied to embedding"
    # rough hand estimate: embed V*d + per-layer (qkv 3d^2 + proj d^2 + mlp 2*d*d_ff)
    d, L, dff = cfg.d_model, cfg.n_layer, cfg.d_ff
    approx = cfg.V * d + L * (3 * d * d + d * d + 2 * d * dff)
    pc = m.param_count()
    assert 0.9 * approx < pc < 1.15 * approx, (pc, approx)
