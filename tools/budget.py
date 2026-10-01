"""Parameter / compute / memory budget calculator for WaveLM (master plan Part 13;
formulas reconciled with the as-built model in optimisation plan item 4).

Pure arithmetic (no torch, no GPU) -- estimates the static budget for a config so
we can pick batch/d/T/n_sub/r that fit a target VRAM before launching a run, and so
the formula tests (tests/test_wave_budget.py) can check it against the real model.

Two parameter conventions are exposed (they differ by 2x on the complex params):
  - params_complex_elems : counts each complex parameter as ONE element. This is
                           exactly what WaveModel.param_count() returns.
  - params_dof           : real degrees of freedom (complex -> x2). This is the
                           §4.1 formula  P = 2*[V*d + Δ*d + p_mix] + d + 3.

Usage: python tools/budget.py --d 384 --T 1024 --n_sub 16 --batch 96 --mix --mix_rank 64
"""

import argparse


def budget(V, d, T, n_sub, delta, batch, mix, mix_every, checkpoint_every,
           mix_rank=0, untie_steps=False, untie_embedding=False):
    """Return the static budget dict for a config.

    mix=False           : no mode coupling.
    mix=True, rank=0     : dense mode coupling M (d,d), shared across mixing points.
    mix=True, rank=r>0   : low-rank residual U,V (d,r) UNTIED per mixing point.
    Mixing is applied at P_pts = ceil(n_sub / mix_every) points (matches the model's
    `tau % mix_every == 0` for tau in 0..n_sub-1).
    """
    cmplx = 8                       # bytes per complex64
    P_pts = ((n_sub + mix_every - 1) // mix_every) if mix else 0   # CEILING (model-exact)
    r = mix_rank

    # ---- parameter counts (complex entries) ----
    p_WE = V * d * (2 if untie_embedding else 1)        # W_E [+ W_out if untied]
    p_K = (n_sub if untie_steps else 1) * delta * d     # per-step K when untied
    if not mix:
        p_mix = 0
    elif mix_rank == 0:
        p_mix = d * d                                   # dense M
    else:
        p_mix = 2 * P_pts * d * r                       # low-rank U + V, untied per point
    complex_elems = p_WE + p_K + p_mix
    real_elems = d + 3                                  # omega (d) + beta, g, T_o
    params_complex_elems = complex_elems + real_elems   # == WaveModel.param_count()
    params_dof = 2 * complex_elems + real_elems         # real DOF (== spec §4.1)
    param_bytes = complex_elems * cmplx + real_elems * 4

    # ---- forward compute (complex MACs per forward over `batch` sequences) ----
    conv = n_sub * batch * T * (d + delta * d)          # beta self-term (d) + delta taps
    if not mix:
        mix_macs = 0
    elif mix_rank == 0:
        mix_macs = P_pts * batch * T * d * d            # dense M @ p
    else:
        mix_macs = P_pts * batch * (2 * T * d * r)      # V^H p (T*d*r) + U z (T*d*r)
    readout = batch * T * V * d
    macs = conv + mix_macs + readout

    # ---- activation memory ----
    tape_full = (n_sub + 1) * batch * T * d * cmplx     # full internal-stack tape
    intermediates = 5
    chunk = checkpoint_every
    backward_peak = chunk * batch * T * d * cmplx * intermediates
    readout_act = batch * T * V * (4 + cmplx)           # logits (fp32) + complex overlap

    gb = 1024 ** 3
    return {
        "P_pts": P_pts,
        "params_complex_elems": params_complex_elems,
        "params_dof": params_dof,
        "param_MB": param_bytes / 1024 ** 2,
        "fwd_macs": macs,
        "fwd_GMACs": macs / 1e9,
        "tape_full_GB": tape_full / gb,
        "ckpt_backward_peak_GB": backward_peak / gb,
        "readout_act_GB": readout_act / gb,
        "est_peak_GB": (backward_peak + readout_act + param_bytes) / gb,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--V", type=int, default=256)
    ap.add_argument("--d", type=int, default=384)
    ap.add_argument("--T", type=int, default=1024)
    ap.add_argument("--n_sub", type=int, default=16)
    ap.add_argument("--delta", type=int, default=2)
    ap.add_argument("--batch", type=int, default=96)
    ap.add_argument("--mix", action="store_true")
    ap.add_argument("--mix_every", type=int, default=4)
    ap.add_argument("--mix_rank", type=int, default=0)
    ap.add_argument("--checkpoint_every", type=int, default=5)
    ap.add_argument("--untie_steps", action="store_true")
    ap.add_argument("--untie_embedding", action="store_true")
    a = ap.parse_args()
    b = budget(a.V, a.d, a.T, a.n_sub, a.delta, a.batch, a.mix, a.mix_every,
               a.checkpoint_every, mix_rank=a.mix_rank, untie_steps=a.untie_steps,
               untie_embedding=a.untie_embedding)
    print(f"config: d={a.d} T={a.T} n_sub={a.n_sub} delta={a.delta} batch={a.batch} "
          f"mix={a.mix}(rank={a.mix_rank}) P_pts={b['P_pts']} "
          f"untie_steps={a.untie_steps}")
    print(f"  params: {b['params_complex_elems']:,} complex-elems "
          f"(== model.param_count())  |  {b['params_dof']:,} real DOF  "
          f"({b['param_MB']:.1f} MB)")
    print(f"  forward: {b['fwd_GMACs']:.2f} GMACs/batch")
    print(f"  field tape (full): {b['tape_full_GB']:.2f} GB")
    print(f"  est. checkpointed peak: {b['est_peak_GB']:.2f} GB "
          f"(backward {b['ckpt_backward_peak_GB']:.2f} + readout {b['readout_act_GB']:.2f})")
    print(f"  {'FITS 16GB' if b['est_peak_GB'] < 15 else 'RISK OOM'}")


if __name__ == "__main__":
    main()
