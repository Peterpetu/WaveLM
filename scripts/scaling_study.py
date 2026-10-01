"""WaveLM capacity scaling study (optimisation plan item 4).

Trains a ladder of models that differ along ONE capacity axis (width d, mode-coupling
rank r, or internal steps n_sub) to a MATCHED token budget, records (params, FLOPs,
val bpb), fits a Chinchilla-style law  bpb(N) ~ E_inf + A * N^(-alpha), and reports
the per-axis exponent and the bpb reduction per e-fold of FLOPs. Run it once per axis
and scale the axis with the best bpb-per-compute (subject to the 16 GB budget).

Use muP (default) so the SAME base lr is valid at every rung (no per-size lr search).

Example:
  python scripts/scaling_study.py --axis d --ladder 384,512,768 --tokens 5_000_000 \
      --batch 24 --T 1024 --n_sub 16 --mix_rank 64 --mix_every 2 --lr 3e-4
"""

import argparse
import json
import math
import os
import sys

# Same path-shadow fix as train_wave.py: keep scripts/ off sys.path (legacy
# scripts/profile.py shadows the stdlib `profile` and breaks torch._dynamo).
_here = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != _here]
sys.path.insert(0, os.path.dirname(_here))

import torch

from wave.config import WaveConfig
from wave.model import WaveModel
from wave.data import ByteData
from wave.train import train_loop, validate

sys.path.insert(0, os.path.join(os.path.dirname(_here), "tools"))
from budget import budget  # noqa: E402


def fit_powerlaw(Ns, bpbs):
    """Fit bpb ~ E_inf + A * N^(-alpha) by grid-searching E_inf and log-log least
    squares for (A, alpha). Returns dict(E_inf, A, alpha, sse). Needs >=3 rungs to
    identify E_inf; with 2 it fixes E_inf=0 (pure power law)."""
    import numpy as np
    N = np.asarray(Ns, dtype=float)
    y = np.asarray(bpbs, dtype=float)
    if len(N) < 3:
        slope, intercept = np.polyfit(np.log(N), np.log(y), 1)
        return {"E_inf": 0.0, "A": float(np.exp(intercept)), "alpha": float(-slope),
                "sse": 0.0, "note": "<3 rungs: E_inf fixed to 0"}
    best = None
    for E in np.linspace(0.0, float(y.min()) * 0.999, 300):
        resid = y - E
        if (resid <= 0).any():
            continue
        slope, intercept = np.polyfit(np.log(N), np.log(resid), 1)
        pred = E + np.exp(intercept) * N ** slope
        sse = float(((pred - y) ** 2).sum())
        if best is None or sse < best["sse"]:
            best = {"E_inf": float(E), "A": float(np.exp(intercept)),
                    "alpha": float(-slope), "sse": sse}
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_txt", default="data/tinystories/train.txt")
    ap.add_argument("--val_txt", default="data/tinystories/val.txt")
    ap.add_argument("--axis", choices=["d", "mix_rank", "n_sub"], default="d")
    ap.add_argument("--ladder", default="384,512,768", help="comma list of axis values")
    ap.add_argument("--tokens", type=int, default=5_000_000, help="matched token budget per rung")
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--T", type=int, default=1024)
    ap.add_argument("--d", type=int, default=384)
    ap.add_argument("--n_sub", type=int, default=16)
    ap.add_argument("--delta", type=int, default=2)
    ap.add_argument("--mix_rank", type=int, default=64)
    ap.add_argument("--mix_every", type=int, default=2)
    ap.add_argument("--lr", type=float, default=3e-4, help="base lr eta0 (muP-transferred)")
    ap.add_argument("--mup", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--val_batches", type=int, default=50)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="checkpoints/scaling_study.json")
    args = ap.parse_args()

    values = [int(v) for v in args.ladder.split(",")]
    steps = max(1, args.tokens // (args.batch * args.T))
    print(f"Scaling study: axis={args.axis} ladder={values} | matched tokens={args.tokens:,} "
          f"-> {steps} steps/rung (batch={args.batch} T={args.T}) | muP={args.mup} lr={args.lr}")

    rungs = []
    for v in values:
        d = v if args.axis == "d" else args.d
        mix_rank = v if args.axis == "mix_rank" else args.mix_rank
        n_sub = v if args.axis == "n_sub" else args.n_sub
        mix_channels = mix_rank > 0
        cfg = WaveConfig(
            V=256, d=d, T=args.T, n_sub=n_sub, delta=args.delta,
            batch_size=args.batch, total_steps=steps, lr_peak=args.lr,
            mix_channels=mix_channels, mix_every=args.mix_every, mix_rank=mix_rank,
            mup=args.mup, seed=args.seed, real_dtype="float32",
        )
        model = WaveModel(cfg)
        N = model.param_count()
        flops = budget(256, d, args.T, n_sub, args.delta, 1, mix_channels,
                       args.mix_every, cfg.checkpoint_every, mix_rank=mix_rank)["fwd_macs"]
        print("=" * 78)
        print(f"RUNG {args.axis}={v}: d={d} n_sub={n_sub} rank={mix_rank} | "
              f"N={N:,} complex-params | {flops/1e9:.2f} GMACs/seq")
        train = ByteData(args.train_txt, cfg.T)
        val = ByteData(args.val_txt, cfg.T)
        train_loop(model, train, cfg, val=val, device=args.device,
                   model_name=f"WaveLM[{args.axis}={v}]", log_every=200,
                   val_every=max(200, steps // 4), ckpt_every=10 ** 9,
                   ckpt_dir=None, metrics_path=None)
        vbpb = validate(model, val, cfg, args.device, max_batches=args.val_batches)
        train.close()
        val.close()
        print(f"RUNG {args.axis}={v}: final val_bpb={vbpb:.4f}")
        rungs.append({"axis_value": v, "d": d, "n_sub": n_sub, "mix_rank": mix_rank,
                      "params": N, "flops": flops, "val_bpb": vbpb})

    Ns = [r["params"] for r in rungs]
    Fs = [r["flops"] for r in rungs]
    bpbs = [r["val_bpb"] for r in rungs]
    fit = fit_powerlaw(Ns, bpbs)
    # bpb reduction per e-fold of FLOPs (best rung vs smallest)
    eff = (bpbs[0] - bpbs[-1]) / max(1e-9, math.log(Fs[-1]) - math.log(Fs[0]))

    print("=" * 78)
    print(f"FIT  bpb ~ E_inf + A*N^(-alpha):  E_inf={fit['E_inf']:.4f}  A={fit['A']:.4g}  "
          f"alpha={fit['alpha']:.4f}  (sse={fit.get('sse', 0):.2e}){'  ' + fit.get('note','')}")
    print(f"EFFICIENCY  d(bpb)/d(log FLOPs) over the ladder = {eff:.4f} bpb per e-fold")
    if fit["alpha"] > 0.02:
        print(f"RECOMMEND scaling axis '{args.axis}': alpha>0 (capacity helps). "
              f"Pick the largest rung that fits 16 GB via tools/budget.py.")
    else:
        print(f"RECOMMEND NOT scaling axis '{args.axis}' on this data: alpha~0 "
              f"(data-limited, not capacity-limited) -- scale data/steps instead.")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"axis": args.axis, "tokens": args.tokens, "steps": steps,
                   "mup": args.mup, "lr": args.lr, "rungs": rungs, "fit": fit,
                   "efficiency_bpb_per_efold_flops": eff}, f, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
