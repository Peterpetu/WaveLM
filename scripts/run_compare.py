"""E1 + E2: WaveLM vs byte-GPT, matched-everything comparison (autonomous driver).

Runs four phases, writing incremental JSON after EVERY run so partial results
survive a crash, and wrapping each run in try/except (+ an OOM retry that turns on
WaveLM activation-checkpointing) so one failure never aborts the night:

  A. lr sweep at the smallest size (pick best base lr per architecture).
  B. matched-token ladder (4 sizes x 2 archs, best lr, seed 0)  -> E2 scaling law
                                                                 -> E1 bpb-vs-params.
  C. seeds (2 sizes x 2 archs x 2 extra seeds)                  -> error bars.
  D. matched-wall-clock sprint (2 sizes x 2 archs, fixed seconds)-> E1 bpb-vs-wall-clock.

Then fits bpb ~ E_inf + A*N^-alpha per architecture and prints the comparison.

Fairness: identical data/metric/optimiser/schedule/precision (see baselines/harness.py);
params matched on real DOF (complex WaveLM param = 2). muP gives WaveLM lr-transfer;
the GPT baseline is given its own best lr from the same sweep (a strong, fair baseline).
"""

import argparse
import json
import math
import os
import sys
import time

# path shim: make `import wave.*` / `import baselines.*` resolve to the repo, not scripts/
_here = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != _here]
sys.path.insert(0, os.path.dirname(_here))

import torch
import numpy as np

from wave.data import ByteData
from baselines.harness import build_wavelm, build_gpt, train_bytes_model


def fit_powerlaw(Ns, bpbs):
    """Fit bpb ~ E_inf + A * N^(-alpha) by grid-searching E_inf + log-log LSQ."""
    N = np.asarray(Ns, float)
    y = np.asarray(bpbs, float)
    if len(N) < 3:
        slope, intercept = np.polyfit(np.log(N), np.log(y), 1)
        return {"E_inf": 0.0, "A": float(np.exp(intercept)), "alpha": float(-slope),
                "sse": 0.0, "note": "<3 points: E_inf=0"}
    best = None
    for E in np.linspace(0.0, float(y.min()) * 0.999, 400):
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
    ap.add_argument("--out", default="checkpoints/compare/compare_results.json")
    ap.add_argument("--T", type=int, default=512)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--ladder_steps", type=int, default=2500)
    ap.add_argument("--sweep_steps", type=int, default=1200)
    ap.add_argument("--wall_seconds", type=float, default=480.0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    TARGETS = [500_000, 1_200_000, 3_000_000, 7_000_000]
    WAVELM_LRS = [1e-4, 3e-4, 1e-3]
    GPT_LRS = [3e-4, 1e-3, 3e-3]
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    tr = ByteData(args.train_txt, args.T)
    va = ByteData(args.val_txt, args.T)

    results = {"meta": {"T": args.T, "batch": args.batch, "ladder_steps": args.ladder_steps,
                        "sweep_steps": args.sweep_steps, "wall_seconds": args.wall_seconds,
                        "targets": TARGETS, "device": args.device,
                        "started": time.strftime("%Y-%m-%d %H:%M:%S")},
               "sweep": [], "ladder": [], "seeds": [], "wallclock": [], "fits": {}}

    def save():
        json.dump(results, open(args.out, "w"), indent=2)

    def run_one(arch, target, lr, steps, seed, max_seconds=None, label=""):
        """Build + train one model. Robust: OOM -> retry with WaveLM checkpointing;
        other errors -> recorded and skipped."""
        t0 = time.perf_counter()
        for attempt, ckpt in enumerate([False, True]):
            model = None
            try:
                torch.cuda.empty_cache()
                if arch == "wavelm":
                    model = build_wavelm(target, T=args.T, seed=seed)
                else:
                    model = build_gpt(target, T=args.T, seed=seed)
                r = train_bytes_model(
                    model, tr, va, steps=steps, batch=args.batch, T=args.T, lr=lr,
                    device=args.device, seed=seed, val_every=250, val_batches=40,
                    wave_checkpoint=ckpt, max_seconds=max_seconds, label=label)
                r["oom_retry"] = (attempt == 1)
                del model
                torch.cuda.empty_cache()
                print(f"  [{label}] arch={arch} dof={r['params_dof']:,} lr={lr:g} "
                      f"seed={seed} -> bpb_best={r['val_bpb_best']} bpb_final={r['val_bpb_final']} "
                      f"tok/s={r['tok_per_s']:,.0f} peak={r['peak_gb']}G "
                      f"steps={r['steps_done']} wall={r['wall_s']}s", flush=True)
                return r
            except RuntimeError as e:
                if model is not None:
                    del model
                torch.cuda.empty_cache()
                if "out of memory" in str(e).lower() and arch == "wavelm" and attempt == 0:
                    print(f"  [{label}] OOM at batch={args.batch}; retrying with checkpointing", flush=True)
                    continue
                print(f"  [{label}] FAILED: {type(e).__name__}: {str(e)[:160]}", flush=True)
                return {"label": label, "arch": arch, "target": target, "lr": lr,
                        "seed": seed, "error": str(e)[:200]}
        return {"label": label, "arch": arch, "target": target, "error": "oom_after_retry"}

    print("=" * 78, flush=True)
    print(f"E1+E2 compare | T={args.T} batch={args.batch} ladder_steps={args.ladder_steps} "
          f"(~{args.ladder_steps*args.batch*args.T/1e6:.0f}M bytes/run) targets={TARGETS}", flush=True)
    print("=" * 78, flush=True)

    # ---- Phase A: lr sweep at the smallest size ----
    print("\n### PHASE A: lr sweep (smallest size) ###", flush=True)
    for arch, lrs in [("wavelm", WAVELM_LRS), ("gpt", GPT_LRS)]:
        for lr in lrs:
            r = run_one(arch, TARGETS[0], lr, args.sweep_steps, seed=0,
                        label=f"sweep_{arch}_lr{lr:g}")
            results["sweep"].append(r); save()
    best_lr = {}
    for arch in ("wavelm", "gpt"):
        cand = [r for r in results["sweep"] if r["arch"] == arch and r.get("val_bpb_best")]
        best_lr[arch] = min(cand, key=lambda r: r["val_bpb_best"])["lr"] if cand else \
            (3e-4 if arch == "wavelm" else 1e-3)
    results["meta"]["best_lr"] = best_lr
    save()
    print(f"  best lr: {best_lr}", flush=True)

    # ---- Phase B: matched-token ladder (E1 + E2) ----
    print("\n### PHASE B: matched-token ladder ###", flush=True)
    for target in TARGETS:
        for arch in ("wavelm", "gpt"):
            r = run_one(arch, target, best_lr[arch], args.ladder_steps, seed=0,
                        label=f"ladder_{arch}_{target}")
            results["ladder"].append(r); save()

    # ---- Phase C: seeds (error bars) at 2 sizes ----
    print("\n### PHASE C: seeds ###", flush=True)
    for target in (TARGETS[0], TARGETS[2]):
        for arch in ("wavelm", "gpt"):
            for seed in (1, 2):
                r = run_one(arch, target, best_lr[arch], args.ladder_steps, seed=seed,
                            label=f"seed_{arch}_{target}_s{seed}")
                results["seeds"].append(r); save()

    # ---- Phase D: matched wall-clock sprint ----
    print("\n### PHASE D: matched wall-clock ###", flush=True)
    for target in (TARGETS[1], TARGETS[2]):
        for arch in ("wavelm", "gpt"):
            r = run_one(arch, target, best_lr[arch], 100000, seed=0,
                        max_seconds=args.wall_seconds, label=f"wall_{arch}_{target}")
            results["wallclock"].append(r); save()

    # ---- Fit scaling laws from the ladder ----
    for arch in ("wavelm", "gpt"):
        pts = [(r["params_dof"], r["val_bpb_best"]) for r in results["ladder"]
               if r.get("arch") == arch and r.get("val_bpb_best")]
        pts.sort()
        if len(pts) >= 3:
            results["fits"][arch] = {"points": pts,
                                     "fit": fit_powerlaw([p for p, _ in pts],
                                                         [b for _, b in pts])}
    results["meta"]["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    save()
    tr.close(); va.close()

    # ---- Print comparison ----
    print("\n" + "=" * 78, flush=True)
    print("LADDER (matched tokens, seed 0): bpb_best vs params", flush=True)
    for arch in ("wavelm", "gpt"):
        row = [(r["params_dof"], r["val_bpb_best"], r["tok_per_s"], r["wall_s"])
               for r in results["ladder"] if r.get("arch") == arch and r.get("val_bpb_best")]
        row.sort()
        print(f"  {arch}: " + "  ".join(f"{p/1e6:.2f}M->{b:.3f}bpb({t/1e3:.0f}k tok/s)"
                                        for p, b, t, w in row), flush=True)
    for arch in ("wavelm", "gpt"):
        f = results["fits"].get(arch, {}).get("fit")
        if f:
            print(f"  FIT {arch}: bpb ~ {f['E_inf']:.3f} + {f['A']:.3g}*N^(-{f['alpha']:.3f})", flush=True)
    print(f"\nwrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
