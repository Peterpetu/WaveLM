"""Analyse the driver-attribution campaign (doc 12): turn ablation_results.json into
the evidence tables the analysis needs. Pure reporting -- no training.

For each scale d it prints:
  - the per-variant best-lr val bpb, mean +/- std over seeds, and delta vs REF;
  - the decisive decompositions:
      * phase            : REF vs ablate_phase            (is the carrier load-bearing?)
      * nonlinearity 2x2 : REF / ablate_saturation / ablate_born / ablate_both
                           (which of {|p|^2 saturation, Born square} lifts past linear?)
      * complex vs real  : REF vs real_twin (matched DOF) (is complex/phase the driver?)
      * hidden lever     : selective vs selective_control (does data-dependence help
                           beyond just adding the same parameters?)
  - a seed-noise estimate (max std over the multi-seed variants) so deltas can be
    called real or not.

Usage:  python scripts/analyze_ablation.py [--results checkpoints/ablation/ablation_results.json]
"""

import argparse
import json
import math
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != _here]
sys.path.insert(0, os.path.dirname(_here))


def _mean_std(xs):
    n = len(xs)
    m = sum(xs) / n
    if n < 2:
        return m, 0.0
    v = sum((x - m) ** 2 for x in xs) / (n - 1)
    return m, math.sqrt(v)


def collect(runs):
    """-> {d: {variant: {lr: [bpb,...]}}} and {(d,variant,lr): dof}."""
    by = {}
    dof = {}
    for r in runs:
        if r.get("val_bpb_best") is None:
            continue
        d, v, lr = r["d"], r["variant"], r["lr"]
        by.setdefault(d, {}).setdefault(v, {}).setdefault(lr, []).append(r["val_bpb_best"])
        dof[(d, v)] = r.get("params_dof", dof.get((d, v)))
    return by, dof


def best_lr_stats(by_lr):
    """Pick the lr with the lowest mean bpb; return (mean, std, lr, n)."""
    best = None
    for lr, bpbs in by_lr.items():
        m, s = _mean_std(bpbs)
        if best is None or m < best[0]:
            best = (m, s, lr, len(bpbs))
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="checkpoints/ablation/ablation_results.json")
    args = ap.parse_args()
    data = json.load(open(args.results))
    runs = data.get("runs", [])
    by, dof = collect(runs)

    n_done = sum(1 for r in runs if r.get("val_bpb_best") is not None)
    n_err = sum(1 for r in runs if r.get("error"))
    print(f"runs: {n_done} done, {n_err} errored, {len(runs)} recorded "
          f"(meta: {data.get('meta', {}).get('steps')} steps, "
          f"{data.get('meta', {}).get('bytes_per_run', 0)/1e6:.0f}M bytes/run)")
    if n_err:
        for r in runs:
            if r.get("error"):
                print(f"   ERROR {r.get('label')}: {r['error'][:120]}")

    for d in sorted(by):
        stats = {v: best_lr_stats(by_lr) for v, by_lr in by[d].items()}
        ref = stats.get("REF")
        ref_bpb = ref[0] if ref else None
        # seed noise: max std among variants with >=2 seeds at their best lr
        noise = max((s for (_, s, _, n) in stats.values() if n >= 2), default=0.0)

        print(f"\n{'='*70}\n  SCALE d={d}   (REF bpb = {ref_bpb:.4f}); "
              f"seed-noise (max std, n>=2) = {noise:.4f}\n{'='*70}")
        print(f"  {'variant':20s} {'bpb':>8s} {'std':>7s} {'lr':>6s} {'n':>2s} "
              f"{'dREF':>9s} {'dof':>10s}")
        for v, (m, s, lr, n) in sorted(stats.items(), key=lambda kv: kv[1][0]):
            dd = f"{m-ref_bpb:+.4f}" if ref_bpb is not None else "   n/a"
            print(f"  {v:20s} {m:8.4f} {s:7.4f} {lr:6g} {n:2d} {dd:>9s} "
                  f"{dof.get((d,v),0):>10,}")

        def g(v):
            return stats[v][0] if v in stats else None

        print("\n  -- decisive decompositions (delta vs REF; |delta|>noise => real) --")
        if g("ablate_phase") is not None:
            print(f"   phase off            : {g('ablate_phase')-ref_bpb:+.4f}   "
                  f"(carrier e^{{i*omega}} load-bearing?)")
        for v, lab in [("ablate_saturation", "saturation off (keep Born+norm)"),
                       ("ablate_born", "Born off (keep g+norm)"),
                       ("ablate_both", "both off (keep norm)")]:
            if g(v) is not None:
                print(f"   {lab:34s}: {g(v)-ref_bpb:+.4f}")
        if g("real_twin") is not None:
            print(f"   real twin (matched DOF){'':12s}: {g('real_twin')-ref_bpb:+.4f}   "
                  f"(complex/phase the driver?)")
        if g("no_mix") is not None:
            print(f"   mode coupling off{'':18s}: {g('no_mix')-ref_bpb:+.4f}   "
                  f"(the documented lever)")
        if g("selective") is not None and g("selective_control") is not None:
            sel = g("selective"); ctl = g("selective_control")
            print(f"   selective vs control{'':14s}: selective {sel-ref_bpb:+.4f}, "
                  f"control {ctl-ref_bpb:+.4f}  -> selectivity gain "
                  f"{sel-ctl:+.4f} (neg = data-dependence helps beyond params)")
        # depth trend
        depth = [(v, g(v)) for v in ("n_sub4", "n_sub8", "REF", "n_sub24") if g(v) is not None]
        if len(depth) >= 2:
            print("   depth (n_sub) trend       : "
                  + "  ".join(f"{v.replace('n_sub','d=').replace('REF','d=12')}={b:.4f}"
                              for v, b in depth))


if __name__ == "__main__":
    main()
