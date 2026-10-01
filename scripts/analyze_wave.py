"""Analyze a WaveLM metrics.jsonl: bpb trajectory, val curve, omega-variance,
gradient/lr health. Pure CPU/file read -- safe to run during GPU training.
"""

import argparse
import json
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != _here]


def load(path):
    train, val = [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "val_bpb" in rec:
                val.append(rec)
            elif "bpb" in rec:
                train.append(rec)
    return train, val


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", default="checkpoints/wavelm_run1/metrics.jsonl")
    args = ap.parse_args()
    if not os.path.exists(args.metrics):
        print(f"no metrics at {args.metrics}")
        return
    train, val = load(args.metrics)
    print("=" * 70)
    print(f"  WaveLM analysis: {args.metrics}")
    print("=" * 70)
    if train:
        last = train[-1]
        first = train[0]
        best_tr = min(train, key=lambda r: r["bpb"])
        print(f"  train steps logged: {len(train)}  (last step {last['step']})")
        print(f"  train bpb: first={first['bpb']:.3f} last={last['bpb']:.3f} "
              f"best={best_tr['bpb']:.3f}@{best_tr['step']}")
        print(f"  grad: last={last['grad']:.2f}  lr last={last['lr']:.2e}  "
              f"Var(omega) last={last['var_omega']:.3f}  lambda_omega last={last['lambda_omega']:.3f}")
        print(f"  GPU GB last={last.get('gpu_gb', float('nan')):.1f}")
        # trajectory every ~10% of logged points
        n = len(train)
        print("\n  train bpb trajectory:")
        for i in range(0, n, max(1, n // 12)):
            r = train[i]
            print(f"    step {r['step']:>6} | bpb={r['bpb']:.3f} ce={r['ce']:.3f} "
                  f"grad={r['grad']:.2f} Var(w)={r['var_omega']:.3f}")
    if val:
        print("\n  VALIDATION (bits/byte):")
        best = min(val, key=lambda r: r["val_bpb"])
        for r in val:
            mark = "  <-- best" if r is best else ""
            print(f"    step {r['step']:>6} | val_bpb={r['val_bpb']:.4f}{mark}")
        print(f"\n  best val: {best['val_bpb']:.4f} bpb @ step {best['step']}")
        print(f"  reference: uniform=8.0  byte-bigram~2.5  good char-LM~1.0-1.3")
        if best["val_bpb"] < 1.5:
            print(f"  *** below the <1.5 bpb gate ***")
    print("=" * 70)


if __name__ == "__main__":
    main()
