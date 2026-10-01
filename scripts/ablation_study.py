"""WaveLM driver-attribution campaign (doc 12): which components actually drive bpb?

Trains a REFERENCE small WaveLM and a battery of one-thing-changed ablations to a
FIXED token budget through the shared fair loop (baselines/harness.train_bytes_model),
so every number is directly comparable. Writes incremental JSON after EVERY run
(crash-safe) and wraps each run in try/except (+ a WaveLM OOM->checkpointing retry).

REF = conv + low-rank mode-coupling + saturation + phase + energy-norm + Born (muP).
Each variant changes exactly one thing vs REF:

  KNOWN LEVERS (re-confirm/quantify)
    no_mix              mode coupling OFF        (the documented 1.93->1.36 lever)
    n_sub{4,8,24}       evolution depth          (REF n_sub=12)
    untie_steps         per-step conv kernel     (dynamics capacity)
  DECISIVE WAVE-SPECIFIC (previously untested)
    ablate_phase        omega=0 (no carrier rotation)
    ablate_saturation   g=0, keep norm+Born      (isolate the |p|^2 nonlinearity)
    ablate_born         linear readout, keep g+norm (isolate the Born square)
    ablate_both         g=0 AND linear readout, keep norm
    real_twin           REAL structural twin at 2x width (matched real-DOF)
  HIDDEN LEVER (data-dependent routing -- the property an LTI WaveLM lacks)
    selective           input-dependent retention gate (Mamba-style selectivity)
    selective_control   params-matched NON-selective control (rank bumped to >= sel DOF)

Fairness: identical data/metric/optimiser/schedule/precision/budget; complex variants
keep muP (lr transfers, doc 11 optimum 3e-3); structurally-different variants
(real_twin, selective) are swept over lr in {1e-3, 3e-3} so neither is undertuned.
"""

import argparse
import json
import math
import os
import sys
import time

# path shim: make `import wave.*` / `import baselines.*` resolve to the repo root
_here = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != _here]
sys.path.insert(0, os.path.dirname(_here))

import torch

from wave.config import WaveConfig
from wave.model import WaveModel
from wave.real_model import RealWaveModel
from wave.data import ByteData
from baselines.harness import train_bytes_model, count_real_dof

CLR = 3e-3          # muP-tuned complex base lr (doc 11 follow-up: optimum at d=256)
ALT_LR = 1e-3       # alternative for the non-muP / new-param variants


def _ref_rank(d: int) -> int:
    return max(8, d // 8)


def ref_cfg(d: int, *, T: int, n_sub: int, mix_every: int, delta: int, seed: int,
            **over) -> WaveConfig:
    """REFERENCE config; `over` toggles exactly one thing for an ablation."""
    kw = dict(V=256, d=d, T=T, n_sub=n_sub, delta=delta, mix_channels=True,
              mix_rank=_ref_rank(d), mix_every=mix_every, mup=True, seed=seed,
              real_dtype="float32")
    kw.update(over)
    return WaveConfig(**kw)


def _matched_control_rank(d, *, T, n_sub, mix_every, delta, seed):
    """Smallest mix_rank whose real-DOF >= the selective model's real-DOF (so the
    non-selective control is NOT handicapped on parameter count)."""
    sel_dof = count_real_dof(WaveModel(ref_cfg(
        d, T=T, n_sub=n_sub, mix_every=mix_every, delta=delta, seed=seed,
        selective=True, selective_rank=16)))
    r = _ref_rank(d)
    while True:
        ctrl = WaveModel(ref_cfg(d, T=T, n_sub=n_sub, mix_every=mix_every,
                                 delta=delta, seed=seed, mix_rank=r))
        if count_real_dof(ctrl) >= sel_dof or r > _ref_rank(d) + 64:
            return r, count_real_dof(ctrl), sel_dof
        r += 1


def build_model(variant: str, d: int, seed: int, *, T, n_sub, mix_every, delta):
    """Construct the model for `variant`. Returns (model, note)."""
    base = dict(T=T, n_sub=n_sub, mix_every=mix_every, delta=delta, seed=seed)
    if variant == "REF":
        return WaveModel(ref_cfg(d, **base)), "conv+mix+sat+phase+norm+Born"
    if variant == "no_mix":
        return WaveModel(ref_cfg(d, mix_channels=False, **base)), "mode coupling OFF"
    if variant == "ablate_phase":
        return WaveModel(ref_cfg(d, ablate_phase=True, **base)), "omega=0"
    if variant == "ablate_saturation":
        return WaveModel(ref_cfg(d, ablate_saturation=True, **base)), "g=0, keep norm+Born"
    if variant == "ablate_born":
        return WaveModel(ref_cfg(d, ablate_born=True, **base)), "linear readout, keep g+norm"
    if variant == "ablate_both":
        return WaveModel(ref_cfg(d, ablate_saturation=True, ablate_born=True, **base)), \
            "g=0 + linear readout, keep norm"
    if variant == "n_sub4":
        return WaveModel(ref_cfg(d, **{**base, "n_sub": 4})), "depth 4"
    if variant == "n_sub8":
        return WaveModel(ref_cfg(d, **{**base, "n_sub": 8})), "depth 8"
    if variant == "n_sub24":
        return WaveModel(ref_cfg(d, **{**base, "n_sub": 24})), "depth 24"
    if variant == "untie_steps":
        return WaveModel(ref_cfg(d, untie_steps=True, **base)), "per-step conv kernel"
    if variant == "selective":
        return WaveModel(ref_cfg(d, selective=True, selective_rank=16, **base)), \
            "data-dependent retention gate (r_sel=16)"
    if variant == "selective_control":
        r, cdof, sdof = _matched_control_rank(d, **base)
        return WaveModel(ref_cfg(d, mix_rank=r, **base)), \
            f"non-selective, rank {r} (DOF {cdof} >= selective {sdof})"
    if variant == "real_twin":
        # REAL structural twin: width 2d, SAME rank as REF -> matched real-DOF.
        return RealWaveModel(ref_cfg(2 * d, mix_rank=_ref_rank(d), **base)), \
            "real-valued twin, width 2d, matched DOF"
    raise ValueError(f"unknown variant {variant}")


# variant -> (seeds, lrs) at the FULL scale (d=256) and the TREND scale (d=512)
SCALE_FULL = [
    ("REF", [0, 1, 2], [CLR]),
    ("no_mix", [0, 1, 2], [CLR]),
    ("ablate_phase", [0, 1, 2], [CLR]),
    ("ablate_saturation", [0, 1, 2], [CLR]),
    ("ablate_born", [0, 1, 2], [CLR]),
    ("real_twin", [0, 1], [ALT_LR, CLR]),
    ("selective", [0, 1], [CLR, ALT_LR]),
    ("selective_control", [0, 1], [CLR]),
    ("ablate_both", [0], [CLR]),
    ("n_sub4", [0], [CLR]),
    ("n_sub8", [0], [CLR]),
    ("n_sub24", [0], [CLR]),
    ("untie_steps", [0], [CLR]),
]
SCALE_TREND = [
    ("REF", [0], [CLR]),
    ("no_mix", [0], [CLR]),
    ("ablate_phase", [0], [CLR]),
    ("ablate_saturation", [0], [CLR]),
    ("ablate_born", [0], [CLR]),
    ("real_twin", [0], [CLR]),
    ("selective", [0], [CLR]),
    ("selective_control", [0], [CLR]),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_txt", default="data/tinystories/train.txt")
    ap.add_argument("--val_txt", default="data/tinystories/val.txt")
    ap.add_argument("--out", default="checkpoints/ablation/ablation_results.json")
    ap.add_argument("--T", type=int, default=512)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--steps", type=int, default=2500)        # ~41M bytes
    ap.add_argument("--n_sub", type=int, default=12)
    ap.add_argument("--mix_every", type=int, default=2)
    ap.add_argument("--delta", type=int, default=2)
    ap.add_argument("--val_every", type=int, default=500)
    ap.add_argument("--val_batches", type=int, default=40)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--scales", default="256,512")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny run (REF + ablate_phase at d=128, 30 steps) to test the driver")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    tr = ByteData(args.train_txt, args.T)
    va = ByteData(args.val_txt, args.T)

    results = {"meta": {"T": args.T, "batch": args.batch, "steps": args.steps,
                        "n_sub": args.n_sub, "mix_every": args.mix_every,
                        "delta": args.delta, "device": args.device,
                        "bytes_per_run": args.steps * args.batch * args.T,
                        "started": time.strftime("%Y-%m-%d %H:%M:%S")},
               "runs": []}

    def save():
        json.dump(results, open(args.out, "w"), indent=2)

    # ---- resume: keep any completed runs from a prior (possibly crashed) campaign ----
    done = set()
    if os.path.exists(args.out):
        try:
            prior = json.load(open(args.out))
            results["runs"] = prior.get("runs", [])
            done = {r["label"] for r in results["runs"] if r.get("val_bpb_best") is not None}
            if done:
                print(f"RESUME: {len(done)} completed runs in {args.out}; skipping them",
                      flush=True)
        except Exception:
            pass

    # ---- build the job list (variant, d, seed, lr) in decisiveness order ----
    if args.smoke:
        plan = [(128, [("REF", [0], [CLR]), ("ablate_phase", [0], [CLR])])]
        steps = 30
    else:
        scales = [int(s) for s in args.scales.split(",")]
        plan = []
        for i, d in enumerate(scales):
            plan.append((d, SCALE_FULL if i == 0 else SCALE_TREND))
        steps = args.steps

    results["meta"]["steps"] = steps        # effective steps (smoke overrides args.steps)
    results["meta"]["bytes_per_run"] = steps * args.batch * args.T

    jobs = []
    for d, spec in plan:
        for variant, seeds, lrs in spec:
            for seed in seeds:
                for lr in lrs:
                    jobs.append((variant, d, seed, lr))

    print("=" * 78, flush=True)
    print(f"ABLATION CAMPAIGN | {len(jobs)} runs | T={args.T} batch={args.batch} "
          f"steps={steps} (~{steps*args.batch*args.T/1e6:.0f}M bytes/run) device={args.device}",
          flush=True)
    print("=" * 78, flush=True)

    def run_one(variant, d, seed, lr):
        label = f"{variant}_d{d}_s{seed}_lr{lr:g}"
        t0 = time.perf_counter()
        for attempt, ckpt in enumerate([False, True]):
            model = None
            try:
                if args.device.startswith("cuda"):
                    torch.cuda.empty_cache()
                model, note = build_model(variant, d, seed, T=args.T, n_sub=args.n_sub,
                                          mix_every=args.mix_every, delta=args.delta)
                r = train_bytes_model(
                    model, tr, va, steps=steps, batch=args.batch, T=args.T, lr=lr,
                    device=args.device, seed=seed, val_every=args.val_every,
                    val_batches=args.val_batches, wave_checkpoint=ckpt, label=label)
                r.update({"variant": variant, "d": d, "note": note, "oom_retry": attempt == 1})
                del model
                if args.device.startswith("cuda"):
                    torch.cuda.empty_cache()
                print(f"  [{label}] dof={r['params_dof']:,} -> bpb_best={r['val_bpb_best']} "
                      f"bpb_final={r['val_bpb_final']} tok/s={r['tok_per_s']:,.0f} "
                      f"peak={r['peak_gb']}G wall={r['wall_s']}s | {note}", flush=True)
                return r
            except RuntimeError as e:
                if model is not None:
                    del model
                if args.device.startswith("cuda"):
                    torch.cuda.empty_cache()
                is_oom = "out of memory" in str(e).lower()
                # only the complex WaveModel has an activation-checkpointing path
                if is_oom and variant != "real_twin" and attempt == 0:
                    print(f"  [{label}] OOM; retrying with activation checkpointing", flush=True)
                    continue
                print(f"  [{label}] FAILED: {type(e).__name__}: {str(e)[:160]}", flush=True)
                return {"label": label, "variant": variant, "d": d, "seed": seed,
                        "lr": lr, "error": str(e)[:200], "wall_s": round(time.perf_counter() - t0, 1)}
            except Exception as e:        # noqa: BLE001 -- record & continue; never kill the campaign
                if model is not None:
                    del model
                if args.device.startswith("cuda"):
                    torch.cuda.empty_cache()
                print(f"  [{label}] FAILED: {type(e).__name__}: {str(e)[:160]}", flush=True)
                return {"label": label, "variant": variant, "d": d, "seed": seed,
                        "lr": lr, "error": f"{type(e).__name__}: {str(e)[:200]}",
                        "wall_s": round(time.perf_counter() - t0, 1)}
        return {"label": label, "variant": variant, "d": d, "error": "oom_after_retry"}

    n_skip = n_run = 0
    for variant, d, seed, lr in jobs:
        if f"{variant}_d{d}_s{seed}_lr{lr:g}" in done:
            n_skip += 1
            continue
        results["runs"].append(run_one(variant, d, seed, lr))
        n_run += 1
        save()
    print(f"\n({n_run} new runs, {n_skip} skipped from resume)", flush=True)

    results["meta"]["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    save()
    tr.close(); va.close()

    # ---- summary: best-lr bpb per (variant, d), averaged over seeds, vs REF ----
    print("\n" + "=" * 78, flush=True)
    print("SUMMARY (best-lr val bpb, mean over seeds; delta vs REF at same scale)", flush=True)
    ok = [r for r in results["runs"] if r.get("val_bpb_best") is not None]
    scales = sorted({r["d"] for r in ok})
    for d in scales:
        # best lr per variant = the lr minimising mean-over-seeds bpb
        per_var = {}
        for r in ok:
            if r["d"] != d:
                continue
            per_var.setdefault(r["variant"], {}).setdefault(r["lr"], []).append(r["val_bpb_best"])
        var_best = {v: min(((sum(b) / len(b)), lr) for lr, b in by_lr.items())
                    for v, by_lr in per_var.items()}
        ref = var_best.get("REF", (None,))[0]
        print(f"\n  d={d}:", flush=True)
        for v, (mean_bpb, lr) in sorted(var_best.items(), key=lambda kv: kv[1][0]):
            n = len(per_var[v][lr])
            delta = f"{mean_bpb - ref:+.4f}" if ref is not None else "  n/a"
            print(f"    {v:20s} bpb={mean_bpb:.4f} (lr={lr:g}, n={n})  dREF={delta}", flush=True)
    print(f"\nwrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
