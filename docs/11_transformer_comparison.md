# WaveLM vs. a byte-level Transformer — Empirical Head-to-Head and Scaling Law (E1 + E2)

> **Verdict up front (honest):** at matched parameters and matched data on
> TinyStories, a vanilla byte-level Transformer is **stronger** than WaveLM. With
> *properly tuned* learning rates, WaveLM is **near-parity at very small scale
> (~0.5M params: 1.409 vs 1.385 bpb)** but **loses the scaling race** — the
> Transformer improves ~2.5× more per parameter and pulls away (at ~3M: 1.027 vs
> 1.266 bpb). On wall-clock and memory the Transformer dominates by 1–2 orders of
> magnitude. This is a **negative result** for WaveLM's competitiveness on this
> data/scale — reported straight. The one silver lining (small-scale parity) and
> the regimes this does **not** test (long-range, the Pile, O(1)/byte inference)
> are stated in the caveats.

Reproducible artifacts: `baselines/byte_gpt.py`, `baselines/harness.py`,
`scripts/run_compare.py`; raw results `checkpoints/compare/compare_results.json`.

---

## 1. Method — what was matched (fairness controls)

Both architectures go through the **identical** loop (`baselines/harness.py`):

```
   same data (ByteData)         same byte batches per seed (shared RNG)
   same loss (wave.readout.ce_loss, next-byte CE)   same metric (bits-per-byte)
   same optimiser family (AdamW, decoupled WD)      same cosine warmup-decay
   same grad clip (1.0)         same precision (fp32)   same token budget
   params matched on REAL DOF (a complex WaveLM param = 2 real DOF)
   each architecture's lr tuned by its own sweep (a strong, fair baseline)
```

- **Baseline** (`ByteGPT`): nanoGPT-style decoder-only — causal SDPA attention +
  RoPE + pre-LN + GELU MLP (4×) + tied embeddings + GPT-2 init. Correctness gated
  by `tests/test_byte_gpt.py` (causality, overfit-a-batch, uniform init, tying).
- **Data:** TinyStories, byte-level (V=256). T=512, batch=32. Matched token budget
  **2500 steps ≈ 41M bytes** per ladder run (a fixed, modest budget — see caveats).
- **Sizes (real DOF), matched within ~5–15% per rung:** ~0.5M, ~1.2M, ~3M, ~7M.
- **Seeds:** 3 seeds at two sizes for error bars.

---

## 2. E2 — Scaling law (the headline)

### 2.1 Matched-token ladder (initial sweep lr; seed 0)

```
   params        WaveLM bpb     byte-GPT bpb     gap (GPT better by)
   ~0.5M           1.528          1.385               0.143
   ~1.2M           1.448          1.093               0.355
   ~3.0M           1.364          1.037               0.327
   ~7.0M           1.324          0.947               0.377
```

Fitted `bpb ≈ E_inf + A·N^(−α)`:

```
   WaveLM :  E_inf = 1.20 ,  α = 0.39
   GPT    :  E_inf = 0.91 ,  α = 0.90
```

The Transformer's curve is **below WaveLM's at every size**, descends **~2.3×
faster** (α 0.90 vs 0.39), and extrapolates to a **lower floor** (0.91 vs 1.20).

### 2.2 The rigor correction (lr was at the grid edge)

Both architectures' best lr sat at the **top** of the initial sweep grid, so I
extended it upward. Result (same 2500-step budget, directly comparable):

```
   WaveLM ~0.5M:  lr 1e-3 -> 1.528  |  lr 3e-3 -> 1.409  |  lr 1e-2 -> 1.463
   WaveLM ~3.0M:  lr 1e-3 -> 1.364  |  lr 3e-3 -> 1.266
   GPT    ~0.5M:  lr 3e-3 -> 1.385  |  lr 1e-2 -> 1.405  |  lr 3e-2 -> 2.840 (diverged)
   GPT    ~3.0M:  lr 3e-3 -> 1.037  |  lr 1e-2 -> 1.027
```

So **WaveLM was undertuned by ~0.1 bpb** (its real optimum is lr≈3e-3, the same at
both sizes — muP's lr-transfer held; the *initial grid* was simply too low). GPT was
already ~optimal. The **fair, tuned head-to-head**:

```
   params     WaveLM (tuned)   byte-GPT (tuned)   gap
   ~0.5M         1.409            1.385           0.024   <- near parity
   ~3.0M         1.266            1.027           0.239   <- GPT clearly ahead
```

Tuned per-parameter slopes (0.5M→3.0M, a 6× param increase):

```
   WaveLM : 1.409 -> 1.266   = -0.143 bpb   (~ -0.08 / e-fold of params)
   GPT    : 1.385 -> 1.027   = -0.358 bpb   (~ -0.20 / e-fold of params)   ~2.5x steeper
```

### 2.3 Picture (bpb vs params, log-x; tuned where measured)

```
 bpb
 1.55 |  o                                   o = WaveLM (initial-sweep lr)
 1.50 |                                       * = WaveLM (tuned lr 3e-3)
 1.45 |        o                              + = byte-GPT (tuned)
 1.40 |  *  +                                 (lower = better)
 1.35 |              o     o
 1.30 |        *
 1.27 |              *
 1.10 |        +
 1.05 |              +
 0.95 |                    +
 0.90 +------|------|------|------|----->  params (log)
        0.5M   1.2M   3M     7M

 Read: tuned WaveLM (*) starts ~level with GPT (+) at 0.5M, but GPT's line
 drops far faster — by 3M it is ~0.24 bpb lower, and the gap keeps widening.
```

**E2 conclusion:** even with WaveLM properly tuned, **the Transformer scales
better** on TinyStories — lower floor, ~2.5× steeper exponent. WaveLM's near-parity
at ~0.5M does not survive scaling.

---

## 3. E1 — Head-to-head details

### 3.1 Significance (seeds; n=3)

```
   WaveLM ~0.5M: 1.528 +/- 0.002        GPT ~0.5M: 1.394 +/- 0.006
   WaveLM ~3.0M: 1.367 +/- 0.002        GPT ~3.0M: 1.033 +/- 0.007
```

Run-to-run noise is **±0.002–0.007 bpb** — the gaps (0.14–0.38 untuned, 0.02–0.24
tuned) are **many σ**, i.e. real, not luck.

### 3.2 Matched wall-clock (≈480 s budget, same GPU)

```
   arch     params     bpb_best   steps in 480s
   GPT      ~1.2M       0.869       25,500
   GPT      ~3.0M       0.842       11,000
   WaveLM   ~1.2M       1.762        2,000
   WaveLM   ~3.0M       1.902        1,250     <- WORSE than its 1.2M: throughput-starved
```

At equal wall-clock the Transformer fits **5–13× more steps** and reaches **~1.0
bpb lower**. For WaveLM, *bigger is worse at fixed time* (it can't afford the steps).

### 3.3 Throughput and memory (the engineering reality)

```
   size    WaveLM tok/s   GPT tok/s    ratio     WaveLM peak   GPT peak
   ~0.5M      117k         2.51M        ~21x       3.9 GB       0.3 GB
   ~1.2M       66k         867k         ~13x       6.1 GB       0.9 GB
   ~3.0M       37k         369k         ~10x      10.3 GB       2.2 GB
   ~7.0M      6.6k         208k         ~31x      16.1 GB       3.6 GB   (WaveLM near OOM)
```

WaveLM is **10–31× slower** and **5–12× heavier**. The 7M WaveLM took **103 min**
for one 41M-byte run; the 7M GPT took **3 min**. (WaveLM is deliberately
unoptimised — kernel fusion was evaluated and dropped, doc 07 §2/§8 — so wall-clock
flatters GPT. But note §2: even the *compute-agnostic* bpb-per-param favours GPT.)

---

## 4. Honest caveats (what this does and does not establish)

1. **TinyStories only.** Small, clean, short-range, and *attention-friendly*. It
   does not probe long-range dependencies, hard/diverse data (the Pile), or the
   `O(1)/byte` streaming-inference regime where WaveLM's constant per-token cost
   could matter. WaveLM's structural advantages (constant-cost long-context
   inference, log-depth receptive field) are **not exercised here**.
2. **One WaveLM scaling axis.** I scaled width `d` (with rank ∝ d/8). WaveLM's
   historical gains came largely from **mode-coupling rank** and **n_sub depth**;
   a different axis might scale better. Untested — `scripts/scaling_study.py`
   (doc 07) supports `--axis mix_rank|n_sub` for this follow-up.
3. **Fixed 41M-byte budget.** Absolute bpb is undertrained (not converged); the
   comparison is valid *at matched budget*, but neither model is at its asymptote.
4. **WaveLM unoptimised.** No kernel fusion (dropped as low-value, doc 07). Its
   wall-clock and throughput would improve with engineering; the bpb-per-param
   result would not.
5. **Baseline strength.** `ByteGPT` is a fair, strong, standard baseline (tuned lr,
   weight-tied, RoPE, GPT-2 init) — not a strawman, but also not the strongest
   possible LM. Byte-level is a regime often considered *favourable* to non-attention
   models, and the Transformer still won.
6. **lr grid.** The initial sweep was too coarse (undertuned WaveLM by ~0.1 bpb);
   corrected by the follow-up (§2.2). Both tuned and untuned numbers are reported.

---

## 5. Verdict and what would change it

**On TinyStories, at the scales tested, WaveLM is not competitive with a vanilla
byte-Transformer.** Properly tuned it reaches near-parity at ~0.5M params, but the
Transformer scales ~2.5× better per parameter, has a lower extrapolated floor, and
is 10–31× faster — so it pulls decisively ahead by ~3M and beyond. WaveLM performs
*consistent with its own documented ~1.36 bpb* (no bug; it is being itself), and the
Transformer is simply the stronger architecture in this regime.

**What would be needed to claim competitiveness (testable next steps, not done here):**
- a regime that rewards WaveLM's structure — **long-context retrieval / streaming
  deployment cost** (constant per-token inference) or **the Pile** (harder, longer
  range) — measured as a **quality-vs-inference-cost** Pareto frontier, not just bpb;
- WaveLM scaled along its **stronger axis** (rank / n_sub) rather than width;
- and, for any wall-clock claim, the throughput gap closed by engineering.

Until one of those shows WaveLM at or above the Transformer frontier, the honest
statement is: **WaveLM is an interesting, working, from-scratch architecture that, on
this benchmark, the Transformer beats.**

---

## 6. Reproduce

```
# full E1+E2 driver (lr sweep -> matched ladder -> seeds -> wall-clock; ~4 h on a 4080S)
python scripts/run_compare.py
# baseline correctness gates
python -m pytest tests/test_byte_gpt.py -q
```

*Numbers in this document are from `checkpoints/compare/compare_results.json`
(run 2026-05-23, 26 runs, 0 errors) plus the wider-lr follow-up. Method:
`baselines/harness.py`. Related: doc 09 (the loss/metric both share), doc 07
(why WaveLM is unoptimised), doc 08 (WaveLM's lineage).*
