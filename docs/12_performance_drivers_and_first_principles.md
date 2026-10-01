# What Fundamentally Drives WaveLM — A First-Principles, Ablation-Grounded Analysis

> **Question (verbatim intent).** WaveLM works, but it is not yet at Transformer
> performance. *What is the key empirical driver of its learning?* Are we using that
> primitive properly? What is **decorative or weak**, and — more importantly — what
> **fundamentally drives** performance? Are we **limiting** the real drivers; is
> something unnecessary **in the way**? Is the "wave" framing the driver, or is the
> true driver something else? Is there a **hidden lever** (a path to paradigm-shifting)?
>
> This document answers those questions **by controlled measurement, not by the design
> prose**. Every load-bearing / decorative verdict below cites a measured Δ in
> bits-per-byte (bpb) from a matched ablation campaign, or an existing test/run. Where
> the evidence is thin or untested, it says so.

**Method.** A reference small WaveLM (REF) and a battery of *one-thing-changed*
ablations were trained to an **identical token budget** (TinyStories, byte-level,
T=512, batch=32, **2500 steps ≈ 41 M bytes/run**) through the same fair loop
(`baselines/harness.train_bytes_model`: same data/optimiser/schedule/precision; muP so
the lr transfers; complex variants at the muP-tuned lr=3e-3; structurally-different
variants lr-swept). Decisive ablations use **3 seeds**; the study spans **two scales
d∈{256,512}**. Seed noise is **±0.002–0.006 bpb** for the well-behaved variants (so any
|Δ|>0.01 is real), rising to ±0.02–0.05 for the degenerate (near-linear / real-valued)
ablations. Artifacts: `scripts/ablation_study.py`, `scripts/analyze_ablation.py`,
`checkpoints/ablation/ablation_results.json`; instruments gated by
`tests/test_wave_ablations.py` (defaults bit-identical; FP64 gradient checks).

---

## 0. Executive verdict

```
   WHAT DRIVES WaveLM (by measured bpb impact, REF d=256 = 1.462 bpb):

   ┌─ FUNDAMENTAL (removing it is catastrophic) ───────────────────────────────┐
   │  1. Pointwise NONLINEARITY (saturation)   remove -> +0.74 bpb   ████████   │
   │  2. CHANNEL MIXING (mode coupling)        remove -> +0.49 bpb   █████      │
   │  3. The EMBEDDING + contrastive readout   (the learning engine; 40-99% of  │
   │     params; not ablatable without deleting the model — see §4.6)           │
   │  4. DEPTH / iteration (n_sub)             cut 12->4 -> +0.10 bpb  █         │
   └────────────────────────────────────────────────────────────────────────┘
   ┌─ MODEST / REAL-BUT-SMALL ─────────────────────────────────────────────────┐
   │  5. COMPLEX-valued representation (package) real twin -> +0.20 bpb  ██      │
   │  6. PHASE carrier  e^{i*omega}             remove -> +0.019 bpb  ·  (shrinks │
   │                                            with scale: +0.008 at d=512)     │
   └────────────────────────────────────────────────────────────────────────┘
   ┌─ DECORATIVE / IN-THE-WAY (removing it HELPS) ─────────────────────────────┐
   │  7. The BORN |.|^2 readout                 remove -> -0.03 bpb (better!)    │
   │                                            (more harmful at scale: -0.04)   │
   └────────────────────────────────────────────────────────────────────────┘
   ┌─ CONSTRUCTIVE LEVERS WE ARE UNDER-USING ──────────────────────────────────┐
   │  • untie_steps (per-step conv kernel)      -> -0.08 bpb  (best single win)  │
   │  • data-dependent SELECTIVITY              -> -0.01..-0.02 vs matched ctrl, │
   │                                               and GROWS with scale (§7)     │
   └────────────────────────────────────────────────────────────────────────┘
```

**The one-paragraph answer.** WaveLM's performance is driven by **generic
deep-learning primitives**, not by the wave-specific machinery. The two largest levers
are a **pointwise nonlinearity** (the per-channel saturation) and **channel mixing**
(mode coupling) — together with the **byte embedding read by a contrastive softmax**
(the engine inherited from the predecessor) and **iterated depth**. The most
*distinctively "wave"* parts fare poorly under measurement: the **Born `|·|²` readout
is not load-bearing — removing it slightly *improves* bpb** (and more so at larger
scale), and the explicit **phase carrier `e^{iω}` is a minor contributor** that shrinks
with scale. The **complex-valued representation as a whole** does contribute a real
~0.2 bpb (the real-valued twin at matched parameters is clearly worse), but that benefit
lives in the **complex convolution/mixing**, *not* in the nominal carrier frequencies
the "wave" story emphasises. The deepest structural limit is that WaveLM is, stripped
of metaphor, an essentially **linear time-invariant (LTI) system with one weak radial
nonlinearity and a quadratic readout — it has no data-dependent routing**, the property
that powers both attention and selective SSMs. A minimal data-dependent gate already
buys a small, **scale-growing** improvement beyond matched parameters — the most
promising lead for closing the gap.

---

## 1. The model stripped of its metaphor (what actually computes)

One internal step `τ→τ+1` on the complex field `ψ ∈ ℂ^{B×T×d}` is five operations
(`wave/field.py:wave_step`). Annotated by what each one *is*, mathematically:

```
   ψ ──▶ (1) dilated causal conv     p[n,k] = β·ψ[n,k] + Σ_j K[j,k]·ψ[n−jD_τ,k]
   │         LINEAR, complex, depthwise; the ONLY cross-position op; INPUT-INDEPENDENT
   │
   ├──▶ (2) mode coupling (opt.)      p ← M·p   or   p ← p + U(Vᴴp)
   │         LINEAR, complex, cross-channel;                  INPUT-INDEPENDENT
   │
   ├──▶ (3) saturation                q[n,k] = p[n,k] / (1 + g·|p[n,k]|²)
   │         NONLINEAR — but RADIAL: the denominator is real, so it scales magnitude
   │         and NEVER changes a channel's phase. The model's ONLY elementwise
   │         nonlinearity. (weakly input-dependent: a magnitude gate)
   │
   ├──▶ (4) phase rotation            r[n,k] = e^{iω_k}·q[n,k]
   │         LINEAR (diagonal unitary), magnitude-preserving;  INPUT-INDEPENDENT
   │
   └──▶ (5) energy norm               ψ⁺[n] = √d·r[n]/‖r[n]‖
             degree-0 homogeneous projection ≈ a complex RMSNorm; per-position
   ────────────────────────────────────────────────────────────────────────────────
   readout   a(o,n)=⟨W_E[o],ψ[n]⟩ ;  ℓ=|a|²/T_o ;  softmax ;  next-byte CE
             QUADRATIC readout (the "Born rule")
```

Read this without the wave vocabulary and a blunt fact emerges:

> **Every operation except the saturation is linear in `ψ`, and none of the mixing is
> data-dependent.** `K`, `ω`, `M`, `U`, `V` are fixed weights; they do not depend on
> the input. The only place the data conditions the computation is the *radial*
> magnitude gate in (3) and the per-position rescale in (5). So WaveLM is, structurally,
> **a complex, energy-normalised, (quasi-)linear-time-invariant sequence-and-channel
> mixer, punctuated by one weak radial nonlinearity, and read out by a single quadratic
> form.** (The only time-variation across `τ` is the dilation *schedule*; within a
> dilation the map is LTI.)

That is a precise, falsifiable structural claim — and §3 and §7 show it is the key to
both *why the wave parts are not the driver* and *why the model scales poorly*.

### 1.1 Where the parameters actually live (exact accounting)

For the campaign's REF config (`d=256`, rank 32, `n_sub=12`, `mix_every=2`), counting
real degrees of freedom (a complex parameter = 2 real DOF):

```
   REF d=256  (328,963 real DOF)              REF d=512 (1,051,139)   base, no mix (d=256)
   ────────────────────────────              ────────────────────    ───────────────────
   embedding W_E    131,072   39.8% ████      262,144   24.9% ██      131,072   99.0% ████████
   channel-mixing   196,608   59.8% ██████    786,432   74.8% ███████   —         —
   "WAVE" dynamics    1,285    0.39% ·           2,562    0.24% ·       1,283    1.0% ·
     (K + ω + β,g,T_o)
```

**The "wave-propagation machinery" everything is named after — the conv kernel `K`, the
carrier frequencies `ω`, the scalars `β,g,T_o` — is 0.2–0.4 % of the parameters.**
Capacity lives almost entirely in two **architecture-agnostic** places: the byte
*embedding* and the *channel-mixing* matrices. This is the first hint that the wave
machinery is not where the model's power is stored. The ablations confirm it.

---

## 2. The empirical adjudication (the ablation matrix)

REF d=256 = **1.4622 ± 0.0032** bpb (3 seeds); REF d=512 = **1.3401** (1 seed).
Δ = (variant − REF) at the same scale; negative = *better than REF*.

```
   variant (d=256)        bpb      ΔREF      n   reading
   ─────────────────────────────────────────────────────────────────────────────
   untie_steps          1.3826   -0.080     1   per-step conv kernels: best single win
   ablate_born          1.4328   -0.029     3   remove Born |·|²  -> BETTER
   selective            1.4453   -0.017     2   data-dependent gate
   selective_control    1.4550   -0.007     2   same params, NOT data-dependent
   REF                  1.4622    0.000     3   conv+mix+sat+phase+norm+Born
   ablate_phase         1.4807   +0.019     3   ω≡0 (no carrier)
   n_sub8               1.5060   +0.044     1   depth 8  (REF is 12)
   n_sub4               1.5587   +0.097     1   depth 4
   n_sub24              1.5958   +0.134     1   depth 24  (MORE depth hurts here)
   real_twin            1.6584   +0.196     2   real-valued, matched DOF, lr-swept
   no_mix               1.9532   +0.491     3   mode coupling OFF
   ablate_saturation    2.1994   +0.737     3   g=0  (saturation OFF)
   ablate_both          2.3291   +0.867     1   g=0 AND linear readout (≈ bigram ceiling)

   variant (d=512)        bpb      ΔREF      reading (1 seed; muP transfers lr)
   ─────────────────────────────────────────────────────────────────────────────
   ablate_born          1.3008   -0.039     remove Born |·|²  -> BETTER (more so at scale)
   selective            1.3175   -0.023     data-dependent gate (gain grows with scale)
   selective_control    1.3370   -0.003     same params, not data-dependent
   REF                  1.3401    0.000
   ablate_phase         1.3485   +0.008     carrier effect SHRINKS with scale
   no_mix               1.7951   +0.455     mode coupling OFF
   ablate_saturation    1.9993   +0.659     saturation OFF
   real_twin            1.5368   +0.197     real-valued twin, matched DOF (lr-tuned: muP
                                            lr 3e-3 DIVERGED to 3.71; 1e-3 is the fair value)
```

> **The complex-vs-real penalty is scale-stable: +0.196 (d=256) and +0.197 (d=512).**
> Complex representation buys a clean, reproducible ~0.2 bpb at both scales — but (§3.1)
> that value lives in the complex conv/mixing, not the carrier `ω`. Rigor note: the real
> twin has no muP, so its lr does not transfer; the muP lr 3e-3 diverged at width 1024
> and an explicit lr sweep was needed (§6).

**Two independent sanity anchors** confirm the harness measures the right thing:
`no_mix → 1.953` reproduces the historically documented base-model plateau **1.93**
(doc 05), and `ablate_both → 2.329` lands at the **byte-bigram ceiling ≈ 2.5** (the
class the predecessor was stuck in). The instrument is faithful.

```
   bpb  Δ vs REF (d=256), descending = more load-bearing
   +0.74 │ saturation off          ███████████████████████████████  the #1 driver
   +0.49 │ mode coupling off       █████████████████████            the #2 driver
   +0.20 │ real twin (no complex)  ████████                         complex helps (some)
   +0.10 │ depth 12→4              ████                             depth helps
   +0.02 │ phase off               █                                carrier: minor
    0.00 │ REF ─────────────────────────────────────────────────────────────────
   -0.03 │ Born off                ▓                                Born HURTS (remove it)
   -0.08 │ untie_steps             ▓▓▓                              cheap win
```

---

## 3. What lifts the model past the bigram ceiling — and what does not

The design thesis (doc 01) was: complex carriers + per-channel saturation (`|p|²`
products) + the Born square together make the conditional a **degree-≥4** function of
the embeddings, escaping the linear-in-embeddings bigram class. The composition gate
(`test_wave_expressivity.py`) proved the *combination* is necessary on modular addition.
But it bundled three things into one "linear" switch. **Decoupled, the ablations show
exactly which factor does the work** on real text:

```
   keep norm + mixing throughout, vary the nonlinearities:

   saturation ✓ , Born ✓   (REF)          1.462    ← both
   saturation ✓ , Born ✗   (ablate_born)  1.433    ← Born removed: BETTER
   saturation ✗ , Born ✓   (ablate_sat)   2.199    ← saturation removed: COLLAPSE
   saturation ✗ , Born ✗   (ablate_both)  2.329    ← bigram ceiling
```

> **The per-channel saturation is the composition nonlinearity; the Born square is
> not.** Removing the saturation (`g=0`) makes the dynamics fully linear and the model
> collapses 0.74 bpb toward the bigram ceiling. Removing the Born square (linear
> readout, saturation kept) *improves* bpb by 0.03 — and the improvement **grows with
> scale** (−0.04 at d=512). The squared readout is redundant given the saturation's
> nonlinearity, and its always-positive, quadratic shape is mildly *worse* than a plain
> linear readout for fitting next-byte logits.

This directly overturns part of the design narrative: doc 09 calls the Born `|·|²` "all
load-bearing" and treats it as a co-equal pillar. **On TinyStories at these scales it is
not load-bearing for performance — it is slightly in the way.** (It remains the elegant
"normalised-state probability" story, and may matter for the modular-addition gate where
*some* readout nonlinearity is required; but as a *performance* driver on real text it is
decorative-to-harmful.)

### 3.1 Is the *complex / wave* structure the driver? A two-level decomposition

Two ablations bracket the value of "complex":

```
   ablate_phase  (ω≡0, but field/conv/mixing stay COMPLEX)   ΔREF = +0.019  (d=256)
   real_twin     (fully REAL: real conv, real mix, no phase   ΔREF = +0.196  (d=256)
                  anywhere, squared-real readout, matched DOF, lr-swept)
```

```
   value of "complex"  ≈ +0.20 bpb            ┌── explicit carrier ω : ~0.02  (10%)
   (real twin penalty)         │ decomposes → │
                               └──────────────┴── complex conv/mixing : ~0.18 (90%)
```

> The complex-valued representation **does** earn its keep (~0.2 bpb, real and clearly
> above noise) — but **not through the "carrier frequencies" the wave story
> emphasises.** Zeroing `ω` costs only ~0.02; going *fully real* costs ~0.20. The
> ~0.18 difference is the value of letting the **convolution and mixing be complex**
> (their taps `K[j,k]` and entries of `M` carry arbitrary phase, and the channels act as
> coupled (re,im) pairs rather than `2d` independent reals). The "carrier `ω_k`" — the
> most explicitly wave-like parameter, and the one the spectral-diversity guard was
> built to protect — is the *least* load-bearing piece of the complex story. This is
> consistent with the measured fact that `ω` barely trains (Var ω 3.2899→3.2898; STATUS
> §3c): it sits near its init because it is not doing much.

So the honest verdict on the "wave" framing: it is a **lens that generated a working
design**, and one of its commitments (complex values) pays off modestly via the
complex conv/mixing — but the **distinctive wave readout/carrier apparatus (Born `|·|²`,
the carrier `ω`, the dispersion story) is not the performance driver**. The drivers are
the generic ingredients any strong sequence model has: a nonlinearity, channel mixing,
an embedding read by a contrastive softmax, and depth.

---

## 4. Are we limiting the real drivers? What is in the way?

### 4.1 The dynamics are parameter-starved, and we under-use the cure

The "wave dynamics" carry 0.2–0.4 % of parameters (§1.1) and `K` is **tied across all
`n_sub` steps**. Untying it (`untie_steps`, a distinct kernel per step) was the **best
single change measured: −0.080 bpb** at +0.4 % params. We are leaving free capacity on
the table by tying the propagation operator. (Doc 06 anticipated this; the measurement
now confirms it is the cheapest real win.)

### 4.2 Depth has an optimum at fixed budget — more is not better

`n_sub`: 4 → 8 → 12 improves (1.559 → 1.506 → 1.462), but 12 → 24 *regresses* to 1.596.
At a fixed token budget more iteration means more per-step parameters/compute that go
undertrained, and a harder optimisation. The receptive-field theorem says `⌈log₂T⌉≈9`
steps suffice for full reach at T=512; beyond ~`n_sub≈`reach, extra depth is not buying
context, only optimisation difficulty. **We should scale depth with the data/step
budget, not blindly.**

### 4.3 The Born square is mildly in the way (§3) — a free 0.03–0.04 bpb by removing it.

### 4.4 The real bottleneck: no data-dependent routing (the LTI ceiling)

§1's structural reading is the deep limit. A transformer routes information with
**content-based attention** `softmax(QKᵀ)`; a selective SSM (Mamba) makes its recurrence
**input-dependent** (`A,B,C,Δ` are functions of the token). **WaveLM's REF has neither**
— its conv, mixing, and phase are fixed filters. Information is delivered by a fixed LTI
transport and combined by a fixed quadratic readout; the only data-conditioning is a
radial magnitude gate. This is precisely the regime that scales *worst*:

```
   per-parameter scaling exponent α (bpb ≈ E∞ + A·N^−α), from doc 11:
      byte-GPT (content-based attention) :  α ≈ 0.90      ████████████  steep
      WaveLM   (fixed LTI + |·|² readout):  α ≈ 0.39      █████         shallow
```

A fixed linear filter bank has limited compositional capacity *per parameter*; adding
parameters to it (wider `d`, higher rank) gives diminishing returns — exactly the shallow
exponent observed. **The thing "in the way" is not a wrong wave detail; it is the absence
of the one mechanism (data-dependent mixing) that lets attention and Mamba convert
parameters into compositional power.**

### 4.5 Throughput, not just quality (context)

WaveLM is 10–31× slower and 5–12× heavier than the byte-GPT (doc 11) — a Python loop over
`n_sub` and complex64 arithmetic. This caps *experiments per hour*, not bpb, but it is a
real limit on iteration speed and on the matched-wall-clock comparison.

### 4.6 The one driver we cannot ablate — and must not forget

The **embedding read by a contrastive softmax** is the inherited learning engine (doc
08: "the one mechanism that demonstrably drove learning" in the predecessor). It cannot
be ablated without deleting the model, but it is 40–99 % of the parameters and is the
source of the learning gradient (pull `W_E[o*]` toward the state, push competitors away).
It is *foundational*, not decorative — the ablations above all assume it and measure what
the **state-builder** around it contributes. The finding is that the best state-builder
parts are generic (nonlinearity, mixing, depth), not wave-specific.

---

## 5. Load-bearing vs. decorative — the evidence ledger (updates doc 10 §5)

```
   component                 measured effect of removing it        verdict
   ───────────────────────────────────────────────────────────────────────────────
   embedding + contrastive   (not ablatable; the learning engine)  FOUNDATIONAL
     readout                  doc 08; 40–99% of params
   per-channel saturation    +0.74 (d256) / +0.66 (d512)           LOAD-BEARING (#1)
   channel mixing (M / UVᴴ)  +0.49 (d256) / +0.46 (d512)           LOAD-BEARING (#2)
   depth (n_sub), to ~8–12   +0.10 cutting 12→4                    LOAD-BEARING (to optimum)
   complex-valued repr.      +0.20 (real twin, matched DOF)        REAL, MODEST
     • via complex conv/mix   ~0.18 of the 0.20                     (the actual carrier)
     • explicit carrier ω     +0.019 (d256) → +0.008 (d512)        WEAK, shrinks w/ scale
   dilated causal conv       K=0 ⇒ zero context (test_info_prop)   LOAD-BEARING (necessary
                                                                     but parametrically tiny)
   energy normalisation      removing it ⇒ vanishing grad at depth DEPTH-ENABLING
                              (doc 04; cures decay) ≈ complex RMSNorm  (stability, not capacity)
   Born |·|² readout         −0.03 (d256) / −0.04 (d512): BETTER   DECORATIVE / mildly HARMFUL
                              without it (on real text)              for performance
   dispersion / "wave" lens  (interpretive; generated the design)  A LENS, not a driver
```

Honest caveats: (a) **TinyStories, fixed 41 M-byte budget** — undertrained, short-range,
attention-friendly; the Born square and saturation could trade places on harder/longer
data or at convergence. (b) The Born square is still required (with *some* readout
nonlinearity) for the **modular-addition composition gate**; "decorative" here means
*for next-byte bpb on natural text at these scales*, not "provably useless everywhere".
(c) `real_twin` removes phase *and* uses real arithmetic *and* a real readout — it
measures the **package** value of complex, not a single knob; it has no muP, so it was
explicitly lr-swept to stay fair (§6).

---

## 6. Are we using the key primitive properly? (the carrier, and a tuning trap)

- **The carrier `ω` is under-worked, not mis-worked.** It barely trains and contributes
  little; the model keeps a maximally-spread fixed spectrum and learns everything else
  around it (STATUS §3c). The spectral-diversity guard built to protect it has never
  fired and is, on this evidence, unnecessary machinery. If we want the carrier to earn
  more, it likely has to become **data-dependent** (a per-token frequency shift) — see §7.
- **A tuning trap to record (rigor).** The real-valued twin has **no muP**, so its
  optimal lr does *not* transfer across width. At d=512 (width 1024) the muP lr 3e-3
  *diverged* to 3.71 bpb; an lr sweep was required to get the fair number (REF-style
  complex variants transfer fine via muP). The fair, lr-tuned real-twin penalty is
  reported in §2; the diverged value is **not** the complex-vs-real result.

---

## 7. The hidden lever — data-dependence (the path worth pursuing)

§4.4 predicts that the missing ingredient is **data-dependent routing**. The campaign
tested the *minimal* version: a `selective` gate that makes the conv self-retention
input-dependent, `s[n]=σ(|ψ[n]|²·W_in·W_out+b)`, modulating `β·ψ` (Mamba-style
selectivity at ~3 % extra params). Crucially it is compared against a **parameter-matched
non-selective control** (rank bumped so the control has *more* DOF than the selective
model), so any win is from *data-dependence*, not from extra parameters:

```
                     d=256                      d=512
   selective        1.4453  (ΔREF −0.017)      1.3175  (ΔREF −0.023)
   matched control  1.4550  (ΔREF −0.007)      1.3370  (ΔREF −0.003)
   ─────────────────────────────────────────────────────────────────
   selectivity gain  −0.010 (beyond params)     −0.020 (beyond params)
   (selective − control)        ↑ and it GROWS with scale (0.010 → 0.020) ↑
```

> A **single** data-dependent gate on **one** term already beats its parameter-matched
> control, and the margin **doubles** from d=256 to d=512. The effect is small in this
> minimal form (near the d=256 seed noise; d=512 is single-seed and wants replication),
> but its *direction and scale-trend* are exactly what the LTI-bottleneck hypothesis
> predicts: data-dependence converts parameters into compositional power, and matters
> more as the model grows. **This is the most promising lead toward closing the
> scaling-exponent gap with attention.**

### Prioritised directions (most-grounded first)

```
   1. RICHER data-dependence (the lever): make the dynamics input-conditioned beyond a
      single self-gate — data-dependent K (selective convolution), data-dependent mode
      coupling, and/or a data-dependent carrier shift Δω(token). Target: lift α from
      0.39 toward attention's 0.90. (Measured seed for this: §7's +0.01→+0.02, growing.)
   2. UNTIE the propagation operator across steps (untie_steps): −0.08 measured, +0.4%
      params. Cheapest confirmed win; do it by default and scale it (per-step low-rank M).
   3. DROP the Born square (linear readout): −0.03→−0.04 measured, free; revisit only if
      a composition-gate or hard-data regime needs the readout nonlinearity back.
   4. INVEST parameters where they pay: capacity is embedding+mixing (§1.1); scale rank /
      mixing and the (now data-dependent, untied) dynamics — not the inert carrier.
   5. RIGHT-SIZE depth to the budget (§4.2): n_sub ≈ reach (⌈log₂T⌉), not more.
```

Note these are **architecture changes**, not metaphor changes. The wave lens produced a
working model; the path forward is to give that model the **data-dependent routing** it
structurally lacks — which is a move *away* from a pure wave (an LTI medium) toward a
*nonlinear, input-steered* dynamical system. The honest framing: the next paradigm step
is not "more wave," it is "**a selectively-steered field**".

---

## 8. Conclusion — answering the question directly

- **What fundamentally drives WaveLM?** A pointwise **nonlinearity** (the saturation,
  #1), **channel mixing** (#2), the **embedding-and-contrastive-readout** learning engine
  (foundational), and **depth** — all generic. Complex-valued representation adds a real
  but modest ~0.2 bpb, mostly through the complex conv/mixing.
- **Are we using that primitive properly?** Partly. The dynamics are parameter-starved
  and tied (untying helps, −0.08); the carrier `ω` is under-worked; depth is mis-sized
  beyond its optimum.
- **What is decorative or weak?** The **Born `|·|²` readout** (mildly harmful for bpb),
  the explicit **carrier `ω`** (weak, shrinking with scale), and the **dispersion/wave
  interpretation** (a generative lens, not a measured driver). The spectral-diversity
  guard is unused machinery.
- **Is something in the way?** Yes — the Born square (a little), over-deep `n_sub` (a
  little), and, fundamentally, the **absence of data-dependent routing** (a lot): WaveLM
  is an LTI medium, which is exactly the architecture class with the shallow per-parameter
  scaling we measure (α≈0.39 vs 0.90).
- **Is there a hidden lever / paradigm path?** Yes — **data-dependence**. A minimal
  selective gate already beats its parameter-matched control and the margin grows with
  scale. Making the propagation/mixing input-steered (selective conv, data-dependent
  coupling, token-conditioned carriers), untying the per-step kernels, and dropping the
  Born square is the evidence-backed route toward Transformer-class scaling.

> **Bottom line.** The "wave" was a *fertile metaphor that produced a working
> architecture*, but it is **not** the engine of its intelligence. The engine is the
> ordinary trinity of (nonlinearity + mixing + a contrastively-trained embedding),
> iterated with depth. WaveLM's path to being paradigm-shifting runs through the one
> thing its current physics forbids — a **data-dependent, selectively-steered** field —
> not through more faithful wave mathematics.

---

*Evidence: `checkpoints/ablation/ablation_results.json` (38 matched runs + the
real-twin lr re-tune, TinyStories byte-level, 41 M bytes/run, 2026-05-24). Instruments:
`wave/{config,field,model,real_model}.py`, gated by `tests/test_wave_ablations.py`
(defaults bit-identical; FP64 gradient checks). Method/driver:
`scripts/ablation_study.py`, `scripts/analyze_ablation.py`. Related: doc 01 (why a
wave), 08 (the inherited driver), 09 (the loss/Born), 10 (conv as propagation — §5
ledger updated here), 11 (the Transformer scaling gap this explains).*
