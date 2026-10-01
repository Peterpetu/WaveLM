# The Convolution Is the Wave-Propagation Operator — Mathematical, Theoretical, Empirical

> Thesis: WaveLM's dilated causal convolution is **not** a substrate or a decorative
> "neural-net layer." It is the **propagation operator of a complex wave field** — the
> single mechanism that carries information across sequence positions. This document
> proves that three ways: **mathematically** (it has an exact dispersion relation and a
> receptive-field theorem), **theoretically** (it makes the model a multi-mode
> waveguide), and **empirically** (remove it and context provably vanishes). It also
> states honestly **where the wave theory is load-bearing and where it is looser**.
>
> Grounded in `wave/field.py` (`causal_dilated_conv`), `wave/config.py` (`dilation`,
> the receptive-field assertion), `02_dynamics_and_gradients.md` (gradients), and
> `tests/test_wave_*.py` (information propagation, causality, init invariants).

---

## 0. TL;DR

```
   The conv is the ONLY operation in WaveLM that reads ψ at a position other than n.
   Everything else (saturation, phase rotation, energy-norm, mode-coupling, readout)
   is per-position. So the conv = "how the wave travels". Delete it → each position
   evolves alone → the model can't even be a bigram (verified: propagation = 0).
```

| Question | Verdict |
|---|---|
| Fundamental or decoration? | **Fundamental** — the only position coupling; ablation kills all context |
| Connected to wave theory? | **Yes, exactly** — it is the discretized propagation/transport operator; with the phase rotation it yields a real dispersion relation and a multi-mode waveguide |
| Where is "wave" load-bearing? | complex values (phase → interference + dispersion), dilation (group velocity/reach), causality (one-way transport); these *generated* the design |
| Where is it looser? | the kernel is freely learned (a physical medium would constrain it); the limiter is a dissipative, not conservative, nonlinearity; energy is *enforced* by renorm, not conserved by the dynamics |

---

## 1. The operator (exactly what runs)

From `wave/field.py`, one internal step `τ` computes, **per channel `k`**:

```
   p[n,k] = β · ψ[n,k]  +  Σ_{j=1}^{Δ}  K[j,k] · ψ[n − j·D_τ , k]          (ψ[m]=0 for m<0)
            └ retain ┘     └──── pull past positions, each via a COMPLEX tap K[j,k] ────┘
```

with dilation `D_τ = 2^(τ mod L_cyc)` (`wave/config.py:dilation`). Three properties,
each of which we will use:

```
   • DEPTHWISE : channel k convolves only with itself (no cross-channel mixing here).
   • CAUSAL    : reads only PAST positions n − j·D_τ  (zero-padded at the start).
   • COMPLEX   : K[j,k] ∈ ℂ → each tap SCALES amplitude (|K|) and ROTATES phase (arg K).
```

A single step, drawn (Δ=2 taps, dilation D):

```
 position:   … n-2D      n-D        n
 field ψ:        •         •         ?
                  \         \        |
            K[2,k] \  K[1,k] \      β·       ← complex taps: attenuate + phase-shift
                    \         \      |
                     ►─────────►─────▼
        p[n,k] = β·ψ[n,k] + K[1,k]·ψ[n-D,k] + K[2,k]·ψ[n-2D,k]
               = a SUPERPOSITION of past field values arriving at position n
```

That word — *superposition of past values arriving at n* — is the entire point. The
conv is the rule by which the field at one place is built from the field elsewhere.
That is what "propagation" means.

---

## 2. Mathematical

### 2.1 It is the *only* thing that couples positions

Walk the forward pass (`wave/field.py:wave_step`) and ask of each op: *does it read ψ at
a position ≠ n?*

```
   dilated conv .......... reads n, n−D, n−2D, …      ← YES (the propagation)
   mode-coupling M ....... reads channels at the SAME n   (no)
   saturation p/(1+g|p|²)  per (n,k)                      (no)
   phase rotation e^{iω} . per (n,k)                      (no)
   energy-norm √d·r/‖r‖ .. per position n (reduces over k)(no)
   Born readout |⟨W_E,ψ⟩|² per position n                (no)
```

So **all** cross-position information flow is the conv. (Stated in
`02_dynamics_and_gradients.md` §1: it is "the only position coupling.") This already
settles "decoration vs fundamental" — but the *why it's a wave* needs the next parts.

### 2.2 The dispersion relation (the heart of the wave claim)

A "wave medium" is precisely a system that imposes a relationship between how fast a
disturbance oscillates **in space** and **in time** — a *dispersion relation* `Ω(κ)`.
Let us derive WaveLM's. Take the **linearized** dynamics (small amplitude, so the
saturation `g·|p|²→0`; the energy-norm is then a global per-step rescale, handled in
§2.5). One step on channel `k` is the conv followed by the carrier rotation:

```
   ψ^{(τ+1)}[n] = e^{i ω_k} · ( β·ψ^{(τ)}[n] + Σ_j K[j,k]·ψ^{(τ)}[n − j·D] )
```

Probe it with a **plane wave** — a single mode of spatial frequency `κ` and
computation-time frequency `Ω`:

```
   ψ^{(τ)}[n] = A · e^{ i ( κ n − Ω τ ) }
                        │        │
              spatial freq κ   time freq Ω
              (wiggle along n)  (wiggle along the internal steps τ)
```

Two shift identities fall out of the ansatz:

```
   ψ^{(τ)}[n − jD] = ψ^{(τ)}[n] · e^{ − i κ j D }       (a spatial shift = a phase factor)
   ψ^{(τ+1)}[n]    = ψ^{(τ)}[n] · e^{ − i Ω }           (a time   shift = a phase factor)
```

Substitute and cancel the common `ψ^{(τ)}[n]`:

```
   ┌─────────────────────────────────────────────────────────────┐
   │   e^{−iΩ}  =  e^{i ω_k} · ( β + Σ_{j=1}^{Δ} K[j,k] e^{−iκ jD} )│
   │              └ω┘           └──────── Ĥ_k(κ) ─────────────────┘ │
   └─────────────────────────────────────────────────────────────┘
```

`Ĥ_k(κ) = β + Σ_j K[j,k] e^{−iκ jD}` is just the **discrete Fourier transform of the
conv kernel** — the filter's frequency response. Splitting into phase and magnitude:

```
   Ω_k(κ) = − ω_k − arg Ĥ_k(κ)        ← DISPERSION RELATION (phase advance per step)
   |Ĥ_k(κ)|                            ← per-step gain (see §2.5: renorm pins it to 1)
```

This is exact, and it is exactly what a wave physicist would write for a discrete,
dispersive transport line. Read it:

- the carrier `ω_k` sets a baseline phase advance for the mode,
- the conv's response `arg Ĥ_k(κ)` bends it **frequency-dependently** — i.e. different
  spatial wiggles travel at different rates: **dispersion**, the defining property of a
  wave medium (versus a non-dispersive delay line).

### 2.3 Group velocity — why dilation *is* propagation speed

How fast does a localized wave-packet of mode `k` actually move along the sequence per
internal step? That is the **group velocity**:

```
   v_g(κ) = dΩ_k/dκ = − d[ arg Ĥ_k(κ) ] / dκ          (positions travelled per step)
```

Because the dilation `D` sits inside `Ĥ_k(κ) = β + Σ_j K[j,k] e^{−iκ jD}`, scaling `D`
stretches the response in `κ` and **scales how far the packet hops per step**. Larger
`D` ⇒ longer hops ⇒ faster reach. So the *dilation schedule is literally the
propagation-speed schedule*: small `D` early (fine, local mixing), exponentially larger
`D` later (long jumps).

### 2.4 Receptive-field theorem — one cycle reaches the whole sequence

Now the "reach" math, made precise. A single step at dilation `D_τ` lets information hop
up to `Δ·D_τ` positions. Chaining steps over **one dilation cycle** `τ = 0 … L_cyc−1`
with `D_τ = 2^τ`, the maximum distance a chain of hops can carry information is the sum:

```
   reach(one cycle) = Σ_{τ=0}^{L_cyc−1} Δ · 2^τ = Δ · (2^{L_cyc} − 1)
```

With `L_cyc = ⌈log₂ T⌉` we have `2^{L_cyc} ≥ T`, hence `reach ≥ Δ·(T−1) ≥ T` for `Δ≥1`.

> **Theorem (full receptive field in O(log T) steps).** After one dilation cycle of
> `L_cyc = ⌈log₂ T⌉` steps, position `n`'s field depends on **all** positions `≤ n`.

This is asserted in code (`wave/config.py:__post_init__`):
`rf_one_cycle = delta*((1<<L_cyc)-1)`, with `assert rf_one_cycle >= T`. The wavefront,
drawn as the support of an impulse started at position 0 (taking `Δ=1`, `D_τ=2^τ` for a
clean trace):

```
   τ (D=2^τ)   positions:  0  1  2  3  4  5  6  7  …
   start                   ●
   0  (D=1)                ●  ●                         reach 2   touched {0,1}
   1  (D=2)                ●  ●  ●  ●                    reach 4   touched {0..3}
   2  (D=4)                ●  ●  ●  ●  ●  ●  ●  ●         reach 8   touched {0..7}
   …                                                    after ⌈log₂T⌉ steps: all of it
```

The wavefront **doubles its reach every step** — geometric, not linear. That is why
WaveLM gets a full-sequence receptive field cheaply, and it is the explicit cure for the
predecessor's fault F2 ("effective context ≈ 1 symbol"; see `08_…paradigm_shift.md`).

### 2.5 Energy, gain, and a lossless line

From §2.2, a mode's per-step gain is `|Ĥ_k(κ)|`, which is **not** generally 1 — a freely
learned filter can amplify or attenuate, and empirically the un-normalised dynamics
*decayed* (`|Ĥ|<1` on average) until the field — and the gradient — vanished at depth.
The **energy normalisation** `ψ⁺[n] = √d · r[n]/‖r[n]‖` fixes this: for a plane wave,
`‖r[n]‖` is constant in `n`, so the renorm is a global per-step rescale that forces the
*effective* gain to `1`. In wave language: **it makes the line lossless** (energy
conserved per step), so modes propagate without decaying.

At initialisation the kernel is set so the medium is non-expansive *before* renorm too:
`tests/test_wave_field.py::test_init_norm_invariants` checks `β + Σ_j |K[j,k]| = 1` per
channel, i.e. the DC gain `|Ĥ_k(0)| ≤ β + Σ|K| = 1`. So WaveLM is *born* a
marginally-stable (non-growing) propagation medium and is *kept* exactly lossless by the
renorm. (Honest caveat: see §5 — conservation is *imposed*, not emergent.)

---

## 3. Theoretical — WaveLM as a multi-mode waveguide

Assemble the pieces and the wave picture is not a metaphor but a structural reading:

```
   complex field ψ[n,k]    =  amplitude  AND  phase   at position n, in mode k
   dilated causal conv     =  TRANSPORT: carry the field forward along n (one-way)
   phase rotation e^{iω_k} =  CARRIER advance: mode k oscillates at its own frequency
   depthwise (per-channel) =  modes propagate INDEPENDENTLY (a linear waveguide)
   saturation p/(1+g|p|²)  =  weak NONLINEARITY: coupling becomes amplitude-dependent
   energy-norm ‖ψ‖=√d      =  LOSSLESS medium (no net gain/decay per step)
   Born readout |⟨W_E,ψ⟩|² =  DETECTOR: observe the field by its intensity
```

Each channel `k` is a **mode**: its own carrier `ω_k` and its own transfer `Ĥ_k(κ)`,
hence its own dispersion `Ω_k(κ)`. A bank of independent dispersive modes that interfere
at readout is exactly a **multi-mode waveguide**. The (optional) mode-coupling matrix
`M` is the waveguide's *mode coupler* — it lets energy hop between modes (cross-channel),
the only place that happens during evolution.

The division of labour is clean and worth holding onto:

```
   CONV  delivers distant bytes into position n's field   (necessary: no conv → nothing to combine)
   PHASE + INTERFERENCE + SQUARING then COMPOSE them       (sufficient for beating the bigram ceiling)
```

So the conv is *necessary but not sufficient*: it is the delivery mechanism; interference
is what is done with the delivered goods. Both are fundamental; they do different jobs.

---

## 4. Empirical

### 4.1 Remove the conv → context provably disappears

`tests/test_wave_model.py::test_information_propagation` perturbs the **first** byte and
measures whether the prediction at the **last** position changes:

```
   perturb byte o₀  ─────────────────────────►  Δ at the last position's logits?

   full model (K ≠ 0):   | Δ ln p(· | T−1) |  >  1e-4      ✔ information reached the end
   ablation   (K = 0):   | Δ ln p(· | T−1) |  =  0.0       ✘ ZERO propagation
```

With the conv off (`K=0`, `β` only), `p[n] = β·ψ[n]`: every position is a function of its
own byte alone. The model cannot represent even a bigram — it degenerates to a function
of the current byte. The conv is *what makes context exist*.

### 4.2 Propagation is one-way (causal)

`tests/test_wave_field.py::test_causality`: changing a **future** byte `o_m` (`m > n`)
leaves `ℓ(·|n)` **bit-identical**. The conv only reads `n − j·D` (the past), so the wave
travels strictly forward — a *retarded* (causal) propagator. This is what makes `p(·|n)`
a valid autoregressive next-byte distribution.

### 4.3 The modes stay distinct, and the cost is tiny

- **Modes stay separated.** Across training runs, `Var(ω)` barely moves (e.g. on the
  Pile `3.2899 → 3.2898`) and the spectral guard never fires — the carrier frequencies
  remain spread, so the dispersion relations of the different modes stay distinct (the
  waveguide does not collapse to one mode). See `05_results.md` / `STATUS.md`.
- **Propagation is cheap.** With `Δ=2` taps the conv is a couple of shifted
  multiply-adds per step; throughput is latency-bound (the per-step Python loop), not
  conv-FLOP-bound — propagation buys full-sequence reach at negligible compute.
- **It works.** The model learns long-range, register-appropriate text (coherent
  TinyStories prose at 1.36 bpb; technical English on the Pile) — behaviour impossible
  for the no-propagation (K=0) degenerate model.

### 4.4 A hand-traceable propagation example

Linearised, single tap (`Δ=1`), real `β=0.5, K=0.5`, impulse `ψ⁰ = [1,0,0,0,…]`. Track
the magnitude reaching each position (ignoring rotation/renorm, which don't change the
*support*):

```
   τ=0, D=1:  p[n]=0.5 ψ[n] + 0.5 ψ[n-1]
              ψ¹ ≈ [0.5, 0.5, 0,   0,   …]          (position 1 now lit: the front moved)
   τ=1, D=2:  p[n]=0.5 ψ[n] + 0.5 ψ[n-2]
              ψ² ≈ [0.25,0.25,0.25,0.25,0, …]       (positions 2,3 now lit: front at 3)
   τ=2, D=4:  ψ³ lights positions 0..7              (front at 7)
```

The disturbance spreads forward, the front doubling each step — a discrete wavefront.
(In the real model the complex `K` and `e^{iω}` also rotate each contribution's phase, so
what spreads is a *phase-carrying* wave that can interfere when fronts overlap.)

---

## 5. Honest ledger — where "wave" is load-bearing vs looser

Per the project's "no unverified claims" rule, the wave reading is not uniformly tight.

**Load-bearing (the wave theory does real work and *generated* the design):**

```
   • COMPLEX values      → carry phase → enable interference AND a real dispersion
                           relation. A real-valued conv is just a filter: no phase,
                           no interference, no Ω(κ). This choice is wave-essential.
   • DILATION            → sets group velocity / reach (§2.3-2.4). The wave must
                           travel; geometric dilation is HOW it travels far, fast.
   • CAUSALITY           → one-way (retarded) propagation; makes p(·|n) autoregressive.
   • The DESIGN itself   → "store history as a superposition of oscillating modes, let
                           them interfere, read by |·|²" produced these exact choices
                           (complex, dilated, squared readout). A generic-conv mindset
                           would not single them out.
```

**Looser (wave-*inspired* engineering, not physics-derived):**

```
   • The kernel K is FREELY LEARNED. A physical medium constrains its stencil (e.g. a
     Laplacian); WaveLM's is a general complex FIR filter — MORE expressive than a
     literal wave (a learned dispersion, not a fixed physical one).
   • The nonlinearity p/(1+g|p|²) is a DISSIPATIVE limiter chosen for gradient
     stability — not the conservative Kerr term i·γ|ψ|²ψ of a physical nonlinear wave.
   • ENERGY is ENFORCED by renormalisation, not conserved by the dynamics. (§2.5 shows
     this is exactly what makes the line lossless — the right wave property, imposed by
     fiat rather than emerging from a Lagrangian.)
   • Arithmetically, a complex depthwise dilated causal conv is a KNOWN primitive
     (WaveNet/TCN/state-space kin). The novelty is its ROLE — the propagation operator
     of a complex field read by Born intensity — and the coherent program around it,
     not the op in isolation.
```

So: **WaveLM is a *learned, engineered* wave, not a literal physical one** — but the
convolution genuinely *is* its propagation operator, with an exact dispersion relation
and a proven receptive-field theorem, and the wave principle is what produced the design.

---

## 6. Verdict

```
   Substrate?    No more than "the engine" is a substrate of a car.
   Decoration?   No — it is the only position coupling; delete it and context = 0 (tested).
   Fundamental?  Yes — one of the three pillars (PROPAGATION), alongside the carrier/
                 interference and the Born readout.
   Wave theory?  Real and load-bearing: complex transport + carrier ⇒ a multi-mode
                 dispersive waveguide with an exact Ω_k(κ); dilation is its group
                 velocity; the renorm makes it lossless. A few pieces are wave-inspired
                 engineering rather than physics-derived (see §5) — stated honestly.
```

## 7. Cheat-sheet

```
   operator       p[n,k] = β ψ[n,k] + Σ_j K[j,k] ψ[n − jD_τ , k]      (depthwise, causal, complex)
   role           the ONLY cross-position operation = wave propagation/transport
   dispersion     e^{−iΩ} = e^{iω_k}(β + Σ_j K[j,k] e^{−iκjD})  ⇒  Ω_k(κ) = −ω_k − arg Ĥ_k(κ)
   group velocity v_g(κ) = −d[arg Ĥ_k(κ)]/dκ   (dilation D scales it: bigger D, faster reach)
   reach          one cycle Σ_τ Δ·2^τ = Δ(2^{L_cyc}−1) ≥ T   ⇒ full receptive field in ⌈log₂T⌉ steps
   lossless       energy-norm forces |Ĥ_eff| = 1  (init: β + Σ|K| = 1)
   empirical      K=0 ⇒ propagation exactly 0 (test_information_propagation);
                  future bytes never leak (test_causality)
```

*Code: `wave/field.py` (`causal_dilated_conv`, `wave_step`), `wave/config.py`
(`dilation`, receptive-field assertion). Math: `02_dynamics_and_gradients.md`. Tests:
`tests/test_wave_field.py`, `tests/test_wave_model.py`. Related: `01_concept_and_theory.md`
(why a wave), `08_…paradigm_shift.md` (F2 lineage), `09_loss_function_explained.md`
(the Born readout that detects the propagated field).*
