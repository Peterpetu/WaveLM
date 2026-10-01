# WaveLM — Complete Design and Mathematical Specification

> The "Wave Neural Network." A clean-slate, evidence-driven language model.
> **Decisions (with user):** complex field · **byte-level (V=256, no vocabulary)** ·
> spec-first · pure-CE loss + one *conditional* spectral-diversity term.
> This document is self-contained: every object, equation, gradient, proof,
> algorithm, and test is written out. On approval it is split verbatim into
> `spec/wave_lm/00..09_*.md` and implemented in a new `wave/` package.

---

## Part 0 — Method

User's 6-step protocol. Steps 1–2 (read + reflect) → Part 1. Step 3 (creative
reasoning, manual + explicit) → Part 2. Step 4 (full spec + all mathematics) →
Parts 3–14. Step 5 (propose) = submit this plan. Step 6 (rigorous implementation)
→ Parts 15–17, gated on approval.

**Inclusion rule:** a mechanism is admitted only if (a) forced by a measured fact
about what drives or blocks learning, or (b) the simplest mathematical object that
supplies a property we proved we need. Nothing is included "because papers do it."

---

## Part 1 — Empirical first principles

We trained a real model on TinyStories (BPE-8k, 43M tokens, RTX 4080, 16 GB),
two runs, fully logged (`checkpoints/*/metrics.jsonl`).

### 1.1 The driver that WORKS (keep it)

The output rule was a bilinear readout + softmax:
`logit(o|s_n) = ⟨s_n, W_E[o]⟩/T_o`. Its CE gradient is the contrastive-embedding
update `∂L/∂W_E[o*] −= s_n/T_o`, `∂L/∂W_E[o] += p(o|s_n)·s_n/T_o`. Since the state
began as `s_n ≈ W_E[o_n]`, this aligned the embedding geometry to the
co-occurrence graph — a well-conditioned, high-SNR objective. It produced the fast
CE drop 9.01 → 5.87 nats.
**→ Driver D1: a bilinear contrastive readout over a learned embedding is the
learning engine; it must survive.**

### 1.2 The mechanism that FAILS (three independent faults)

After the model's single internal update,
`s_n ≈ (1+ε)W_E[o_n] + (small)·(o_{n−1} term) + noise`, current-token ~10× the
previous-token term. So `logit(o*) ≈ ⟨W_E[o_n], W_E[o*]⟩/T_o` — "predict whatever
embeds near the current token." Best PPL 355 ⇒ between unigram (CE≈6.5–7) and
bigram (CE≈3.5–4.5); generations were `"and and and…"`. The three faults:

- **F1 read-what-you-wrote** — the decoded vector was dominated by `W_E[o_n]`; no
  room for a "next" that differs from "now."
- **F2 one step of computation** — context propagated ≤1 position/step and only
  one step ran (deeper OOMed); effective context ≈1.
- **F3 no load-bearing nonlinearity** — between input and output the map was
  effectively linear ⇒ conditional confined to linear-in-embeddings = the bigram
  ceiling.

### 1.3 Requirements (derived)

1. **R-SEPARATE** — decoded vector must be free to differ from `W_E[o_n]`. (kills F1)
2. **R-DEPTH** — many cheap computation steps so far-back tokens accumulate, and a
   receptive field ≥ sequence length. (kills F2)
3. **R-NONLINEAR** — a nonlinearity of non-vanishing magnitude on the gradient path
   between embeddings and logits. (kills F3)
4. **R-READOUT** — keep the bilinear contrastive readout + embedding table. (keeps D1)
5. **R-BUDGET** — train on 16 GB; O(1)/token inference in context length.

### 1.4 Why byte-level (V=256), not a vocabulary

BPE *pre-composes* structure (`" the"`, `"ing"` are meaningful units), letting a
bilinear associator score well by association without composing — exactly the
bigram ceiling — and its Zipfian frequencies create a unigram trap (cheap early
win, then plateau). WaveLM is *built for composition*; bytes are not
pre-composed, so the model **must** assemble recent bytes ("mid-word 'the'"),
exercising the depth×interference machinery we add. Bytes are near-uniform, so the
trap is shallow; readout shrinks `V` 8192→256 (32× cheaper), offsetting longer
sequences. **De-risk:** the synthetic-trigram gate (Part 15, Phase D) validates
composition *representation-independently* before any byte run; a BPE diagnostic
remains as fallback.

---

## Part 2 — Creative reasoning (Tier 3, transformative; done manually)

**T1 — rules of the failed model.** Explicit: E1 one real state/position; E2
softmax over bilinear readout; E3 next-token CE; E4 gradient descent. Hidden:
**H1** state anchored to current-token embedding (→F1); **H2** info moves by local
one-hop updates (→F2); **H3** state real-valued (direction in ℝ^d only); **H4**
depth and length share one memory-linear axis (→ the OOM behind F2); **H5** readout
sees magnitude/direction only — no phase.

**T2 — far-then-analogical donors.** Distant domains that "store a history in a
fixed state and read it out": a struck **bell/cavity** (history → superposition of
oscillating modes; identity in *frequency*, time in *phase*); **ripples on water**
(events combine by *interference*, multiplicative in phase ⇒ nonlinear observable);
**AM/FM radio** (many channels coexist by carrier frequency; detector reads
*squared magnitude*); a **hologram** (phase-encoded distributed storage); **heat
diffusion** (state = convolution of past sources with a kernel; cheap in a
frequency basis). The invariant thought across the most distant domains:
**store a sequence as a superposition of oscillating components; let new inputs
interfere with old; read out by squared magnitude.**

**T3 — mutate the hidden rules.** H3→**complex state** `ψ∈ℂ^d` (phase = a free
memory register; enables interference; doubles capacity). H1→**injection not
anchoring** (a token *adds a wave*; the decoded field may point elsewhere ⇒
R-SEPARATE). H2/H4→**split position `n` from computation-time `τ`**; evolve `N_sub`
internal steps along `τ`, propagate along `n` by **(dilated) convolution** so info
crosses the whole sequence; depth decoupled from length and from memory (⇒ R-DEPTH
in budget). H5→**squared-overlap readout** `|⟨W_E[o],ψ⟩|²` (squaring = a genuine,
non-vanishing nonlinearity ⇒ R-NONLINEAR; "detector reads the envelope").

**T4 — candidates.** C1 real heat-diffusion (rejected: linear, real). C2 complex
linear recurrence (close, but linear until readout, depth-1). **C3 complex wave
field, `N_sub` dilated-conv steps + bounded interference nonlinearity, |·|² readout
(selected — meets all five).** C4 2-D space×comp-time PDE (kept as the ≥1B scale-up
variant). **T5 — evaluation:** C3 is minimal, removes F1/F2/F3, keeps D1, fits the
budget, and uses no borrowed block — every piece traces to a T2 thought or a
Part-1 requirement.

---

## Part 3 — Notation and objects

| Symbol | Type / shape | Domain | Meaning |
|---|---|---|---|
| `V` | int | `=256` | byte alphabet (no learned vocabulary) |
| `d` | int | base `256` | complex channel count |
| `T` | int | base `2048` | sequence length in bytes |
| `N_sub` | int | base `24` | internal computation steps per forward |
| `Δ` | int | base `2` | conv taps per step (offsets `1..Δ`, dilated) |
| `L_cyc` | int | `=⌈log₂T⌉=11` | dilation cycle length |
| `D_τ` | int | `2^(τ mod L_cyc)` | dilation at internal step `τ` (fixed schedule) |
| `o_n` | int | `[0,256)` | byte at position `n`, `n=0..T−1` |
| `ψ^{(τ)}` | `ℂ^{T×d}` | — | wave field after `τ` internal steps |
| `W_E` | `ℂ^{256×d}` | — | byte embedding (learned) |
| `K` | `ℂ^{Δ×d}` | — | depthwise causal conv kernel (learned) |
| `ω` | `ℝ^{d}` | `(−π,π]` | per-channel carrier frequency (learned) |
| `β` | `ℝ` | `[0,1)` | self-retention weight (learned) |
| `g` | `ℝ` | `>0` | interference (nonlinearity) strength (learned) |
| `T_o` | `ℝ` | `>0` | readout temperature (learned) |
| `λ_ω` | `ℝ` | `≥0` | spectral-diversity coeff (default 0; conditional) |

Complex inner product (conjugate-linear in the first slot):
`⟨a,b⟩ = Σ_k ā_k b_k`. Per-position L2 norm `‖ψ[n]‖² = Σ_k|ψ[n,k]|²`. Storage:
`torch.cfloat` (complex64); all reductions accumulate in fp32. Real parameters are
stored unconstrained and mapped through Part 7.

---

## Part 4 — Forward pass (complete)

### 4.1 Injection (initial condition)
```
ψ^{(0)}[n,k] = W_E[o_n, k] ,         n=0..T−1,  k=0..d−1.
```

### 4.2 The wave step, for τ = 0,1,…,N_sub−1

**(a) Dilated depthwise causal convolution** (the only position-coupling op):
```
p^{(τ)}[n,k] = β·ψ^{(τ)}[n,k] + Σ_{δ=1}^{Δ} K[δ,k]·ψ^{(τ)}[n − δ·D_τ , k] ,
              with ψ^{(τ)}[m,k] ≡ 0 for m < 0 (causal zero-pad),  D_τ = 2^(τ mod L_cyc).
```

**(b) Intensity-coupled bounded nonlinearity** (load-bearing; couples channels):
```
I^{(τ)}[n] = (1/d) Σ_{k} |p^{(τ)}[n,k]|²                        (per-position intensity)
q^{(τ)}[n,k] = p^{(τ)}[n,k] / (1 + g·I^{(τ)}[n]) .
```

**(c) Per-channel phase rotation** (distinct carriers ⇒ interference):
```
ψ^{(τ+1)}[n,k] = e^{ i ω_k } · q^{(τ)}[n,k] .
```

Read field: `ψ ≡ ψ^{(N_sub)}`.

### 4.3 Readout (Born-style squared overlap) and loss
```
a(o,n)   = ⟨W_E[o], ψ[n]⟩ = Σ_k  conj(W_E[o,k])·ψ[n,k]            ∈ ℂ
ℓ(o|n)   = |a(o,n)|² / T_o                                          ∈ ℝ≥0
p(o|n)   = softmax_o ℓ(o|n) = exp(ℓ(o|n)) / Σ_{o'} exp(ℓ(o'|n))
L_CE     = − Σ_{n=0}^{T−2} ln p( o_{n+1} | n ) .
```
`ψ[n]` depends only on `o_0..o_n` (causal conv), so `p(·|n)` is a valid
autoregressive next-byte distribution. Full loss (Part 9): `L = L_CE + λ_ω·R_spec`.

---

## Part 5 — Stability theorem (magnitude is bounded for all N_sub)

**Theorem.** For any parameters and any `N_sub`, `‖ψ^{(τ)}[n]‖ ≤ √d/(2√g)` for all
`τ ≥ 1`, `n`. (No blow-up; depth is safe.)

**Proof.** Fix `τ,n`. With `u = I^{(τ)}[n] ≥ 0`,
`Σ_k|q^{(τ)}[n,k]|² = Σ_k|p^{(τ)}[n,k]|²/(1+gu)² = (d·u)/(1+gu)²` (using
`Σ_k|p|² = d·u`). The scalar `h(u)=u/(1+gu)²` on `u≥0` has `h'(u)=(1−gu)/(1+gu)³`,
zero at `u=1/g`, giving `h_max = (1/g)/4 = 1/(4g)`. Hence
`Σ_k|q^{(τ)}[n,k]|² ≤ d/(4g)`, i.e. `‖q^{(τ)}[n]‖ ≤ √d/(2√g)`. Rotation (4.2c)
preserves magnitude: `|ψ^{(τ+1)}[n,k]| = |q^{(τ)}[n,k]|`, so
`‖ψ^{(τ+1)}[n]‖ ≤ √d/(2√g)` for every `τ≥0`, i.e. `‖ψ^{(τ)}[n]‖ ≤ √d/(2√g)` ∀τ≥1. ∎

**Corollary (Lipschitz readout).** `|a(o,n)| ≤ ‖W_E[o]‖·‖ψ[n]‖ ≤ ‖W_E[o]‖·√d/(2√g)`,
so logits are bounded given bounded embeddings — softmax cannot overflow (with the
standard max-subtraction). Test: `test_wave_step_norm_bound`.

---

## Part 6 — Gradients (complete, Wirtinger calculus)

Real loss `L`; for complex `z=x+iy` treat `z, z̄` independent; descent uses the
conjugate cogradient `g_z := ∂L/∂z̄` (PyTorch stores `z.grad = g_z`). We give every
local Jacobian; backprop composes them in reverse `τ`.

### 6.1 Readout → field
`a = Σ_k conj(W_E[o,k])ψ[n,k]` (holomorphic in `ψ`, antiholo in `W_E`). With
`ℓ=|a|²/T_o = a ā /T_o`:
```
∂ℓ/∂ψ̄[n,k] = (a/T_o)·W_E[o,k] .            (since ∂a/∂ψ̄=0, ∂ā/∂ψ̄[n,k]=W_E[o,k])
```
CE+softmax: `∂L/∂ℓ(o|n) = p(o|n) − 1[o=o_{n+1}] =: s(o,n)`. Therefore
```
g_ψ[n,k] := ∂L/∂ψ̄[n,k] = Σ_o s(o,n)·(a(o,n)/T_o)·W_E[o,k] .
```

### 6.2 Through one wave step (reverse: ψ^{(τ+1)} → ψ^{(τ)})
Let `G := g_{ψ^{(τ+1)}}` (incoming cogradient).

**(c) rotation** `ψ^{(τ+1)}[n,k]=e^{iω_k}q[n,k]` (holo, factor `c=e^{iω_k}`):
cogradient pulls back by `c̄`:
```
g_q[n,k] = e^{−iω_k}·G[n,k] .
ω-grad (real):  ∂L/∂ω_k += Σ_n 2·Re( conj(G[n,k]) · i · ψ^{(τ+1)}[n,k] )
                          = −Σ_n 2·Im( conj(G[n,k])·ψ^{(τ+1)}[n,k] ).
```

**(b) intensity nonlinearity** `q[n,k]=p[n,k]/(1+gI[n])`, `I=(1/d)Σ_{k'}p p̄`.
Local Jacobian (per position `n`):
```
∂q[n,k]/∂p[n,j]  = δ_{kj}/(1+gI) − (g/d)·p[n,k]·conj(p[n,j])/(1+gI)²
∂q[n,k]/∂p̄[n,j]  =            − (g/d)·p[n,k]·p[n,j]/(1+gI)² .
```
Backprop (the correct real-cogradient contraction; both pieces required because
`q` is non-holomorphic):
```
g_p[n,j] = Σ_k [ conj(∂q[n,k]/∂p[n,j])·g_q[n,k] + (∂q[n,k]/∂p̄[n,j])·conj(g_q[n,k]) ].
g-grad (real):  ∂L/∂g += Σ_{n,k} 2·Re( conj(g_q[n,k])·(−p[n,k]·I[n]/(1+gI[n])²) ).
```

**(a) dilated conv** `p[n,k]=β ψ[n,k] + Σ_δ K[δ,k] ψ[n−δD_τ,k]` (holo, linear):
```
g_ψ^{(τ)}[m,k] = conj(β)·g_p[m,k] + Σ_{δ=1}^{Δ} conj(K[δ,k])·g_p[m+δD_τ, k]   (anti-causal scatter; out-of-range dropped)
∂L/∂β       += Σ_{n,k} 2·Re( conj(g_p[n,k])·ψ^{(τ)}[n,k] )                     (β real)
g_{K[δ,k]}  += Σ_n conj(ψ^{(τ)}[n−δD_τ,k]) · g_p[n,k]                          (complex)
```

### 6.3 Injection → embedding, and readout → embedding
`W_E` gets gradient from (i) the readout (6.1) and (ii) the injection
`ψ^{(0)}[n,k]=W_E[o_n,k]`:
```
readout:    g_{W_E}[o,k] += Σ_n s(o,n)·(conj(a(o,n))/T_o)·ψ[n,k]
injection:  g_{W_E}[v,k] += Σ_{n: o_n=v} g_{ψ^{(0)}}[n,k]
```
(`g_{ψ^{(0)}}` is the result of backprop through all `N_sub` steps.)

### 6.4 Temperature
`ℓ=|a|²/T_o ⇒ ∂L/∂T_o = Σ_{o,n} s(o,n)·(−|a(o,n)|²/T_o²)` (real). Constraint maps
(Part 7) contribute the usual `dσ, d·softplus, d·tanh` factors.

**Non-holomorphic points are only (b) and the `|·|²` readout; both Jacobians are
finite everywhere (`1+gI ≥ 1`).** Implementation uses PyTorch complex autograd;
these closed forms exist solely to gate it with `test_grad_finite_diff` (FP64,
real & imag parts perturbed separately, rel-err < 1e-5) and
`test_limiter_derivative`.

---

## Part 7 — Initialization and parameter maps (no magic numbers)

| Param | Stored | Map to constrained | Init value | Justification |
|---|---|---|---|---|
| `W_E[o,k]` | `x,y∈ℝ` | `W_E=x+iy` | `x,y ~ 𝒩(0, 1/(2d))` | `E‖W_E[o]‖²=1` (unit injection energy) |
| `K[δ,k]` | `x,y∈ℝ` | `K=x+iy` | magnitude `(1−β)·γ^δ/Σ_{δ'}γ^{δ'}` (γ=0.6), phase `~U(0,2π]` | guarantees `β+Σ_δ|K[δ,k]| = 1` ⇒ `C=1` (Part 5 non-growth); geometric tap decay |
| `ω_k` | `θ∈ℝ` | `ω=π·tanh(θ)` | `ω_k = π·(2(k+½)/d − 1)` | deterministic uniform spectral coverage; distinct carriers (anti-collapse at init) |
| `β` | `θ∈ℝ` | `β=σ(θ)` | `0.5` | retain half / propagate half |
| `g` | `θ∈ℝ` | `g=softplus(θ)` | `1.0` | knee at unit intensity; cap `√d/(2√g)` |
| `T_o` | `θ∈ℝ` | `T_o=softplus(θ)` | `1.0` | neutral readout temperature |

Choosing `γ=0.6, Δ=2`: `Σγ^δ = 0.6+0.36 = 0.96`; taps `K∝{0.625,0.375}·(1−β)`.
With `β=0.5`, `Σ|K|=0.5`, total `C=1.0`. Test: `test_init_norm_invariants`.

---

## Part 8 — Receptive field and dilation (why R-DEPTH holds)

A single conv step at dilation `D_τ` reaches back `Δ·D_τ` positions. Stacking
steps `τ=0..N_sub−1` with `D_τ=2^(τ mod L_cyc)` gives, over one full cycle
`τ=0..L_cyc−1`, reach `Σ_{j=0}^{L_cyc−1} Δ·2^j = Δ·(2^{L_cyc}−1) ≥ 2^{L_cyc} ≥ T`
(for `Δ≥1`, `L_cyc=⌈log₂T⌉`). With `N_sub=24 > L_cyc=11` we get >2 full cycles, so
**receptive field ≥ T = 2048** and information from `o_0` can reach the last
position. Test: `test_information_propagation` (perturb `o_0`, require
`|Δ ln p(·|T−1)| > 1e−3` nats). This is the explicit cure for F2.

---

## Part 9 — Training (complete)

### 9.1 Objective
`L = L_CE + λ_ω·R_spec` with `L_CE` from §4.3 and the **conditional**
spectral-diversity term protecting the `ω_k` primitive (§1.3 reasoning):
```
R_spec = − (2 / d(d−1)) Σ_{j<k} ( 1 − cos(ω_j − ω_k) )       ∈ [−1,0]   (frequency repulsion)
∂R_spec/∂ω_k = − (2/d(d−1)) Σ_{j≠k} sin(ω_j − ω_k) .
```
`λ_ω = 0` by default. Instrument `Var(ω) := (1/d)Σ_k(ω_k−ω̄)²` every step;
**activate** `λ_ω ← λ_ω0` (default `0.01`) iff `Var(ω) < ρ·Var₀` for `P` consecutive
validations (`ρ=0.25`, `P=2`, `Var₀` = init variance). Deactivate when
`Var(ω) > 0.5·Var₀`. Bounded (`|R_spec|≤1`), so it cannot crowd out CE. Gated by
`test_spectral_diversity_only_when_collapsing`.

### 9.2 Optimizer, schedule, precision
- **AdamW** (decoupled WD), `β1=0.9, β2=0.95, wd=0.01, ε=1e−8`. (Standard
  first-order; the previous project's pSGLD reduced to this anyway at `T_θ=0`.)
- **LR**: cosine warmup-decay, warmup `5%`, peak `3e−4`, floor `3e−5`.
- **Precision**: complex64 (fp32 real/imag) throughout; the model is <70 MB so
  fp32 is affordable and avoids complex-bf16 gaps.
- **Grad clip**: global L2 norm `1.0`.
- **Activation checkpointing**: checkpoint every `⌈√N_sub⌉≈5` internal steps;
  backward peak `O(√N_sub·T·d)`.

### 9.3 Training step (pseudocode)
```
for batch (B sequences of length T bytes):
    ψ = W_E[bytes]                              # (B,T,d) complex
    for τ in 0..N_sub-1:
        ψ = wave_step(ψ, τ)                      # §4.2, checkpointed
    ℓ = |einsum('btk,vk->btv', ψ, conj(W_E))|² / T_o   # (B,T,V=256)
    L = cross_entropy(ℓ[:, :-1], bytes[:, 1:]) + λ_ω·R_spec(ω)
    L.backward();  clip_grad_norm_(1.0);  adamw.step();  adamw.zero_grad()
    log: bits_per_byte = L_CE/ln(2)/(B·(T-1)); Var(ω); ‖ψ‖; tok/s; GPU; lr; ETA
```
Logging format (user preference): `Step n/N [WaveLM] | bpb=… CE=… | tok/s … grad=… GPU=…G lr=… ETA=…`.

---

## Part 10 — Inference (streaming, O(1)/byte in context length)

The training forward is parallel over `n` (dilated conv). Streaming generation
appends one byte at a time and reuses a cache.

**Cache.** For every committed position `m` store the full internal stack
`{ψ^{(τ)}[m] : τ=0..N_sub}` (shape `T×(N_sub+1)×d`, fp32 complex; ≈105 MB at base,
linear in context — analogous to a KV cache).

**Append byte `o_n` (decode `o_{n+1}`):**
```
ψ^{(0)}[n] = W_E[o_n]
for τ in 0..N_sub-1:
    p = β·ψ^{(τ)}[n] + Σ_{δ=1}^{Δ} K[δ]⊙ ψ^{(τ)}[n − δ·D_τ]      # reads cached past stacks
    I = mean_k |p|²;  q = p/(1+g·I);  ψ^{(τ+1)}[n] = e^{iω}⊙ q
    cache[n][τ+1] = ψ^{(τ+1)}[n]
a = conj(W_E) · ψ^{(N_sub)}[n];  ℓ = |a|²/T_o
o_{n+1} ~ softmax(ℓ)              # or argmax / temperature / top-k as configured
```
Cost per byte `O(N_sub·Δ·d)` (independent of context length) ⇒ R-BUDGET for
inference. Tests: `test_stream_equals_parallel` (streaming logits == parallel
forward, bit-exact fp32), `test_generation_nondegenerate`.

---

## Part 11 — Byte data pipeline (`wave/data.py`)

- Download the same raw TinyStories text already cached locally (reuse text only).
- Encode to raw `uint8`; concatenate with a single `0x00` story separator.
- Pack into non-overlapping windows of `T=2048` bytes ⇒ training tensor
  `(N_windows, 2048) uint8`. Held-out 1% tail as validation.
- No tokenizer, no vocabulary training. Decode for samples: bytes→UTF-8 with
  `errors="replace"` (tolerate a partial multibyte char at a truncation boundary).
- Spec: `spec/wave_lm/09_data.md`. Test: `test_byte_packing_roundtrip`.

---

## Part 12 — Reductions and expressivity (claims + tests)

- **R-bigram (sanity).** Set `Δ=0` (drop conv sum), `N_sub=1`, `g=0`. Then
  `ψ[n]=e^{iω}⊙(βW_E[o_n])` and `ℓ(o|n)=β²|Σ_k conj(W_E[o,k])e^{iω_k}W_E[o_n,k]|²/T_o`,
  a fixed squared-bilinear form in `(o,o_n)` ⇒ a pure bigram. Test
  `test_reduce_to_bigram`: CE within 1% of a directly-fit squared-bilinear bigram.
- **Beyond bigram (the point).** With `Δ≥1, N_sub≥2, g>0`, `ψ[n]` is a bounded,
  intensity-nonlinear, interference superposition of `{W_E[o_m]}_{m≤n}`; the `|·|²`
  readout makes `p(·|n)` a degree-`≥4` function of the embeddings ⇒ strictly
  exceeds linear-in-embeddings. **Gate (Phase D):** on synthetic data drawn from a
  known trigram law that no bilinear model can fit, WaveLM reaches the true entropy
  within 5% while a linear-readout ablation cannot beat the bigram floor.
  (`test_wavelm_fits_trigram`, `test_linear_ablation_cannot_fit_trigram`.)
- **R-SEPARATE.** 2-byte probe: show the decoded direction `ψ[n]/‖ψ[n]‖` has
  `|⟨W_E[o_n], ψ[n]⟩|/(‖W_E[o_n]‖‖ψ[n]‖) < 0.5` after `N_sub` steps (decoded ≠
  current embedding). `test_separation`.

---

## Part 13 — Complexity and memory budget (16 GB)

Base `V=256,d=256,T=2048,N_sub=24,Δ=2`. Per sequence/forward:
- Conv: `N_sub·T·Δ·d = 24·2048·2·256 ≈ 2.6×10⁷` complex MACs.
- Readout: `T·V·d = 2048·256·256 ≈ 1.3×10⁸` MACs.
Both negligible. Memory: field tape `(N_sub+1)·T·d` complex ≈ `25·2048·256·8 B ≈
105 MB`/seq; checkpointed backward peak `O(√N_sub·T·d)` ≈ `21 MB`/seq. Params:
`W_E 256·256` + `K 2·256` + `ω,β,g,T_o` ⇒ **< 5 MB** at `d=256`. ⇒ batch can be
large; the open lever is `d` (try 256→512→1024 nearly free since `W_E` is tiny).
Exact batch fixed by `tools/budget.py` (Phase 0). Test: `test_memory_budget`
(measured peak < 14 GB).

---

## Part 14 — Verification protocol (binding; no test too small)

Tests written **before** the code they cover (failing skeletons in Phase 0).
A phase is done only when its tests + the full suite are green.

| Group | Test | Pass criterion |
|---|---|---|
| identity | `test_wave_step_norm_bound` | `‖ψ^{(τ)}[n]‖ ≤ √d/(2√g)+1e−4` ∀τ≥1 (Part 5) |
| identity | `test_limiter_derivative` | §6.2(b) Jacobian vs FD, rel-err <1e−5 (FP64) |
| identity | `test_readout_normalisation` | `Σ_o p(o|n)=1 ± 1e−9` |
| identity | `test_causality` | changing `o_m,(m>n)` leaves `ℓ(·|n)` bit-identical |
| identity | `test_init_norm_invariants` | `β+Σ_δ|K[δ,k]|=1±1e−6`; `E‖W_E[o]‖²=1±5%` |
| gradient | `test_grad_finite_diff` | all params (W_E,K,ω,β,g,T_o) vs FD, rel-err <1e−5 (FP64) |
| reduction | `test_reduce_to_bigram` | CE within 1% of fitted squared-bilinear bigram |
| expressivity | `test_wavelm_fits_trigram` | ≤ true-entropy·1.05 on synthetic trigram |
| expressivity | `test_linear_ablation_cannot_fit_trigram` | linear ablation ≥ bigram floor |
| dynamics | `test_information_propagation` | perturb `o_0` ⇒ `|Δ ln p(·|T−1)|>1e−3` |
| dynamics | `test_separation` | decoded-vs-current cosine `<0.5` (R-SEPARATE) |
| loss | `test_spectral_diversity_only_when_collapsing` | `λ_ω` activates iff `Var(ω)` collapses |
| system | `test_stream_equals_parallel` | streaming logits == parallel, bit-exact fp32 |
| system | `test_byte_packing_roundtrip` | pack→unpack == original bytes |
| system | `test_memory_budget` | measured peak < 14 GB at base config |
| system | `test_generation_nondegenerate` | unique-byte ratio in first 64 bytes > 0.5 on 10 prompts |
| system | `smoke_train` | 50 steps synthetic: monotone CE↓, no NaN |
| **target gate** | TinyStories byte run | **val < 1.5 bpb by ~10k steps** (byte-bigram barrier ≈2.5 bpb). Stretch ≤1.0 bpb. |

---

## Part 15 — Implementation phases (gated; one concern each)

> Doctrine: tests first; one mechanism per phase; full suite + smoke green at every
> boundary; any regression rolled back and re-analysed, never patched. Entirely new
> `wave/` package; the old `aifs/` is left frozen until WaveLM passes the target
> gate, then archived.

- **Phase 0 — spec + scaffolding (no model code).** Split this document verbatim
  into `spec/wave_lm/00..09_*.md`; write `tools/budget.py` (verifies Part 13);
  write **failing** skeletons for every Part-14 test. *User reviews spec docs.*
- **Phase A — `wave/field.py`: the wave step (§4.2).** Gate: `norm_bound`,
  `limiter_derivative`, `causality`, `init_norm_invariants`.
- **Phase B — `wave/readout.py`: Born readout + CE (§4.3).** Gate:
  `readout_normalisation`, `grad_finite_diff` (readout+W_E), `reduce_to_bigram`.
- **Phase C — `wave/model.py`: full forward (`N_sub` steps + checkpointing).**
  Gate: full `grad_finite_diff`, `information_propagation`, `separation`, `smoke_train`.
- **Phase D — expressivity go/no-go (synthetic).** Gate: `wavelm_fits_trigram`,
  `linear_ablation_cannot_fit_trigram`. **If WaveLM cannot beat the bigram ceiling
  on data engineered to require composition, STOP and rethink before any real run.**
- **Phase E — pipeline.** `wave/data.py` (Part 11), `wave/train.py` (Part 9 logging
  format), `wave/infer.py` (Part 10). Gate: `byte_packing_roundtrip`,
  `stream_equals_parallel`, `memory_budget`, `generation_nondegenerate`.
- **Phase F — TinyStories byte run + analysis.** Train to target gate (<1.5 bpb);
  written analysis (trajectory, ablations, samples, `Var(ω)` history, old-vs-new).
  Miss ⇒ diagnose against R-SEPARATE/DEPTH/NONLINEAR and iterate within the design.
- **Phase G — docs + freeze.** Finalise `spec/wave_lm/*`; archive `aifs/`; write the
  empirical old-vs-new comparison.

---

## Part 16 — Files

New package `wave/`: `config.py`, `field.py`, `readout.py`, `model.py`, `data.py`,
`train.py`, `infer.py`, `__init__.py`. New `spec/wave_lm/00..09_*.md`. New
`tools/budget.py`. New `tests/test_wave_*.py` (one file per Part-14 group). Reused
(raw text only): the cached TinyStories download. Untouched→archived: all of
`aifs/` and its tests.

---

## Part 17 — Risk register

| Risk | Mitigation |
|---|---|
| Complex autograd subtlety | FP64 finite-diff gate on every param (Part 6) before any training. |
| Frequency collapse degrades interference | `Var(ω)` instrumented; conditional `R_spec` (§9.1); `test_spectral_diversity_only_when_collapsing`. |
| Receptive field < T | geometric dilation (Part 8) proven to reach ≥T; `test_information_propagation`. |
| `|·|²` readout conditioning worse than softmax-of-linear | learnable `T_o`; if needed add a real readout scale; `grad_finite_diff` watches it. |
| Per-channel-only dynamics under-expressive | intensity nonlinearity already couples channels; Phase-D synthetic gate decides before any run; dense channel-mix `M∈ℂ^{d×d}` reserved as `08_scaling.md` fallback. |
| Design itself insufficient | Phase D stops us before a costly run; C4 2-D PDE variant is the documented fallback. |
| Scope creep back to old axioms | Part 13 budget + Part 14 gates keep us honest; no term added without a named primitive + measured failure. |

---

## Part 18 — Decisions (resolved) and the one open choice

Resolved with user: **complex field**; **byte-level, no vocabulary (V=256)**;
**spec-first hard gate**; **pure CE + one conditional spectral-diversity term** tied
to the `ω_k` primitive. Open (cosmetic, decide at Phase 0): the **name** — keep
"WaveLM" or pick *Resonance Field Model* / *Interference Net* / *Ripple* / *PSI*.
