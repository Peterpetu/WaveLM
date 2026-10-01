# WaveLM — As-Built Specification

This documents the model **as implemented** in the `wave/` package. It supersedes
the original master-plan math where the two differ; every divergence is the
result of an empirical finding during implementation (per the project's
"redesign whatever is needed" directive). The master plan
(`hello-welcome-to-the-velvet-canyon.md`) holds the motivation, the
creative-reasoning derivation, and the verification philosophy.

## 1. Object

A complex byte-level language model. Alphabet = 256 raw bytes (no tokenizer).
State: a complex field `psi[b, n, k] ∈ ℂ`, batch `b`, position `n ∈ [0,T)`,
channel `k ∈ [0,d)`. Parameters: byte embedding `W_E ∈ ℂ^{256×d}`, causal conv
kernel `K ∈ ℂ^{Δ×d}`, carrier frequencies `ω ∈ ℝ^d`, scalars `β,g,T_o`, and an
optional mode-coupling matrix `M ∈ ℂ^{d×d}`.

## 2. Forward pass

Injection: `psi^(0)[n] = W_E[o_n]`.

For `τ = 0 … n_sub−1`, with dilation `D_τ = 2^(τ mod L_cyc)`, `L_cyc = ⌈log₂T⌉`:

1. **Dilated causal conv** (per channel):
   `p[n,k] = β·psi[n,k] + Σ_{j=1}^{Δ} K[j,k]·psi[n − j·D_τ, k]`  (zero-padded, causal).
2. **Mode coupling** (optional, every `mix_every` steps): `p[n] ← M · p[n]`.
3. **Per-channel saturation** (the load-bearing nonlinearity):
   `q[n,k] = p[n,k] / (1 + g·|p[n,k]|²)`.
4. **Phase rotation**: `r[n,k] = e^{i ω_k}·q[n,k]`.
5. **Energy normalisation** (key divergence from plan): `psi^(τ+1)[n] = √d · r[n] / ‖r[n]‖`.

Readout (Born rule): `a(o,n) = ⟨W_E[o], psi^(N_sub)[n]⟩`,
`ℓ(o|n) = |a(o,n)|² / T_o`, `p(o|n) = softmax_o ℓ`. Loss = mean next-byte CE.

## 3. Divergences from the master plan (and why)

- **Per-channel saturation, not mean-intensity.** The plan used
  `q = p/(1+g·mean_k|p|²)` (a per-position scalar). Empirically that is killed by
  the normalisation in step 5 (a pure scalar rescale is cancelled by norming).
  The per-channel form compresses high-magnitude channels more than low ones, so
  it changes the *direction* of `psi[n]` and survives normalisation. Verified to
  still solve the modular-addition composition gate (`CE≈0` vs linear `≈ln K`).
- **Energy normalisation to ‖psi[n]‖ = √d.** The plan relied on the
  nonlinearity alone to bound magnitude. Empirically, at the depth needed for full
  receptive field (`n_sub ≥ L_cyc`, e.g. 16–24), the contractive dynamics decayed
  the field toward zero → uniform logits → **vanishing gradient** (`‖grad‖ → 0`).
  Fixing the per-position energy makes the dynamics energy-conserving (a wave
  with no dissipation), eliminating the decay and the vanishing gradient. It also
  makes `psi[n]` a normalised state, so the Born readout is a genuine probability
  amplitude. The `√d` target (RMS=1 per channel, not unit norm) keeps logits at
  O(1) scale at init so learning has signal from step 1.
- **Mode coupling `M` (optional, new).** Without it channels evolve as
  independent 1-D filters that interact only at the readout. `M` lets modes
  exchange information during evolution (a waveguide mode-coupling analogue),
  raising capacity for many-feature composition. Off by default; init near
  identity; applied every `mix_every` steps to bound the O(B·T·d²) cost.

## 4. What is carried over from the prior (AIFS) model — the empirical driver

The single mechanism that demonstrably drove learning in the prior model was the
**bilinear contrastive readout over a learned embedding** `⟨state, W_E[o]⟩` +
softmax (it aligned the embedding geometry to the byte co-occurrence graph). That
is preserved exactly here as the Born readout `|⟨W_E[o], psi⟩|²` (the same engine,
squared). Everything that did *not* drive learning in the prior model (the
four-term EFE loss whose epistemic term was antagonistic and whose policy/compute
priors were inert; the one-step Langevin field that could not propagate context)
was discarded. The novel parts — complex carriers, interference nonlinearity,
dilated propagation, mode coupling — are what give composition past the bigram
ceiling the prior model hit.

## 5. Stability (revised theorem)

With step-5 normalisation, `‖psi^(τ)[n]‖ = √d` exactly for all `τ ≥ 1`. No decay,
no blow-up, at any depth. (Replaces the plan's `≤ √d/(2√g)` bound, which held but
permitted decay to ~0.)

## 6. Training

Pure next-byte CE + one *conditional* spectral-diversity term `λ_ω·R_spec`
(`R_spec = −(d²−|Σ_k e^{iω_k}|²)/(d(d−1))`), activated only if `Var(ω)` collapses
(monitored each step; off by default). AdamW, cosine warmup-decay, grad-clip 1.0,
complex64 throughout, activation checkpointing every `⌈√n_sub⌉` steps. Logged in
bits-per-byte.

## 7. Inference

Streaming, O(1)/byte in context length: each new position computes its full
internal stack `{psi^(τ)}` from the cached past stacks at dilated offsets. Verified
bit-identical to the parallel forward (`test_stream_equals_parallel`), including
with mode coupling.

## 8. Verification (all green)

`tests/test_wave_*.py`: norm invariance, causality, init invariants, limiter
Jacobian vs finite-diff, full-parameter gradient finite-diff (incl. `M`), bigram
reduction, **modular-addition composition gate** (the go/no-go: full model CE≈0,
linear ablation stuck at ln K), information propagation across the full sequence,
representation separation, streaming==parallel, byte-packing roundtrip, spectral
guard logic, generation variety.
