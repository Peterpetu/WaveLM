# WaveLM — Dynamics and Gradients (as-built mathematics)

Complete mathematics of the implemented model. Inner product is Hermitian,
conjugate-linear in the first slot: `⟨a,b⟩ = Σ_k ā_k b_k`. Field `ψ ∈ ℂ^{B×T×d}`.

## 1. Forward (one internal step τ → τ+1)

Dilation `D_τ = 2^(τ mod L_cyc)`, `L_cyc = ⌈log₂T⌉`.

**(a) Dilated causal convolution** (depthwise; the only position coupling):
```
p[n,k] = β·ψ[n,k] + Σ_{j=1}^{Δ} K[j,k]·ψ[n − j·D_τ, k]      (ψ[m]=0 for m<0)
```

**(b) Optional mode coupling** (every `mix_every` steps): `p[n,:] ← M·p[n,:]`,
`M ∈ ℂ^{d×d}`.

**(c) Per-channel saturation** (load-bearing nonlinearity):
```
q[n,k] = p[n,k] / (1 + g·|p[n,k]|²)
```

**(d) Phase rotation**: `r[n,k] = e^{i ω_k}·q[n,k]`.

**(e) Energy normalisation**: `ψ⁺[n] = √d · r[n] / ‖r[n]‖`,
`‖r[n]‖ = (Σ_k |r[n,k]|²)^{1/2}`.

Readout after `N_sub` steps:
```
a(o,n) = ⟨W_E[o], ψ[n]⟩ ,   ℓ(o|n) = |a(o,n)|²/T_o ,   p(o|n)=softmax_o ℓ ,
L = − (1/(B(T−1))) Σ_{b,n<T−1} ln p(o_{n+1}|n) .
```

## 2. Local Jacobians (Wirtinger; descent uses `g_z := ∂L/∂z̄`)

**Readout → field.** `s(o,n) := p(o|n) − 1[o=o_{n+1}]`,
```
g_ψ[n,k] = ∂L/∂ψ̄[n,k] = Σ_o s(o,n)·(a(o,n)/T_o)·W_E[o,k] .
```
**Embedding** gets gradient from the readout and the injection ψ⁰=W_E[o_n]:
```
g_{W_E}[o,k] += Σ_n s(o,n)·(conj(a(o,n))/T_o)·ψ[n,k]   (readout)
g_{W_E}[v,k] += Σ_{n: o_n=v} g_{ψ⁰}[n,k]              (injection, after full backprop)
```
**Normalisation (e).** With `ρ = ‖r[n]‖`, `ψ⁺[n] = √d·r[n]/ρ`. For incoming `G=g_{ψ⁺}`:
```
g_r[n,k] = (√d/ρ)·( G[n,k] − (Re Σ_j Ḡ[n,j] r[n,j]) · r[n,k]/ρ² )
```
(the radial component is removed — this is the projection Jacobian that makes the
recurrence norm-preserving and gradient-stable at depth).
**Rotation (d).** `g_q[n,k]=e^{−iω_k} g_r[n,k]`;
`∂L/∂ω_k = −Σ_n 2 Im( conj(g_r[n,k]) · r[n,k] )`.
**Saturation (c)**, per-channel `u=|p[n,k]|²`, `q=p/(1+gu)`:
```
∂q/∂p  = 1/(1+gu)²              (note: (1+gu) − g|p|² = 1)
∂q/∂p̄  = −g p²/(1+gu)²
g_p[n,k] = conj(∂q/∂p)·g_q[n,k] + (∂q/∂p̄)·conj(g_q[n,k])
∂L/∂g  += Σ 2 Re( conj(g_q[n,k])·(−p[n,k]·u/(1+gu)²) )
```
**Mode coupling (b).** `p'=pM`: `g_p = g_{p'} M^ᴴ`; `g_M += Σ_n p[n]^ᴴ g_{p'}[n]` (per
the complex matmul adjoint).
**Conv (a).** `g_ψ[m,k] = conj(β) g_p[m,k] + Σ_j conj(K[j,k]) g_p[m+jD_τ,k]`;
`∂L/∂β += Σ 2 Re(conj(g_p[n,k]) ψ[n,k])`;
`g_{K[j,k]} += Σ_n conj(ψ[n−jD_τ,k]) g_p[n,k]`.

Parameter maps (`β=σ(θ_β)`, `g=softplus(θ_g)`, `T_o=softplus(θ_T)`,
`ω=π·tanh(θ_ω)`) contribute their scalar derivatives by the chain rule. All
implemented via PyTorch complex autograd; the closed forms above are the
finite-difference test oracle (`tests/test_wave_*`, rel-err < 1e-5 in FP64).

## 3. Stability

Step (e) gives `‖ψ⁺[n]‖ = √d` exactly ⇒ field energy constant for all τ≥1, any
depth: no decay, no blow-up. Empirically this removed the vanishing gradient that
occurred at `n_sub ≥ ~12` under the earlier non-normalised dynamics.

## 4. Conditional spectral-diversity term

`R_spec = −(d² − |Σ_k e^{iω_k}|²)/(d(d−1)) ∈ [−d/(d−1), 0]`,
`∂R_spec/∂ω_k = −(2/(d(d−1))) Σ_{j≠k} sin(ω_j−ω_k)`. Added as `λ_ω·R_spec` only
when `Var(ω)` collapses below `ρ·Var₀` for `patience` validations (default off).
