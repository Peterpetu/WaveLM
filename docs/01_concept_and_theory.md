# WaveLM — Concept and Theory

## The one empirical fact we built on

A real predecessor model (AIFS) was trained and measured. Stripped of its
elaborate machinery, exactly one mechanism demonstrably drove its learning: a
**bilinear contrastive readout over a learned embedding**, `⟨state, W_E[o]⟩`,
turned into a distribution by a softmax. Its gradient is the classic
contrastive-embedding update; it aligned the embedding geometry to the symbol
co-occurrence graph. Everything else the predecessor carried (a four-term
free-energy loss, a one-step stochastic field) either did no measurable work or
actively fought that readout, and the model stalled between unigram and bigram
quality because its "state" was, after one update, just a rotation of the current
symbol's embedding — it could not represent *what comes next* as distinct from
*what is now*, and it could not propagate context.

WaveLM keeps that one working mechanism verbatim (as the Born readout
`|⟨W_E[o], ψ⟩|²`) and replaces the failed state-building machinery with something
designed, from first principles, to do the two things the predecessor could not:
**(i) carry a representation that differs from the current symbol, and
(ii) compose information across many past symbols.**

## Why a wave

We needed a state that (a) stores a long history in a fixed-size object, (b)
combines new inputs with old ones *nonlinearly* (so the model is not stuck in the
linear-in-embeddings / bigram class), and (c) does so cheaply at depth. Looking
across the most physically distant systems that solve "store a history and read it
out" — a struck bell, ripples on water, AM/FM radio, a hologram, heat diffusion —
one structure recurs: **store the sequence as a superposition of oscillating
components; let new inputs interfere with old ones; read out by squared
magnitude.** Identity lives in *frequency*, time in *phase*, combination is
*interference* (multiplicative in phase, hence nonlinear), and observation is
`|·|²`. That is a wave. So WaveLM's state is a complex field `ψ`, each channel a
carrier at its own frequency `ω_k`; tokens are injected as amplitudes; a dilated
causal convolution propagates them; a saturating nonlinearity + phase rotation
make them interfere; and the Born rule reads the next byte.

## Why it escapes the bigram ceiling (the theory)

A model whose logits are linear in the embeddings can only express pairwise
(`symbol_i`–`candidate`) affinities — bigram-class statistics. Composition tasks
(e.g. modular addition `c=(a+b) mod K`) are *linearly inseparable*: the answer
depends on a product/Fourier feature of two inputs, not a sum of single-input
features. WaveLM forms exactly those features: complex carriers are Fourier
features by construction, the per-channel saturation creates `|p|²` products, and
the Born readout squares overlaps — so the conditional becomes a degree-≥4
function of the embeddings. The decisive empirical check (Phase D): on data drawn
from a modular-addition law, WaveLM reaches CE≈0 while a linear-readout ablation is
pinned at the bigram floor `ln K`. This is the property the predecessor lacked.

## Why it does not decay or blow up (theory → practice)

A wave that only propagates and interferes tends to either diverge or dissipate.
We make the dynamics **energy-conserving**: after each internal step the field is
renormalised to fixed per-position energy (`‖ψ[n]‖ = √d`). This is also the
physically correct statement that `ψ[n]` is a *normalised state*, which is what
makes `|⟨W_E[o], ψ⟩|²` a genuine Born probability. Practically, it was essential:
without it, at the depth needed for a full receptive field the field decayed to
zero and the gradient vanished. With it, depth is free and stable.

## Where "thinking time" and "context length" come from

Two axes are deliberately separated: *sequence position* `n` and *computation
time* `τ`. The field evolves `n_sub` internal steps along `τ`, propagating along
`n` by convolution with **geometrically growing dilation** `D_τ = 2^(τ mod L_cyc)`.
One dilation cycle reaches `≥ T`, so a single forward pass has full-sequence
receptive field — unlike the predecessor, whose one-hop update gave an effective
context of one symbol. Depth (composition) and length (reach) are thus decoupled,
and neither is tied to a memory-linear axis (which is what made the predecessor
run out of memory when it tried to "think longer").

## What is novel vs. borrowed

Borrowed (because it empirically works): the contrastive embedding readout.
Everything else — a complex carrier field, injection-not-anchoring of tokens,
interference as the composition primitive, energy-normalised wave dynamics, the
Born readout as the output law, dilated propagation as the reach mechanism, and
optional waveguide mode-coupling for channel capacity — is built from the wave
first-principles above rather than copied from an existing architecture. There is
no attention, no FFN, no gating cell, no tokenizer.
