# WaveLM's Loss Function — A Visual, From-Scratch Walkthrough

> This file teaches the WaveLM loss end to end, building the math up from scratch
> with ASCII pictures. It is grounded in the actual code: `wave/readout.py`
> (`born_logits`, `ce_loss`, `bits_per_byte`) and `wave/train.py` (the full
> objective `L = L_CE + λ_ω·R_spec`). The gradient math is from
> `02_dynamics_and_gradients.md`. Read top to bottom; each section assumes only the
> ones before it.

---

## 0. The whole thing in one picture

A language model's job is: *given the bytes so far, predict the next byte.* The
**loss** is one number that says how badly it did, so that "make the number smaller"
= "predict better." Here is the entire pipeline, for a single position `n`:

```
   ψ[n] ∈ ℂ^d                 W_E[o] ∈ ℂ^d   (one learned vector per byte o = 0..255)
   the wave-state at           the "key" for
   position n  (a complex      candidate byte o
   vector of d numbers)
        \                         /
         \        overlap        /
          ▼                     ▼
   a(o,n) = ⟨W_E[o], ψ[n]⟩            ← a COMPLEX number per byte:
        │                              "how strongly does the wave at n
        │                               resonate with byte o's key?"
        ▼   square the magnitude  (the Born rule)
   ℓ(o,n) = |a(o,n)|² / T_o           ← a REAL score ≥ 0, one per byte
        │
        ▼   softmax over all 256 bytes
   p(o|n) = exp(ℓ_o) / Σ_o' exp(ℓ_o') ← a PROBABILITY distribution over the next byte
        │
        ▼   look up the TRUE next byte o*  and take −ln
   loss_n = − ln p(o* | n)            ← "surprise" at the truth, in nats
        │
        ▼   average over every position of every sequence in the batch
   L_CE = mean_n loss_n               ← the number we minimise
        │
        ▼   ÷ ln 2
   bpb = L_CE / 0.6931                ← the same number, in bits per byte
```

Everything below unpacks one arrow at a time, teaching the math as we go.

---

## 1. What a "loss" is (the ground rule)

Training = nudging the model's parameters so a **loss function** `L(θ)` gets smaller.
`L` must be:
- a single real number (so we can say "smaller is better"),
- differentiable (so we know which way to nudge — see §10),
- and a faithful proxy for what we want (here: good next-byte predictions).

WaveLM's loss is **cross-entropy**: the average *surprise* the model feels when it
sees the true next byte. Confident-and-right → ~0 surprise. Confident-and-wrong →
huge surprise. We minimise average surprise.

```
   want:  L small  ⇔  model is rarely surprised by the truth  ⇔  good predictions
```

---

## 2. The two ingredients: the wave-state ψ and the byte keys W_E

After the wave dynamics have run (`n_sub` internal steps; see `04_as_built.md`), every
position `n` holds a **complex vector** `ψ[n] = (ψ[n,0], …, ψ[n,d−1]) ∈ ℂ^d`. And every
one of the 256 bytes `o` has a learned **complex embedding** `W_E[o] ∈ ℂ^d`. Both live
in the same `d`-dimensional complex space.

### Detour — complex numbers (the only number system this loss uses)

A complex number is `z = a + b·i`, where `i² = −1`. Picture it as an arrow in a plane:

```
        imaginary
           ↑
        b ·- - - - • z = a + b·i
           |       ╱:
           |      ╱ :
           |     ╱  :     length:     |z| = √(a² + b²)        (Pythagoras)
           |    ╱   :     conjugate:  z̄  = a − b·i  (mirror across the real axis)
           |   ╱    :     |z|² = z·z̄ = a² + b²       (length, squared)
           |  ╱ θ   :
           +-+------+----→ real
           0        a
```

Three facts we will use:
- **magnitude** `|z| = √(a²+b²)` — the arrow's length (always ≥ 0).
- **conjugate** `z̄ = a − b·i` — flip the sign of the imaginary part.
- **`|z|² = a² + b²`** — length squared; this is the Born "intensity."

Why complex at all? Each channel `k` is a little **wave** (carrier frequency `ω_k`); a
complex number stores both its *amplitude* (length) and its *phase* (angle θ). Phase is
how waves **interfere** — the thing that lets WaveLM compose information. (See doc 01.)

---

## 3. Arrow 1 — the overlap `a(o,n)`: a complex inner product

```python
# wave/readout.py
a = torch.einsum("btk,vk->btv", psi, W_E.conj())     # a(o,n) = Σ_k conj(W_E[o,k])·ψ[n,k]
```

For each candidate byte `o`, we compute the **Hermitian inner product** of its key with
the wave-state:

```
   a(o,n) = ⟨W_E[o], ψ[n]⟩ = Σ_{k=0}^{d−1}  conj(W_E[o,k]) · ψ[n,k]      ∈ ℂ
```

### Detour — what an inner product means

The inner product measures **alignment** between two vectors. For ordinary real
vectors, `⟨u, v⟩ = Σ_k u_k v_k = |u||v|cos(angle)` — big and positive when they point
the same way, zero when perpendicular. The complex (Hermitian) version conjugates the
first vector so that `⟨v,v⟩ = Σ|v_k|² = ‖v‖²` stays a real, non-negative length.

```
   ψ[n]  ────────────────►            "how much of ψ lies along W_E[o]?"
         ╲                            a(o,n) is that (complex) amount.
          ╲   a(o,n) = projection
           ╲  of ψ onto byte o's key      large |a|  → wave RESONATES with byte o
            ▶ W_E[o]                       small |a|  → wave is indifferent to byte o
```

So `a(o,n)` is one complex number per byte: *the resonance between the wave at position
`n` and byte `o`'s key.* There are 256 of them (one per byte) — the vector of "raw
affinities" before we turn them into a probability.

Because the field is energy-normalised (`‖ψ[n]‖ = √d`) and the embeddings are
unit-energy at init, each `a(o,n)` is an `O(1)` number — neither vanishing nor
exploding. That normalisation is what keeps this whole readout well-behaved at depth.

---

## 4. Arrow 2 — the Born rule `ℓ = |a|² / T_o`: square the magnitude

```python
# wave/readout.py
mag2 = a.real**2 + a.imag**2          # |a(o,n)|²
return mag2 / T_o                     # ℓ(o,n)
```

We turn each complex resonance `a(o,n)` into a real **score** by squaring its length and
dividing by a temperature `T_o`:

```
   ℓ(o,n) = |a(o,n)|² / T_o = ( a_re² + a_im² ) / T_o          ∈ ℝ≥0
```

```
   a = a_re + i·a_im                       imaginary
                                              ↑
   |a|² = a_re² + a_im²                  a_im•- - •  a
        = (the squared length             |    ╱:
           of the arrow)                  |   ╱ :   |a|² is the SHADED square's area
                                          |  ╱  :        of the arrow's length
                                          | ╱   :
                                          +-+---+--→ real
                                            a_re
```

**Why square the magnitude?** Three reasons, all load-bearing:

1. **It makes a valid score.** `a` can be negative or complex; `|a|² ≥ 0` is always a
   clean non-negative "how much byte `o`" number.
2. **It is a genuine nonlinearity.** `|a|²` is *quadratic* in `ψ` (degree 2), and since
   `ψ` itself is built from the embeddings, the conditional ends up **degree-≥4** in the
   embeddings. A model whose scores are merely *linear* in the embeddings can only
   express pairwise (bigram) statistics; squaring is exactly what lets WaveLM represent
   **compositions** (it passes the modular-addition gate where a linear readout is stuck
   at `ln K`). See doc 01.
3. **It is the Born rule.** In wave physics and quantum mechanics, the observable
   intensity / probability is the **squared amplitude**: `probability ∝ |amplitude|²`.
   The readout reads the wave the way a detector reads light — by its intensity.

### Detour — the temperature `T_o`

`T_o > 0` (a learned scalar, `softplus(θ_To)`) divides the scores before the softmax:

```
   small T_o  →  scores spread out  →  sharper, more confident distribution
   large T_o  →  scores squashed    →  flatter, more uncertain distribution
```

It is a single knob that lets the model calibrate how decisive its probabilities are.

---

## 5. Arrow 3 — softmax: scores → a probability distribution

We have 256 real scores `ℓ(0,n), …, ℓ(255,n)`. The **softmax** turns them into
probabilities that are all ≥ 0 and sum to 1:

```
                       exp( ℓ(o,n) )
   p(o | n) = ───────────────────────────
                 Σ_{o'} exp( ℓ(o',n) )
```

### Detour — `exp` and why softmax works

`exp(x) = e^x` is always positive and grows fast. So:
- `exp(score)` makes every score positive (needed for a probability), and
- dividing by the sum makes them total 1 (a real distribution),
- and because `exp` is steep, a slightly larger score becomes a *much* larger
  probability — "winner takes much."

```
   raw scores ℓ:                 after softmax:  p (sums to 1)
   byte A  ██     1.0            byte A  ██          0.21
   byte B  ██     1.0            byte B  ██          0.21
   byte C  ████   2.0            byte C  █████████   0.58
                  └ exp() blows up the gap, then we normalise ┘
```

Now `p(·|n)` is the model's actual prediction: "I think the next byte is C with
probability 0.58, A or B with 0.21 each."

---

## 6. Arrow 4 — cross-entropy: `−ln p(true byte)`

The data tells us the **true** next byte `o*`. The loss at this position is simply how
*surprised* the model is to see it:

```
   loss_n = − ln p(o* | n)
```

### Detour — logarithms, surprise, and "nats"

Information theory measures the **surprise** of an event of probability `p` as `−ln p`:

- a sure thing (`p = 1`) is no surprise: `−ln 1 = 0`,
- a coin-flip (`p = 0.5`) costs `−ln 0.5 = 0.693` nats (= exactly **1 bit**),
- an event the model ruled out (`p → 0`) is infinitely surprising: `−ln p → ∞`.

(A "nat" is the natural-log unit of information; `1 bit = ln 2 = 0.693 nats`.)

```
  loss = −ln p(true)
  loss
   4 │•
     │ •                          confident & WRONG
   3 │  •                         (model gave the truth tiny p)  → huge loss
     │   ••
   2 │     ••
     │       •••
   1 │          ••••
     │              •••••••              confident & RIGHT
   0 │                     ••••••••••••  (model gave the truth p≈1) → ~0 loss
     └────┬─────┬─────┬─────┬─────┬──→  p(true byte)
        0.0   0.2   0.4   0.6   0.8  1.0
```

So minimising `−ln p(o*|n)` *directly* pushes the model to put **probability mass on
the byte that actually comes next**. That is the whole training signal.

> "Cross-entropy" = the average of `−ln p(true)` over the data. Minimising it is the
> same as maximising the likelihood the model assigns to the real text.

---

## 7. Arrow 5 — the shift, and averaging over everything

A subtlety that the code makes precise (`wave/readout.py:ce_loss`): the prediction at
position `n` is graded against the byte at position `n+1`.

```python
pred = logits[:, :-1, :]      # positions 0 … T−2   (drop the last: it has no "next")
tgt  = targets[:, 1:]         # bytes    1 … T−1     (shifted left by one)
F.cross_entropy(pred, tgt, reduction="mean")
```

```
   position n:   0    1    2    3    4
   bytes:      [ h    e    l    l    o ]
   logits ℓ(·|n):0    1    2    3            (position 4 has no target → unused)
                 │    │    │    │
                 ▼    ▼    ▼    ▼
   graded vs:    e    l    l    o            (the byte at n+1)
```

This shift is what makes `p(·|n)` a valid **autoregressive next-byte** distribution:
because the wave dynamics are *causal* (position `n` only ever reads `n' ≤ n`),
`ψ[n]` — and therefore `p(·|n)` — depends only on the past. Finally we average the
per-position surprise over every predicted position of every sequence in the batch:

```
                1
   L_CE = ───────────── Σ_b Σ_{n=0}^{T−2}  [ − ln p(o_{n+1} | n) ]      (nats / byte)
            B · (T−1)
```

One scalar. That is the number `.backward()` differentiates.

---

## 8. Arrow 6 — bits per byte (the reported metric)

```python
# wave/readout.py
bits_per_byte = ce_nats / math.log(2.0)
```

Cross-entropy comes out in **nats** (natural log). Dividing by `ln 2 = 0.693` converts
to **bits**: `bpb = L_CE / ln 2`. Bits-per-byte is the honest, tokenizer-independent
yardstick — "how many yes/no questions, on average, to pin down the next byte."

```
   bits/byte   (lower = better)
   8.0  ████████████████  random guess (uniform over 256 bytes = log₂256)
   4.5  █████████         byte unigram (just letter frequencies)
   2.5  █████             byte bigram (letter pairs)         ← the ceiling AIFS hit
   1.36 ███               WaveLM on TinyStories (+ mode coupling)
   1.0  ██                a strong char/byte LM
   0.0                    perfect (always knows the next byte)
```

(`bpb` is just a rescaling of `L_CE`; the thing being minimised is identical.)

---

## 9. The full objective — and the one optional extra term

The complete training loss (`wave/train.py`) is:

```
   L = L_CE  +  λ_ω · R_spec
```

`L_CE` is everything above and is **the entire loss in normal operation**, because
`λ_ω = 0` by default. `R_spec` is a *conditional safety net* for the carrier
frequencies `ω_k`, switched on **only** if they collapse together:

```
   R_spec = − ( d² − |Σ_k e^{iω_k}|² ) / ( d(d−1) )        ∈ [−d/(d−1), 0]
```

### Detour — what this term watches

Each channel is a wave at frequency `ω_k`. Their power to **interfere** (and thus to
compose features) comes from being *different*. Plot each `ω_k` as a point `e^{iω_k}` on
the unit circle:

```
   healthy — frequencies SPREAD          collapsed — frequencies CLUSTERED
          •   •   •                                   •••
        •           •                                ••••
       •             •                                •
        •           •                          (all nearly equal)
          •   •   •
   Σ e^{iω_k} ≈ 0   (they cancel)         |Σ e^{iω_k}| ≈ d   (they add up)
   R_spec ≈ −1  (diverse: good, no push)   R_spec ≈ 0  → the guard activates and
                                            the −R_spec gradient pushes them apart
```

`Var(ω)` is monitored every step; `λ_ω` flips from `0` to a small value only if the
variance stays collapsed for a couple of validations (the `SpectralGuard`), and flips
back off when health returns. In every run so far the frequencies stayed spread on
their own, so this term **never fired** — it is insurance, not part of the working
objective.

---

## 10. WHY the loss teaches — the contrastive gradient

A loss only matters through its **gradient**: the direction in parameter space that
*most decreases* `L`. Training repeatedly takes a small step downhill
(`θ ← θ − η·∂L/∂θ`). So the real question is: *what does `∂L/∂(stuff)` tell each part of
the model to do?* For the readout it is beautifully simple. Define the **error signal**

```
   s(o,n) = p(o|n) − 1[o = o_{n+1}]        (predicted prob, minus 1 for the true byte)
```

Then (from `02_dynamics_and_gradients.md` §2) the gradient on the byte keys is

```
   ∂L/∂W_E[o]  ∝  Σ_n  s(o,n) · (conj a(o,n)/T_o) · ψ[n]
```

Read it as a **pull / push** on each byte's key:

```
   for the TRUE next byte o*:   s = p(o*) − 1 < 0   →   −s > 0  →  PULL  W_E[o*] toward ψ
   for every OTHER byte o:      s = p(o)       > 0   →            PUSH  W_E[o]  away from ψ

                W_E[wrong] ◄──┐         ψ[n]  (the wave-state)
                              │          •
                       push   │         ╱ ╲
                       away   │        ╱   ╲   pull toward
                              ▼       ╱     ◄─ W_E[o*]   (the true next byte's key)
```

In words: **make the true next byte resonate *more* with this wave-state, and make the
competitors resonate *less*.** Do that across all the data and each byte's key gets
dragged into alignment with the wave-states that actually precede it — the embedding
**learns the byte co-occurrence geometry**. (The magnitude `a(o,n)` and the temperature
`T_o` just weight how hard each pull/push is.)

This contrastive pull/push is *exactly* the mechanism that drove the predecessor model
(see doc 08); WaveLM keeps it verbatim and squares it (the Born rule), then feeds it a
far richer wave-state `ψ` so the keys finally have something worth aligning to.

### Detour — gradients of complex numbers (Wirtinger), lightly

Our parameters are complex, so "the derivative" needs care. The trick (Wirtinger
calculus) is to treat `z` and its conjugate `z̄` as independent variables; the quantity
that points downhill is `∂L/∂z̄`, and PyTorch's autograd stores exactly that in
`z.grad`. You do not have to compute these by hand — autograd does — but every closed
form in doc 02 is checked against finite differences to `< 1e-5` (FP64), so the engine
is verified, not assumed.

---

## 11. Numerical stability (why it never overflows)

Two places could blow up; both are handled:

- **`exp` in the softmax.** `exp(big score)` overflows. The standard fix is to subtract
  the max score first: `exp(ℓ_o − max) / Σ exp(ℓ_o' − max)` — mathematically identical,
  numerically safe. PyTorch's `F.cross_entropy` does this internally (it fuses
  `log_softmax + nll` in a stable log-sum-exp form), which is exactly why `ce_loss`
  calls it instead of writing `−ln(exp/Σexp)` by hand.
- **The field magnitude.** Energy normalisation pins `‖ψ[n]‖ = √d`, so `a(o,n)` and the
  logits `|a|²/T_o` stay `O(1)` at any depth — the readout can neither vanish nor
  explode. (This was an empirically forced fix; see `04_as_built.md` §3.)

---

## 12. A tiny worked example (numbers you can check by hand)

Take `d = 2` channels, `V = 3` candidate bytes, `T_o = 1`. Suppose the wave-state and
the three byte keys are:

```
   ψ[n]    = [ 1 + 0i ,  0 + 1i ]            (note ‖ψ‖ = √(1+1) = √2 = √d ✓ normalised)
   W_E[0]  = [ 1 + 0i ,  0 + 0i ]
   W_E[1]  = [ 0 + 0i ,  1 + 0i ]
   W_E[2]  = [ 1 + 0i ,  1 + 0i ]
```

**Step 1 — overlaps** `a(o) = Σ_k conj(W_E[o,k])·ψ[n,k]`:

```
   a(0) = conj(1)·(1+0i) + conj(0)·(0+1i) = 1 + 0i        |a(0)|² = 1
   a(1) = conj(0)·(1+0i) + conj(1)·(0+1i) = 0 + 1i        |a(1)|² = 1
   a(2) = conj(1)·(1+0i) + conj(1)·(0+1i) = 1 + 1i        |a(2)|² = 1² + 1² = 2
```

Byte 2's key best "lines up" with the wave (it has support on both channels), so it
gets the biggest intensity.

**Step 2 — Born logits** `ℓ = |a|²/T_o = [1, 1, 2]`.

**Step 3 — softmax** `exp([1,1,2]) = [2.718, 2.718, 7.389]`, sum `= 12.826`:

```
   p = [ 2.718/12.826 , 2.718/12.826 , 7.389/12.826 ] = [ 0.212 , 0.212 , 0.576 ]
```

**Step 4 — cross-entropy** for two possible truths:

```
   if true byte o* = 2:  loss = −ln(0.576) = 0.552 nats  =  0.552/0.693 = 0.796 bits
   if true byte o* = 0:  loss = −ln(0.212) = 1.551 nats  =  2.238 bits
```

The model is *less* surprised when the truth is byte 2 (the one it favoured) — exactly
as a good loss should behave.

**Step 5 — the gradient's instruction** (`s(o) = p(o) − 1[o=o*]`, with `o* = 2`):

```
   s = [ 0.212 , 0.212 , 0.576 − 1 ] = [ +0.212 , +0.212 , −0.424 ]
        push 0 away   push 1 away    pull 2 toward ψ  (strongest move: it was the truth)
```

Next step, `W_E[2]` is nudged toward `ψ[n]`, and `W_E[0], W_E[1]` away — so on the next
forward pass byte 2 resonates *even more* and its probability rises. Multiply this over
millions of positions and the embedding geometry self-organises around what actually
follows what.

---

## 13. One-screen cheat-sheet

```
   a(o,n) = ⟨W_E[o], ψ[n]⟩            complex resonance of wave n with byte-key o
   ℓ(o,n) = |a(o,n)|² / T_o           real score ≥ 0   (Born rule: intensity = |amp|²)
   p(o|n) = softmax_o ℓ(o,n)          probability distribution over the next byte
   L_CE   = mean_n [ −ln p(o_{n+1}|n) ]   average surprise at the true next byte (nats)
   bpb    = L_CE / ln 2               same number, in bits/byte
   L      = L_CE + λ_ω·R_spec         full objective (λ_ω = 0 by default)

   learning signal:  s(o,n) = p(o|n) − 1[o=o_{n+1}]
       → pull the TRUE byte's key toward ψ, push the others away (contrastive update)
```

- The loss is **cross-entropy**: minimise average surprise = maximise the probability
  assigned to the real text.
- The **Born square** `|·|²` is the one piece beyond a plain bilinear readout; it makes
  the model nonlinear enough to compose (escape the bigram ceiling) and gives a true
  probability-from-amplitude readout.
- Everything is **causal** and **energy-normalised**, so `p(·|n)` is a valid
  autoregressive distribution with `O(1)`, stable logits at any depth.
- The spectral term is **off unless the carrier frequencies collapse** — insurance, not
  part of the day-to-day objective.

*Code: `wave/readout.py` (`born_logits`, `linear_logits` (ablation), `ce_loss`,
`bits_per_byte`), `wave/train.py` (`L = L_CE + λ_ω·R_spec`, `spectral_repulsion`,
`SpectralGuard`). Gradients: `02_dynamics_and_gradients.md`. Why the readout is the
learning engine: `08_reflection_driver_and_paradigm_shift.md`.*
