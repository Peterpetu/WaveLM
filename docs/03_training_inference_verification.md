# WaveLM — Training, Inference, Verification (as-built)

## Training (`wave/train.py`)

- **Loss:** mean next-byte cross-entropy `L_CE` (predict byte n+1 from the field
  at position n). Optional conditional spectral-diversity term `λ_ω·R_spec`,
  default off (activates only if `Var(ω)` collapses; see 02_dynamics §4).
- **Optimiser:** AdamW (`β1=0.9, β2=0.95, wd=0.01, eps=1e-8`), cosine
  warmup-decay LR (`warmup 5%`, peak `3e-4`, floor `3e-5`), global grad-clip `1.0`.
- **Precision:** complex64 (fp32 real/imag) throughout; activation checkpointing
  every `⌈√n_sub⌉` internal steps to bound backward memory.
- **Data:** byte windows sampled at random offsets from a memory-mapped text file
  (`wave/data.py`), no tokenizer.
- **Logging (bits-per-byte):**
  `Step n/N [WaveLM] | bpb=… CE=… | tok/s … grad=… GPU=…G lr=… Var(w)=… lam_w=… ETA=…`,
  plus a JSONL metrics file (train + periodic validation), checkpoints
  (`best.pt`, `step_*.pt`) with optimiser state for **resume**.

## Inference (`wave/infer.py`)

Streaming, O(1)/byte in context length. A cache holds, for every committed
position, its full internal stack `{ψ^(τ)}_{τ=0..n_sub}`. A new byte computes its
stack level-by-level, reading cached past stacks at the dilated offsets; then the
Born readout gives the next-byte distribution (sample / temperature / top-k /
greedy). Verified **bit-identical** to the parallel training forward, including
with dense and low-rank mode coupling (`test_*_stream_equals_parallel`).

## Verification (`tests/test_wave_*.py`, all green)

- **Identity:** energy/norm invariance (‖ψ[n]‖=√d ∀τ≥1), causality (future bytes
  cannot change past logits), init invariants (`β+Σ|K|=1`, `E‖W_E[o]‖²=1`),
  saturating-nonlinearity Jacobian vs finite diff.
- **Gradient:** full-parameter central finite differences in FP64 (W_E, K, ω, β,
  g, T_o, and dense/low-rank mixing), rel-err < 1e-5.
- **Reductions / expressivity:** bigram reduction (K=0,n_sub=1,g→0 → squared
  bilinear bigram, matches closed form); **modular-addition composition gate**
  (full model CE≈0; linear-readout ablation pinned at ln K) — the go/no-go that
  proves escape from the bigram ceiling.
- **Dynamics:** information propagation across the full sequence (cure for the
  predecessor's one-hop limit) vs a no-conv ablation (exactly zero propagation);
  representation separation (decoded direction ≠ current-byte embedding).
- **System:** streaming == parallel (bit-exact); byte-packing roundtrip; spectral
  guard activation logic; generation variety; smoke-train (CE decreases, no NaN).

## Empirical results (TinyStories byte-level, RTX 4080 16 GB)

- Base config d=384, T=1024, n_sub=16, batch=96, ~59k tok/s, ~12 GB.
- Validation bpb: 3.96 (step 250) → 2.71 (500) → 2.37 (750) → 2.21 (1000),
  crossing the byte-bigram barrier (~2.5) by step 750 and still descending.
- Generations at step 1000 are English story-like text (names, casing, sentence
  structure) — qualitatively far beyond the predecessor's degenerate output.
- Reference points (bpb): uniform 8.0; byte-unigram ~4.5; byte-bigram ~2.5; a
  competent char/byte LM ~1.0–1.3.
