"""Fair shared training/eval harness for the WaveLM-vs-Transformer comparison.

Design goal: make the comparison identical-by-construction. Both architectures go
through the SAME loop, the SAME data sampler (same seed => same byte batches), the
SAME loss (wave.readout.ce_loss), the SAME optimiser family (AdamW + decoupled WD),
the SAME cosine schedule, the SAME grad clip, and the SAME fp32 precision. The only
difference is the architecture.

Param budgets are matched on **real degrees of freedom** (a complex WaveLM parameter
= 2 real DOF; Transformer params are real), via count_real_dof + the size-matched
builders. Returns a metrics dict per run (val bpb, wall-clock, tok/s, params).
"""

from __future__ import annotations

import math
import time
from typing import Optional

import torch

from wave.config import WaveConfig
from wave.model import WaveModel
from wave.readout import ce_loss
from wave.train import lr_cosine
from wave.data import ByteData
from baselines.byte_gpt import GPTConfig, ByteGPT


# ---------------------------------------------------------------------------
# Parameter accounting (real DOF; complex counts twice; shared weights once)
# ---------------------------------------------------------------------------
def count_real_dof(model: torch.nn.Module) -> int:
    seen, total = set(), 0
    for p in model.parameters():
        if not p.requires_grad or id(p) in seen:
            continue
        seen.add(id(p))
        total += p.numel() * (2 if p.is_complex() else 1)
    return total


# ---------------------------------------------------------------------------
# Size-matched builders (hit a target real-DOF budget)
# ---------------------------------------------------------------------------
def build_wavelm(target_dof: int, *, n_sub: int = 12, mix_every: int = 2,
                 delta: int = 2, T: int = 1024, seed: int = 0,
                 mup: bool = True) -> WaveModel:
    """WaveLM sized to ~target_dof by scaling width d (rank ~ d/8)."""
    best = None
    for d in range(64, 2049, 32):
        r = max(8, (d // 8 // 8) * 8)            # rank ~ d/8, rounded to multiple of 8
        cfg = WaveConfig(V=256, d=d, T=T, n_sub=n_sub, delta=delta,
                         mix_channels=True, mix_rank=r, mix_every=mix_every,
                         mup=mup, seed=seed, real_dtype="float32")
        dof = count_real_dof(WaveModel(cfg))
        if best is None or abs(dof - target_dof) < abs(best[1] - target_dof):
            best = (cfg, dof)
        if dof > target_dof * 1.25:
            break
    return WaveModel(best[0])


def build_gpt(target_dof: int, *, T: int = 1024, seed: int = 0) -> ByteGPT:
    """Byte-GPT sized to ~target_dof by searching (d_model, n_layer)."""
    torch.manual_seed(seed)
    best = None
    for d_model in (128, 192, 256, 320, 384, 448, 512, 640, 768):
        per_layer = 12 * d_model * d_model       # ~ 3d^2 (qkv) + d^2 (proj) + 8d^2 (mlp)
        n_layer = max(1, round((target_dof - 256 * d_model) / per_layer))
        if n_layer < 1 or n_layer > 24:
            continue
        cfg = GPTConfig(V=256, d_model=d_model, n_layer=n_layer,
                        n_head=d_model // 64, T=T)
        dof = count_real_dof(ByteGPT(cfg))
        if best is None or abs(dof - target_dof) < abs(best[1] - target_dof):
            best = (cfg, dof)
    torch.manual_seed(seed)
    return ByteGPT(best[0])


# ---------------------------------------------------------------------------
# Forward / validate (handle WaveModel's (logits, psi) tuple vs GPT's logits)
# ---------------------------------------------------------------------------
def _forward_logits(model, batch, *, use_checkpoint: bool) -> torch.Tensor:
    if isinstance(model, WaveModel):
        out = model(batch, use_checkpoint=use_checkpoint)
    else:
        out = model(batch)                # GPT (bare logits) or RealWaveModel (tuple)
    return out[0] if isinstance(out, tuple) else out


@torch.no_grad()
def validate_bytes(model, val: ByteData, *, batch: int, device: str,
                   max_batches: int = 40) -> float:
    was_training = model.training
    model.eval()
    tot, nb = 0.0, 0
    for b in val.sequential_batches(batch, max_batches=max_batches, device=device):
        tot += float(ce_loss(_forward_logits(model, b, use_checkpoint=False), b))
        nb += 1
    if was_training:
        model.train()
    return (tot / max(1, nb)) / math.log(2.0)        # bits per byte


# ---------------------------------------------------------------------------
# Optimiser (model-aware: muP groups for WaveLM, standard decay split for GPT)
# ---------------------------------------------------------------------------
def _build_optimizer(model, lr: float, wd: float, betas, eps):
    if isinstance(model, WaveModel) and getattr(model.cfg, "mup", False):
        groups = model.mup_param_groups(lr, wd)
        return torch.optim.AdamW(groups, lr=lr, betas=betas, eps=eps)
    decay = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.requires_grad and p.dim() < 2]
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": wd, "lr_mult": 1.0},
         {"params": no_decay, "weight_decay": 0.0, "lr_mult": 1.0}],
        lr=lr, betas=betas, eps=eps)


# ---------------------------------------------------------------------------
# The shared training loop
# ---------------------------------------------------------------------------
def train_bytes_model(model, train: ByteData, val: ByteData, *,
                      steps: int, batch: int, T: int, lr: float,
                      device: str = "cuda", seed: int = 0,
                      warmup_frac: float = 0.05, wd: float = 0.01,
                      grad_clip: float = 1.0, val_every: int = 250,
                      val_batches: int = 40, wave_checkpoint: bool = False,
                      max_seconds: Optional[float] = None,
                      label: str = "") -> dict:
    """Train any byte_ids->logits model with the fair shared recipe. Returns a
    metrics dict. Data batches are seeded by `seed`, so two architectures trained
    with the same seed see the IDENTICAL byte stream.

    If max_seconds is set, training stops early once that wall-clock budget is
    exceeded (for the matched-wall-clock comparison). The cosine schedule still
    targets `steps`, so set steps generously when using a time budget."""
    model.to(device).train()
    opt = _build_optimizer(model, lr, wd, betas=(0.9, 0.95), eps=1e-8)
    gen = torch.Generator().manual_seed(seed)           # data sampler (shared across archs)
    lr_floor = 0.1 * lr
    use_ckpt = isinstance(model, WaveModel) and wave_checkpoint

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    history = []
    best = math.inf
    n_tok = 0
    steps_done = 0
    t0 = time.perf_counter()
    for step in range(1, steps + 1):
        cur_lr = lr_cosine(step, steps, warmup_frac, lr, lr_floor)
        for g in opt.param_groups:
            g["lr"] = cur_lr * g.get("lr_mult", 1.0)
        b = train.random_batch(batch, gen, device=device)
        logits = _forward_logits(model, b, use_checkpoint=use_ckpt)
        loss = ce_loss(logits, b)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()
        n_tok += b.numel()
        steps_done = step
        if step % val_every == 0 or step == steps:
            vbpb = validate_bytes(model, val, batch=batch, device=device,
                                  max_batches=val_batches)
            best = min(best, vbpb)
            history.append((step, round(vbpb, 4)))
            if max_seconds is not None and (time.perf_counter() - t0) > max_seconds:
                break
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    peak_gb = (torch.cuda.max_memory_allocated() / 1024 ** 3
               if device.startswith("cuda") else 0.0)
    return {
        "label": label,
        "arch": "wavelm" if isinstance(model, WaveModel) else "gpt",
        "params_dof": count_real_dof(model),
        "steps_target": steps, "steps_done": steps_done,
        "batch": batch, "T": T, "lr": lr, "seed": seed,
        "tokens": n_tok,
        "val_bpb_final": round(history[-1][1], 4) if history else None,
        "val_bpb_best": round(best, 4),
        "wall_s": round(wall, 1),
        "tok_per_s": round(n_tok / max(1e-6, wall), 0),
        "peak_gb": round(peak_gb, 2),
        "history": history,
    }


__all__ = ["count_real_dof", "build_wavelm", "build_gpt", "validate_bytes",
           "train_bytes_model"]
