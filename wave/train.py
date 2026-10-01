"""WaveLM training loop (master plan Part 9).

Pure next-byte cross-entropy + one CONDITIONAL spectral-diversity term that
protects the omega_k carrier-frequency primitive. AdamW, cosine warmup-decay,
gradient clipping, activation checkpointing (inside the model). Structured
logging in bits-per-byte (the byte-level analogue of PPL).
"""

from __future__ import annotations

import json
import math
import os
import shutil
import sys
import time
from dataclasses import dataclass, field
from typing import Optional

import torch
from torch import Tensor

# Emit the nabla (gradient) symbol and other Unicode safely on Windows consoles.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def _ndof(p: torch.nn.Parameter) -> int:
    """Real degrees of freedom (complex params count twice)."""
    return p.numel() * (2 if p.is_complex() else 1)

from .config import WaveConfig
from .model import WaveModel
from .readout import ce_loss
from .data import ByteData, decode_bytes, encode_str
from .infer import generate


# ---------------------------------------------------------------------------
# Schedules and the spectral-diversity term
# ---------------------------------------------------------------------------


def lr_cosine(step: int, total: int, warmup_frac: float, peak: float, floor: float) -> float:
    warm = max(1, int(total * warmup_frac))
    if step < warm:
        return peak * step / warm
    prog = min(1.0, (step - warm) / max(1, total - warm))
    return floor + 0.5 * (peak - floor) * (1.0 + math.cos(math.pi * prog))


def spectral_repulsion(omega: Tensor) -> Tensor:
    """R_spec = -(d^2 - |sum_j e^{i omega_j}|^2)/(d(d-1)) in [-d/(d-1), 0].

    More negative when frequencies are spread; minimising L=L_CE+lambda*R_spec
    therefore pushes the omega_k apart. O(d).
    """
    d = omega.shape[0]
    S = torch.exp(1j * omega.to(torch.float64)).sum()
    S2 = (S.real**2 + S.imag**2)
    return (-(d * d - S2) / (d * (d - 1))).to(omega.dtype)


@dataclass
class SpectralGuard:
    """Activates lambda_omega only when Var(omega) collapses (Part 9.1)."""
    var0: float
    lambda0: float
    rho: float
    patience: int
    release_frac: float
    active: bool = False
    below_count: int = 0

    def update(self, var_now: float) -> float:
        if not self.active:
            if var_now < self.rho * self.var0:
                self.below_count += 1
                if self.below_count >= self.patience:
                    self.active = True
            else:
                self.below_count = 0
        else:
            if var_now > self.release_frac * self.var0:
                self.active = False
                self.below_count = 0
        return self.lambda0 if self.active else 0.0


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


@dataclass
class TrainState:
    step: int = 0
    best_val_bpb: float = math.inf
    started_at: float = field(default_factory=time.perf_counter)


def _gpu_gb() -> float:
    return torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0.0


@torch.no_grad()
def validate(model: WaveModel, val: ByteData, cfg: WaveConfig, device: str,
             max_batches: int = 20) -> float:
    model.eval()
    tot, nb = 0.0, 0
    for batch in val.sequential_batches(cfg.batch_size, max_batches=max_batches, device=device):
        logits, _ = model(batch, use_checkpoint=False)
        tot += float(ce_loss(logits, batch))
        nb += 1
    model.train()
    ce = tot / max(1, nb)
    return ce / math.log(2.0)            # bits per byte


def train_loop(
    model: WaveModel,
    train: ByteData,
    cfg: WaveConfig,
    *,
    val: Optional[ByteData] = None,
    device: str = "cuda",
    model_name: str = "WaveLM",
    log_every: int = 20,
    val_every: int = 500,
    ckpt_every: int = 1000,
    ckpt_dir: Optional[str] = None,
    metrics_path: Optional[str] = None,
    resume_ckpt: Optional[str] = None,
    init_from: Optional[str] = None,
    sample_prompt: str = "The ",
    sample_bytes: int = 96,
) -> TrainState:
    model.to(device).train()
    if cfg.mup:
        # Per-role muP groups: lr_mult = 1/fan_in so the base lr (lr_peak) transfers
        # across width d (and rank r). See WaveModel.mup_param_groups.
        opt = torch.optim.AdamW(
            model.mup_param_groups(cfg.lr_peak, cfg.weight_decay),
            lr=cfg.lr_peak, betas=(cfg.adam_beta1, cfg.adam_beta2), eps=cfg.adam_eps,
        )
    else:
        # Weight decay only on >=2-D params (embedding/kernels/mixing); none on the
        # 1-D / scalar dynamics params (frequencies, beta, g, T_o).
        decay = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2]
        no_decay = [p for p in model.parameters() if p.requires_grad and p.dim() < 2]
        opt = torch.optim.AdamW(
            [{"params": decay, "weight_decay": cfg.weight_decay, "lr_mult": 1.0, "name": "decay"},
             {"params": no_decay, "weight_decay": 0.0, "lr_mult": 1.0, "name": "no_decay"}],
            lr=cfg.lr_peak, betas=(cfg.adam_beta1, cfg.adam_beta2), eps=cfg.adam_eps,
        )
    gen = torch.Generator().manual_seed(cfg.seed)
    state = TrainState()
    if init_from:
        ck = torch.load(init_from, map_location=device, weights_only=False)
        missing, unexpected = model.load_state_dict(ck["model"], strict=False)
        print(f"  WARM-START base weights from {init_from} "
              f"(fresh opt/schedule; new params: {list(missing)})")
    if resume_ckpt:
        ck = torch.load(resume_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        if "opt" in ck:
            opt.load_state_dict(ck["opt"])
        state.step = ck.get("step", 0)
        state.best_val_bpb = ck.get("best_val_bpb", math.inf)
        print(f"  RESUMED from {resume_ckpt} at step {state.step}")
    guard = SpectralGuard(
        var0=max(model.omega_variance(), 1e-9), lambda0=cfg.lambda_omega0,
        rho=cfg.spec_collapse_rho, patience=cfg.spec_collapse_patience,
        release_frac=cfg.spec_release_frac,
    )

    if ckpt_dir:
        os.makedirs(ckpt_dir, exist_ok=True)
    mmode = "a" if resume_ckpt else "w"
    mfile = open(metrics_path, mmode, encoding="utf-8") if metrics_path else None

    token_budget = cfg.total_steps * cfg.batch_size * cfg.T
    total_dof = 0
    for g in opt.param_groups:
        dof = sum(_ndof(p) for p in g["params"])
        total_dof += dof
        print(f"  group {g.get('name', 'grp'):<8} {dof:>14,} DOF  wd={g['weight_decay']}"
              f"  lr_mult={g.get('lr_mult', 1.0):.4f}")
    print("=" * 78)
    print(f"  {model_name}  (complex wave-field byte LM)")
    print("=" * 78)
    print(f"  Params: {total_dof:,} total (real DOF; complex counted 2x)")
    print(f"  Config: {cfg.total_steps} steps, batch={cfg.batch_size}, seq_len={cfg.T} bytes")
    print(f"  Token budget: {token_budget:,} bytes")
    print(f"  Arch: d={cfg.d} n_sub={cfg.n_sub} delta={cfg.delta} L_cyc={cfg.L_cyc} "
          f"mix={cfg.mix_channels}(rank={cfg.mix_rank}) untie_steps={cfg.untie_steps} "
          f"Var0(w)={guard.var0:.3f}")
    print(f"  Opt: grad_clip={cfg.grad_clip} muP={cfg.mup}"
          f"{f' (d0={cfg.mup_d0} r0={cfg.mup_r0} untie_emb={cfg.untie_embedding})' if cfg.mup else ''}")
    print(f"  Device: {device}")
    print("=" * 78)

    tok_win, t_win = 0, time.perf_counter()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    while state.step < cfg.total_steps:
        state.step += 1
        lr = lr_cosine(state.step, cfg.total_steps, cfg.warmup_frac, cfg.lr_peak, cfg.lr_floor)
        for grp in opt.param_groups:
            grp["lr"] = lr * grp.get("lr_mult", 1.0)   # muP: preserve per-group ratios

        batch = train.random_batch(cfg.batch_size, gen, device=device)
        logits, _ = model(batch)
        L_ce = ce_loss(logits, batch)

        var_now = model.omega_variance()
        lam = guard.update(var_now)
        L = L_ce + (lam * spectral_repulsion(model.omega()) if lam > 0 else 0.0)

        opt.zero_grad(set_to_none=True)
        L.backward()
        gnorm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip))
        opt.step()

        tok_win += batch.numel()

        if state.step % log_every == 0 or state.step == 1:
            dt = time.perf_counter() - t_win
            tps = tok_win / max(1e-6, dt)
            tok_win, t_win = 0, time.perf_counter()
            ce = float(L_ce)
            bpb = ce / math.log(2.0)
            ppl = math.exp(min(ce, 20.0))
            elapsed = time.perf_counter() - state.started_at
            eta = elapsed / state.step * (cfg.total_steps - state.step)
            eta_str = f" ETA={eta/3600:.1f}h" if state.step >= 5 else ""
            print(f"Step {state.step:>5}/{cfg.total_steps} [{model_name}] | "
                  f"CE={ce:.4f} PPL={ppl:.2f} bpb={bpb:.3f} | {tps:,.0f} tok/s "
                  f"∇={gnorm:.2f} GPU={_gpu_gb():.1f}G lr={lr:.2e}{eta_str}")
            if mfile:
                mfile.write(json.dumps({
                    "step": state.step, "bpb": bpb, "ce": float(L_ce),
                    "grad": gnorm, "lr": lr, "var_omega": var_now,
                    "lambda_omega": lam, "gpu_gb": _gpu_gb(),
                }) + "\n")
                mfile.flush()

        if val is not None and state.step % val_every == 0:
            vbpb = validate(model, val, cfg, device)
            vce = vbpb * math.log(2.0)
            vppl = math.exp(min(vce, 20.0))
            print(f"  >>> VAL: CE={vce:.4f} PPL={vppl:.2f} ({vbpb:.4f} bpb)")
            # compact generation sample so progress is visible in the log
            try:
                ids = generate(model, encode_str(sample_prompt), max_new=sample_bytes,
                               temperature=0.8, top_k=40, device=device, seed=state.step)
                txt = decode_bytes(ids).replace("\n", "\\n").replace("\r", "")
                print(f"  >>> SAMPLE: {txt[:180]!r}")
            except Exception as ex:
                print(f"  >>> SAMPLE failed: {type(ex).__name__}: {ex}")
            model.train()
            if mfile:
                mfile.write(json.dumps({"step": state.step, "val_bpb": vbpb}) + "\n")
                mfile.flush()
            if ckpt_dir and vbpb < state.best_val_bpb:
                state.best_val_bpb = vbpb
                free_before = shutil.disk_usage(ckpt_dir).free / 1e9
                print(f"  Disk free before save ({ckpt_dir}): {free_before:.2f} GB")
                torch.save({"model": model.state_dict(), "cfg": cfg, "step": state.step,
                            "val_bpb": vbpb, "opt": opt.state_dict(),
                            "best_val_bpb": state.best_val_bpb},
                           os.path.join(ckpt_dir, "best.pt"))
                free_after = shutil.disk_usage(ckpt_dir).free / 1e9
                print(f"  Saved checkpoint: best.pt")
                print(f"  Disk free after save ({ckpt_dir}): {free_after:.2f} GB")
                print(f"  >>> New best Val PPL: {vppl:.2f}  ({vbpb:.4f} bpb)")

        if ckpt_dir and state.step % ckpt_every == 0:
            torch.save({"model": model.state_dict(), "cfg": cfg, "step": state.step,
                        "opt": opt.state_dict(), "best_val_bpb": state.best_val_bpb},
                       os.path.join(ckpt_dir, f"step_{state.step}.pt"))

    best_ppl = math.exp(min(state.best_val_bpb * math.log(2.0), 20.0))
    print("=" * 78)
    print(f"  DONE {state.step} steps in {(time.perf_counter()-state.started_at)/60:.1f} min")
    print(f"  Best Val: PPL={best_ppl:.2f}  CE={state.best_val_bpb*math.log(2):.4f}  "
          f"{state.best_val_bpb:.4f} bpb")
    print("=" * 78)
    if mfile:
        mfile.close()
    return state


__all__ = ["train_loop", "validate", "lr_cosine", "spectral_repulsion", "SpectralGuard", "TrainState"]
