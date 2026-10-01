"""Train WaveLM on byte-level TinyStories (master plan Phase F).

Usage:
  python scripts/train_wave.py --steps 8000 --batch 16 --d 256 --n_sub 24
"""

import argparse
import os
import sys

# Python auto-prepends this script's own directory (scripts/) to sys.path, where a
# legacy scripts/profile.py shadows the stdlib `profile` module (breaks torch._dynamo).
# Remove scripts/ and add the repo root so only the `wave` package is importable.
_here = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != _here]
sys.path.insert(0, os.path.dirname(_here))

import torch

from wave.config import WaveConfig
from wave.model import WaveModel
from wave.data import ByteData
from wave.train import train_loop


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_txt", default="data/tinystories/train.txt")
    ap.add_argument("--val_txt", default="data/tinystories/val.txt")
    ap.add_argument("--ckpt_dir", default="checkpoints/wavelm")
    ap.add_argument("--metrics", default="checkpoints/wavelm/metrics.jsonl")
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--T", type=int, default=2048)
    ap.add_argument("--n_sub", type=int, default=24)
    ap.add_argument("--delta", type=int, default=2)
    ap.add_argument("--checkpoint_every", type=int, default=5)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--val_every", type=int, default=500)
    ap.add_argument("--ckpt_every", type=int, default=1000)
    ap.add_argument("--log_every", type=int, default=20)
    ap.add_argument("--mix_channels", action="store_true")
    ap.add_argument("--mix_every", type=int, default=4)
    ap.add_argument("--mix_rank", type=int, default=0)
    # ---- v0.4: muP width-scaled lr (item 3) ----
    ap.add_argument("--mup", action="store_true")
    ap.add_argument("--mup_d0", type=int, default=256)
    ap.add_argument("--mup_r0", type=int, default=32)
    ap.add_argument("--untie_embedding", action="store_true")
    # ---- v0.4: capacity (item 4) ----
    ap.add_argument("--untie_steps", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--init_from", default=None)
    ap.add_argument("--seed", type=int, default=1729)
    ap.add_argument("--sample_prompt", default="The ")
    ap.add_argument("--sample_bytes", type=int, default=96)
    args = ap.parse_args()

    cfg = WaveConfig(
        V=256, d=args.d, T=args.T, n_sub=args.n_sub, delta=args.delta,
        batch_size=args.batch, total_steps=args.steps, lr_peak=args.lr,
        checkpoint_every=args.checkpoint_every,
        mix_channels=args.mix_channels, mix_every=args.mix_every,
        mix_rank=args.mix_rank, seed=args.seed,
        mup=args.mup, mup_d0=args.mup_d0, mup_r0=args.mup_r0,
        untie_embedding=args.untie_embedding, untie_steps=args.untie_steps,
        real_dtype="float32",
    )
    model = WaveModel(cfg)
    train = ByteData(args.train_txt, cfg.T)
    val = ByteData(args.val_txt, cfg.T)
    train_loop(
        model, train, cfg, val=val, device=args.device,
        log_every=args.log_every, val_every=args.val_every,
        ckpt_every=args.ckpt_every,
        ckpt_dir=args.ckpt_dir, metrics_path=args.metrics,
        resume_ckpt=args.resume, init_from=args.init_from,
        sample_prompt=args.sample_prompt, sample_bytes=args.sample_bytes,
    )


if __name__ == "__main__":
    main()
