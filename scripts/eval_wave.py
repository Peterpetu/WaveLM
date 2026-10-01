"""Clean full-validation bits-per-byte for a WaveLM checkpoint.

Tiles the entire validation file with non-overlapping windows (not just the
20-batch training-time estimate), for a final reportable number.
"""

import argparse
import math
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != _here]
sys.path.insert(0, os.path.dirname(_here))

import torch

from wave.config import WaveConfig
from wave.model import WaveModel
from wave.data import ByteData
from wave.readout import ce_loss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/wavelm_run2/best.pt")
    ap.add_argument("--val_txt", default="data/tinystories/val.txt")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg: WaveConfig = ck["cfg"]
    model = WaveModel(cfg)
    model.load_state_dict(ck["model"])
    model.to(args.device).eval()

    val = ByteData(args.val_txt, cfg.T)
    tot, nb = 0.0, 0
    with torch.no_grad():
        for batch in val.sequential_batches(args.batch, device=args.device):
            logits, _ = model(batch, use_checkpoint=False)
            tot += float(ce_loss(logits, batch))
            nb += 1
    val.close()
    ce = tot / max(1, nb)
    print(f"ckpt={args.ckpt} step={ck.get('step')}")
    print(f"full-val: {ce/math.log(2):.4f} bits/byte  ({ce:.4f} nats/byte)  over {nb} batches")


if __name__ == "__main__":
    main()
