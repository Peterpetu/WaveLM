"""Generate byte-level samples from a WaveLM checkpoint (master plan Part 10).

Runs on CPU by default so it can be used while a GPU training run is in progress.
"""

import argparse
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != _here]
sys.path.insert(0, os.path.dirname(_here))

import torch

from wave.config import WaveConfig
from wave.model import WaveModel
from wave.infer import generate
from wave.data import encode_str, decode_bytes


PROMPTS = [
    "Once upon a time",
    "The little dog",
    "Lily and Tom",
    "One sunny day,",
    "There was a small",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/wavelm_run1/best.pt")
    ap.add_argument("--max_new", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_k", type=int, default=40)
    ap.add_argument("--greedy", action="store_true")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg: WaveConfig = ckpt["cfg"]
    model = WaveModel(cfg)
    model.load_state_dict(ckpt["model"])
    model.to(args.device).eval()

    print(f"loaded {args.ckpt}  step={ckpt.get('step')}  val_bpb={ckpt.get('val_bpb')}")
    print(f"  d={cfg.d} T={cfg.T} n_sub={cfg.n_sub} delta={cfg.delta}")
    print("=" * 78)
    for p in PROMPTS:
        ids = encode_str(p)
        out = generate(model, ids, max_new=args.max_new, temperature=args.temperature,
                       top_k=(None if args.greedy else args.top_k), greedy=args.greedy,
                       device=args.device, seed=args.seed)
        text = decode_bytes(out)
        cont = decode_bytes(out[len(ids):])
        uniq = len(set(out[len(ids):])) / max(1, len(out) - len(ids))
        print(f"\n>>> PROMPT: {p!r}  (unique-byte ratio {uniq:.2f})")
        print(f"    {text!r}")


if __name__ == "__main__":
    main()
