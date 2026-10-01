"""Build a byte-level Pile dataset for WaveLM by detokenizing the local GPT-2
token cache back to raw UTF-8 bytes (GPT-2 byte-level BPE is reversible, so this
reconstructs the original Pile text bytes). No download needed.
"""

import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != _here]

import numpy as np
import tiktoken
import torch

CACHE = os.environ.get("PILE_TOKEN_CACHE", "data_cache/pile_250M_gpt2.pt")  # the 250M-token GPT-2 cache built by the DPNCA-LM repository
OUT_DIR = "data/pile"
TRAIN_TOKENS = 150_000_000     # ~600 MB of bytes
VAL_LO, VAL_HI = 150_000_000, 153_000_000   # separate, non-overlapping val slice
CHUNK = 2_000_000


def main():
    enc = tiktoken.get_encoding("gpt2")
    print("loading GPT-2 token cache (1.9 GB) ...")
    toks = np.asarray(torch.load(CACHE, map_location="cpu", weights_only=False))
    print(f"  tokens: {toks.shape[0]:,}")
    os.makedirs(OUT_DIR, exist_ok=True)

    def decode_range(lo, hi, out_path):
        n_bytes = 0
        with open(out_path, "wb") as f:
            for s in range(lo, hi, CHUNK):
                e = min(s + CHUNK, hi)
                b = enc.decode_bytes(toks[s:e].tolist())
                f.write(b)
                n_bytes += len(b)
        print(f"  wrote {out_path}: {n_bytes/1e6:.1f} MB "
              f"({(hi-lo)/1e6:.0f}M tokens, {n_bytes/(hi-lo):.2f} bytes/token)")

    print("decoding train ...")
    decode_range(0, TRAIN_TOKENS, os.path.join(OUT_DIR, "train.bytes"))
    print("decoding val ...")
    decode_range(VAL_LO, VAL_HI, os.path.join(OUT_DIR, "val.bytes"))
    print("done.")


if __name__ == "__main__":
    main()
