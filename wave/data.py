"""Byte-level data pipeline for WaveLM (master plan Part 11).

No tokenizer, no vocabulary: the raw bytes of the text file ARE the symbol
stream (alphabet = 256 byte values). We memory-map the file as uint8 and sample
random length-T windows (nanoGPT-style). The next-byte target is the shifted
window, handled inside readout.ce_loss.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from torch import Tensor


class ByteData:
    """Random / sequential length-T byte windows over a memory-mapped file."""

    def __init__(self, path: str, T: int):
        self.path = path
        self.T = T
        # The text file's raw bytes are the byte-level tokens directly.
        self.data = np.memmap(path, dtype=np.uint8, mode="r")
        self.n = self.data.shape[0]
        if self.n < T + 1:
            raise ValueError(f"file {path} too short ({self.n} bytes) for T={T}")

    def random_batch(self, B: int, gen: torch.Generator, device: str = "cpu") -> Tensor:
        """B random windows of length T -> (B, T) long byte ids."""
        hi = self.n - self.T - 1
        starts = torch.randint(0, hi, (B,), generator=gen).tolist()
        out = np.empty((B, self.T), dtype=np.int64)
        for i, s in enumerate(starts):
            out[i] = self.data[s : s + self.T]
        return torch.from_numpy(out).to(device)

    def close(self) -> None:
        """Close the underlying memory map (needed on Windows before file delete)."""
        mm = getattr(self.data, "_mmap", None)
        if mm is not None:
            self.data = None
            mm.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def sequential_batches(self, B: int, max_batches: Optional[int] = None,
                           device: str = "cpu"):
        """Yield non-overlapping (B, T) windows tiling the file (for validation)."""
        win = self.T
        n_windows = self.n // win
        if max_batches is not None:
            n_windows = min(n_windows, max_batches * B)
        buf = []
        for w in range(n_windows):
            s = w * win
            buf.append(self.data[s : s + win].astype(np.int64))
            if len(buf) == B:
                yield torch.from_numpy(np.stack(buf)).to(device)
                buf = []
        if buf:
            yield torch.from_numpy(np.stack(buf)).to(device)


def decode_bytes(ids) -> str:
    """Decode a list/tensor of byte ids to a UTF-8 string (lossy at boundaries)."""
    if isinstance(ids, Tensor):
        ids = ids.tolist()
    return bytes(int(x) & 0xFF for x in ids).decode("utf-8", errors="replace")


def encode_str(s: str) -> list:
    """Encode a string to a list of byte ids."""
    return list(s.encode("utf-8"))


__all__ = ["ByteData", "decode_bytes", "encode_str"]
