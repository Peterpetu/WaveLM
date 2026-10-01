"""Lightweight tokenised data loader.

Reads packed token files (uint16 binary) and yields batches.
"""

import os
import numpy as np
import torch
from torch.utils.data import IterableDataset, DataLoader


class PackedTokenDataset(IterableDataset):
    """Yields fixed-length token sequences from a flat uint16 binary file."""

    def __init__(self, path: str, ctx_len: int, n_samples_per_epoch: int = 100_000):
        super().__init__()
        self.path = path
        self.ctx_len = ctx_len
        self.n_samples_per_epoch = n_samples_per_epoch
        # mmap the file
        self.data = np.memmap(path, dtype=np.uint16, mode="r")
        self.n_tokens = len(self.data)

    def __iter__(self):
        worker = torch.utils.data.get_worker_info()
        seed = (worker.seed if worker else 0) % (2 ** 32)
        rng = np.random.default_rng(seed)
        for _ in range(self.n_samples_per_epoch):
            i = rng.integers(0, self.n_tokens - self.ctx_len - 1)
            chunk = np.array(self.data[i:i + self.ctx_len + 1], dtype=np.int64)
            yield {"tokens": torch.from_numpy(chunk[:-1])}


def collate(batch):
    return {"tokens": torch.stack([b["tokens"] for b in batch], dim=0)}


def make_loader(path: str, ctx_len: int, batch_size: int, n_workers: int = 0):
    ds = PackedTokenDataset(path, ctx_len)
    return DataLoader(ds, batch_size=batch_size, num_workers=n_workers, collate_fn=collate)


def pack_corpus(text_path: str, tokenizer, out_path: str, eos_id: int) -> int:
    """Pack a text corpus (one document per line) into a flat uint16 file."""
    n_tokens = 0
    with open(text_path, "r", encoding="utf-8") as f_in, open(out_path, "wb") as f_out:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            ids = tokenizer.encode(line)
            ids.append(eos_id)
            arr = np.asarray(ids, dtype=np.uint16)
            f_out.write(arr.tobytes())
            n_tokens += len(arr)
    return n_tokens


__all__ = ["PackedTokenDataset", "make_loader", "pack_corpus", "collate"]
