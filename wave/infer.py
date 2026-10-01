"""Streaming inference for WaveLM (master plan Part 10).

The training forward is parallel over positions. At inference we append one byte
at a time and reuse a cache of each committed position's full internal stack
{psi^(tau)[m] : tau=0..N_sub}. A new position computes its stack level-by-level,
reading the cached past stacks at dilated offsets. Cost per byte O(N_sub*Delta*d),
independent of context length. The streaming logits are bit-identical to the
parallel forward (test_stream_equals_parallel).
"""

from __future__ import annotations

import math
from typing import List, Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from .model import WaveModel


class WaveStreamer:
    def __init__(self, model: WaveModel, batch_size: int = 1, device: str = "cpu"):
        self.m = model
        self.cfg = model.cfg
        self.B = batch_size
        self.device = device
        cdtype = model._cdtype
        # cache[b, position, tau, channel]
        self.cache = torch.zeros(
            batch_size, self.cfg.T, self.cfg.n_sub + 1, self.cfg.d,
            dtype=cdtype, device=device,
        )
        self.length = 0
        # cache constrained params once
        self._K = model.K.detach()
        self._beta = model.beta().detach()
        self._g = model.g().detach()
        self._omega = model.omega().detach()
        self._W_E = model.W_E.detach()
        # readout basis: W_out if untied (item 3), else W_E (tied, the default)
        self._W_out = (model.W_out.detach()
                       if getattr(model, "W_out", None) is not None else self._W_E)
        self._T_o = model.T_o().detach()
        self._rot = torch.exp(1j * self._omega).to(cdtype)   # (d,)
        self._M = model.M.detach() if model.M is not None else None
        self._mU = model.mix_U.detach() if model.mix_U is not None else None
        self._mV = model.mix_V.detach() if model.mix_V is not None else None

    @torch.no_grad()
    def step(self, byte_ids: Tensor) -> Tensor:
        """Ingest one byte per batch element; return next-byte logits (B, V).

        byte_ids: (B,) long. Updates the cache at position = current length.
        """
        cfg, n = self.cfg, self.length
        if n >= cfg.T:
            raise RuntimeError("cache full; context length exceeded")
        d = cfg.d
        beta, g = self._beta.to(self._W_E.dtype), self._g
        # level 0: inject embedding
        stack0 = self._W_E[byte_ids]                          # (B, d)
        self.cache[:, n, 0, :] = stack0
        for tau in range(cfg.n_sub):
            D = cfg.dilation(tau)
            K_tau = self._K[tau] if cfg.untie_steps else self._K   # per-step kernel (item 4)
            p = beta * self.cache[:, n, tau, :]               # self term (B,d)
            for j in range(1, cfg.delta + 1):
                m = n - j * D
                if m >= 0:
                    p = p + K_tau[j - 1].view(1, d) * self.cache[:, m, tau, :]
            if cfg.mix_channels and tau % cfg.mix_every == 0:
                pt = tau // cfg.mix_every
                if self._M is not None:
                    p = torch.einsum("bk,kj->bj", p, self._M)
                elif self._mU is not None:
                    z = torch.einsum("bk,kr->br", p, self._mV[pt].conj())
                    p = p + torch.einsum("br,kr->bk", z, self._mU[pt])
            # per-channel saturation -> rotation -> RMS=1 projection (matches
            # wave.field.wave_step exactly so streaming == parallel forward).
            u = (p.real**2 + p.imag**2)
            q = p / (1.0 + g.to(u.dtype) * u).to(p.dtype)
            r = self._rot.view(1, d) * q
            norm = (r.real**2 + r.imag**2).sum(dim=-1, keepdim=True).sqrt()
            target = math.sqrt(d)
            self.cache[:, n, tau + 1, :] = r * target / (norm + 1e-8).to(r.dtype)
        psi = self.cache[:, n, cfg.n_sub, :]                  # (B, d)
        a = torch.einsum("bk,vk->bv", psi, self._W_out.conj())  # (B, V) readout basis
        logits = (a.real**2 + a.imag**2) / self._T_o.to(a.real.dtype)
        self.length += 1
        return logits


@torch.no_grad()
def generate(model: WaveModel, prompt: List[int], max_new: int,
             temperature: float = 1.0, top_k: Optional[int] = None,
             greedy: bool = False, device: str = "cpu",
             seed: int = 0) -> List[int]:
    """Stream a prompt then sample max_new bytes. Returns the full byte list."""
    model.eval()
    s = WaveStreamer(model, batch_size=1, device=device)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    out = list(prompt)
    logits = None
    for b in prompt:
        logits = s.step(torch.tensor([b], device=device, dtype=torch.long))
    for _ in range(max_new):
        if logits is None:  # empty prompt: seed with byte 0
            logits = s.step(torch.tensor([0], device=device, dtype=torch.long))
        lg = logits[0].float().cpu()
        if greedy:
            nxt = int(lg.argmax())
        else:
            lg = lg / max(1e-6, temperature)
            if top_k is not None:
                v, ix = lg.topk(min(top_k, lg.numel()))
                mask = torch.full_like(lg, float("-inf"))
                mask[ix] = v
                lg = mask
            probs = F.softmax(lg, dim=-1)
            nxt = int(torch.multinomial(probs, 1, generator=gen))
        out.append(nxt)
        logits = s.step(torch.tensor([nxt], device=device, dtype=torch.long))
    return out


__all__ = ["WaveStreamer", "generate"]
