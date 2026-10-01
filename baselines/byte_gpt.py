"""A from-scratch byte-level decoder-only Transformer (the WaveLM comparison baseline).

Standard nanoGPT-style architecture, intentionally vanilla and *strong* (a fair
baseline, not a strawman):
  - token embedding over 256 bytes, **weight-tied** to the output projection;
  - N pre-LN decoder blocks: causal multi-head self-attention with **RoPE**
    (rotary position embeddings, no learned positional params) + GELU MLP (4x);
  - final LayerNorm; logits = x @ W_E^T.
  - GPT-2-style init, residual-projection scaling 1/sqrt(2*n_layer).

Forward signature matches WaveLM's usable output: `forward(byte_ids) -> logits
(B,T,V)`. The same `wave.readout.ce_loss` / `bits_per_byte` are used downstream,
so the metric is identical-by-construction.

No dropout (WaveLM has none either — keeps the comparison clean). fp32 by default.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass
class GPTConfig:
    V: int = 256
    d_model: int = 256
    n_layer: int = 6
    n_head: int = 4
    d_ff: int = 0          # 0 => 4 * d_model
    T: int = 1024          # max sequence length (for the RoPE cache)
    rope_base: float = 10000.0

    def __post_init__(self):
        if self.d_ff == 0:
            self.d_ff = 4 * self.d_model
        assert self.d_model % self.n_head == 0, "d_model must divide n_head"
        assert (self.d_model // self.n_head) % 2 == 0, "head_dim must be even for RoPE"


# ---------------------------------------------------------------------------
# RoPE (rotary position embeddings), LLaMA/HF convention
# ---------------------------------------------------------------------------
def _build_rope_cache(T: int, head_dim: int, base: float, device, dtype):
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
    t = torch.arange(T, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)                  # (T, head_dim/2)
    emb = torch.cat((freqs, freqs), dim=-1)           # (T, head_dim)
    return emb.cos().to(dtype), emb.sin().to(dtype)   # (T, head_dim) each


def _rotate_half(x: Tensor) -> Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def _apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    # x: (B, n_head, T, head_dim); cos/sin: (T, head_dim) -> broadcast (1,1,T,hd)
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    return x * cos + _rotate_half(x) * sin


# ---------------------------------------------------------------------------
# Blocks
# ---------------------------------------------------------------------------
class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.n_head = cfg.n_head
        self.head_dim = cfg.d_model // cfg.n_head
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        # (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        q = _apply_rope(q, cos[:T], sin[:T])
        k = _apply_rope(k, cos[:T], sin[:T])
        # causal scaled dot-product attention (no dropout)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


class MLP(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.fc = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.proj = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.proj(F.gelu(self.fc(x)))


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = MLP(cfg)

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        x = x + self.attn(self.ln1(x), cos, sin)   # pre-LN residual
        x = x + self.mlp(self.ln2(x))
        return x


class ByteGPT(nn.Module):
    """Byte-level decoder-only Transformer. forward(byte_ids)->logits (B,T,V)."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok = nn.Embedding(cfg.V, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        # weight tying: output projection shares the embedding matrix.
        self.head = nn.Linear(cfg.d_model, cfg.V, bias=False)
        self.head.weight = self.tok.weight

        head_dim = cfg.d_model // cfg.n_head
        cos, sin = _build_rope_cache(cfg.T, head_dim, cfg.rope_base,
                                     device="cpu", dtype=torch.float32)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)
        # GPT-2 residual scaling: shrink the projections that write into the residual
        for name, p in self.named_parameters():
            if name.endswith("proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, byte_ids: Tensor) -> Tensor:
        B, T = byte_ids.shape
        x = self.tok(byte_ids)                                  # (B, T, d)
        cos = self.rope_cos.to(x.dtype)
        sin = self.rope_sin.to(x.dtype)
        for blk in self.blocks:
            x = blk(x, cos, sin)
        x = self.ln_f(x)
        return self.head(x)                                     # (B, T, V)

    def param_count(self) -> int:
        # Real DOF. Weight-tied head shares tok.weight, so count it once.
        seen, total = set(), 0
        for p in self.parameters():
            if id(p) in seen:
                continue
            seen.add(id(p))
            total += p.numel()
        return total


__all__ = ["GPTConfig", "ByteGPT"]
