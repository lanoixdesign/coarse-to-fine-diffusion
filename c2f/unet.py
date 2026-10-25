"""A small UNet conditioned on diffusion time and a label/text embedding.

in_channels = generated channels + conditioning-image channels (concatenated).
"""
from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device).float() / half)
    args = t.float()[:, None] * freqs[None]
    return torch.cat([args.sin(), args.cos()], dim=1)


def _gn(ch: int) -> nn.GroupNorm:
    return nn.GroupNorm(min(8, ch), ch)


class ResBlock(nn.Module):
    def __init__(self, cin: int, cout: int, emb_dim: int):
        super().__init__()
        self.n1, self.c1 = _gn(cin), nn.Conv2d(cin, cout, 3, padding=1)
        self.emb = nn.Linear(emb_dim, cout * 2)  # FiLM: scale & shift
        self.n2, self.c2 = _gn(cout), nn.Conv2d(cout, cout, 3, padding=1)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, emb):
        h = self.c1(F.silu(self.n1(x)))
        scale, shift = self.emb(F.silu(emb))[:, :, None, None].chunk(2, dim=1)
        h = self.n2(h) * (1 + scale) + shift
        h = self.c2(F.silu(h))
        return h + self.skip(x)


class UNet(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, cond_dim: int,
                 base_channels: int = 32, channel_mults: Tuple[int, ...] = (1, 2, 2)):
        super().__init__()
        emb_dim = base_channels * 4
        self.base = base_channels
        self.t_mlp = nn.Sequential(nn.Linear(base_channels, emb_dim), nn.SiLU(), nn.Linear(emb_dim, emb_dim))
        self.c_proj = nn.Linear(cond_dim, emb_dim)
        self.inp = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        chs = [base_channels * m for m in channel_mults]
        self.down = nn.ModuleList()
        self.pools = nn.ModuleList()
        prev = base_channels
        for i, c in enumerate(chs):
            self.down.append(ResBlock(prev, c, emb_dim))
            self.pools.append(nn.Conv2d(c, c, 3, stride=2, padding=1) if i < len(chs) - 1 else nn.Identity())
            prev = c
        self.mid = ResBlock(prev, prev, emb_dim)
        self.up = nn.ModuleList()
        self.ups = nn.ModuleList()
        for i, c in reversed(list(enumerate(chs))):
            self.up.append(ResBlock(prev + c, c, emb_dim))
            self.ups.append(nn.Upsample(scale_factor=2, mode="nearest") if i > 0 else nn.Identity())
            prev = c
        self.out = nn.Sequential(_gn(prev), nn.SiLU(), nn.Conv2d(prev, out_channels, 3, padding=1))

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        emb = self.t_mlp(timestep_embedding(t, self.base)) + self.c_proj(cond)
        h = self.inp(x)
        skips = []
        for block, pool in zip(self.down, self.pools):
            h = block(h, emb)
            skips.append(h)
            h = pool(h)
        h = self.mid(h, emb)
        for block, up in zip(self.up, self.ups):
            h = block(torch.cat([h, skips.pop()], dim=1), emb)
            h = up(h)
        return self.out(h)
