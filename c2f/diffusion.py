"""Minimal DDPM: linear beta schedule, epsilon-prediction MSE, DDIM sampler."""
from __future__ import annotations

from typing import Callable, Optional

import torch
import torch.nn.functional as F


class Diffusion:
    def __init__(self, timesteps: int = 1000, beta_start: float = 1e-4, beta_end: float = 0.02):
        self.T = timesteps
        betas = torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)
        alphas_cumprod = torch.cumprod(1.0 - betas, dim=0)
        self.alphas_cumprod = alphas_cumprod.float()

    def _ab(self, t: torch.Tensor, device) -> torch.Tensor:
        return self.alphas_cumprod.to(device)[t].view(-1, 1, 1, 1)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        ab = self._ab(t, x0.device)
        return ab.sqrt() * x0 + (1 - ab).sqrt() * noise

    def loss(self, model: Callable, x0: torch.Tensor, cond_img: Optional[torch.Tensor],
             cond_emb: torch.Tensor, channel_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Noise-prediction MSE, optionally weighted per output channel."""
        b = x0.shape[0]
        t = torch.randint(0, self.T, (b,), device=x0.device)
        noise = torch.randn_like(x0)
        xt = self.q_sample(x0, t, noise)
        inp = xt if cond_img is None else torch.cat([xt, cond_img], dim=1)
        eps = model(inp, t, cond_emb)
        se = F.mse_loss(eps, noise, reduction="none").mean(dim=(0, 2, 3))  # per channel
        if channel_weights is None:
            return se.mean()
        w = channel_weights.to(se.device)
        return (se * w).sum() / w.sum().clamp_min(1e-8)

    @torch.no_grad()
    def sample(self, model: Callable, shape, cond_img: Optional[torch.Tensor], cond_emb: torch.Tensor,
               steps: int = 50, device="cpu") -> torch.Tensor:
        """Deterministic DDIM (eta=0) over `steps` evenly spaced timesteps."""
        x = torch.randn(shape, device=device)
        ts = torch.linspace(self.T - 1, 0, steps, device=device).long()
        ac = self.alphas_cumprod.to(device)
        for i, t in enumerate(ts):
            tt = torch.full((shape[0],), int(t), device=device, dtype=torch.long)
            inp = x if cond_img is None else torch.cat([x, cond_img], dim=1)
            eps = model(inp, tt, cond_emb)
            ab = ac[t]
            x0 = ((x - (1 - ab).sqrt() * eps) / ab.sqrt()).clamp(-1, 1)
            ab_prev = ac[ts[i + 1]] if i + 1 < len(ts) else torch.tensor(1.0, device=device)
            x = ab_prev.sqrt() * x0 + (1 - ab_prev).sqrt() * eps
        return x
