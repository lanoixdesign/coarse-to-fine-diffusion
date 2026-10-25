"""The cascade: one UNet per stage + one shared conditioning encoder."""
from __future__ import annotations

from typing import List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ALL_CHANNELS, CascadeConfig, StageConfig
from .diffusion import Diffusion
from .text import build_encoder, labels_to_prompts
from .unet import UNet


def channel_index(names: Sequence[str]) -> List[int]:
    return [ALL_CHANNELS.index(n) for n in names]


def build_condition(prev_img: torch.Tensor, resolution: int) -> torch.Tensor:
    """Upsample the previous stage's output to this stage's resolution."""
    return F.interpolate(prev_img, size=(resolution, resolution), mode="bilinear", align_corners=False)


def channel_weights(stage: StageConfig) -> torch.Tensor:
    return torch.tensor([1.0 if c == "Y" else stage.chroma_weight for c in stage.channels])


def stage_target(ycc: torch.Tensor, stage: StageConfig) -> torch.Tensor:
    """Select this stage's channels from a full YCbCr image in [-1,1] at stage res.

    With chroma_downsample > 1, chroma targets are low-pass (coarse chroma).
    """
    x = ycc[:, channel_index(stage.channels)]
    if stage.chroma_downsample > 1 and stage.n_out > 1:
        f = stage.chroma_downsample
        chroma = x[:, 1:]
        chroma = F.interpolate(F.avg_pool2d(chroma, f), scale_factor=f, mode="bilinear", align_corners=False)
        x = torch.cat([x[:, :1], chroma], dim=1)
    return x


class Cascade(nn.Module):
    def __init__(self, cfg: CascadeConfig):
        super().__init__()
        self.cfg = cfg
        # One trainable encoder per stage so stages can be trained independently.
        # A frozen pretrained encoder (CLIP/T5) could be shared across stages.
        self.encoders = nn.ModuleList(build_encoder(cfg.cond_type, cfg.cond_dim, cfg.num_classes)
                                      for _ in cfg.stages)
        self.diffusion = Diffusion(cfg.timesteps, cfg.beta_start, cfg.beta_end)
        nets = []
        for k, s in enumerate(cfg.stages):
            cond_ch = 0 if k == 0 else cfg.stages[k - 1].n_out
            nets.append(UNet(s.n_out + cond_ch, s.n_out, cfg.cond_dim, s.base_channels, s.channel_mults))
        self.stages = nn.ModuleList(nets)

    # ---- conditioning -------------------------------------------------
    def encode(self, k: int, labels: torch.Tensor, drop_prob: float = 0.0) -> torch.Tensor:
        enc = self.encoders[k]
        cond = labels if self.cfg.cond_type == "label" else labels_to_prompts(labels.tolist())
        emb = enc(cond)
        if drop_prob > 0:
            mask = (torch.rand(emb.shape[0], device=emb.device) < drop_prob)[:, None]
            emb = torch.where(mask, enc.null(emb.shape[0], emb.device), emb)
        return emb

    # ---- training -----------------------------------------------------
    def stage_loss(self, k: int, ycc_full: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """ycc_full: (B,3,H,W) YCbCr in [-1,1] at any res >= stage res."""
        s = self.cfg.stages[k]
        x_k = F.interpolate(ycc_full, size=(s.resolution,) * 2, mode="area")
        target = stage_target(x_k, s)
        cond_img = None
        if k > 0:
            p = self.cfg.stages[k - 1]
            x_prev = stage_target(F.interpolate(ycc_full, size=(p.resolution,) * 2, mode="area"), p)
            cond_img = build_condition(x_prev, s.resolution)
            if s.cond_noise_std > 0:
                cond_img = cond_img + s.cond_noise_std * torch.randn_like(cond_img)
        emb = self.encode(k, labels, self.cfg.cond_drop_prob)
        return self.diffusion.loss(self.stages[k], target, cond_img, emb, channel_weights(s))

    # ---- sampling -----------------------------------------------------
    @torch.no_grad()
    def sample(self, labels: torch.Tensor, steps: int = 50, upto: Optional[int] = None) -> List[torch.Tensor]:
        """Run the cascade; returns per-stage outputs in [-1,1] (stage channels only)."""
        device = next(self.parameters()).device
        labels = labels.to(device)
        outs, prev = [], None
        n = len(self.stages) if upto is None else upto
        for k in range(n):
            s = self.cfg.stages[k]
            emb = self.encode(k, labels)
            cond_img = None if prev is None else build_condition(prev, s.resolution)
            x = self.diffusion.sample(self.stages[k], (labels.shape[0], s.n_out, s.resolution, s.resolution),
                                      cond_img, emb, steps=steps, device=device)
            outs.append(x)
            prev = x
        return outs


    # ---- per-stage checkpoints -----------------------------------------
    def stage_state(self, k: int) -> dict:
        return {"unet": self.stages[k].state_dict(), "encoder": self.encoders[k].state_dict()}

    def load_stage_state(self, k: int, state: dict) -> None:
        self.stages[k].load_state_dict(state["unet"])
        self.encoders[k].load_state_dict(state["encoder"])


def to_ycbcr3(x: torch.Tensor, names: Sequence[str]) -> torch.Tensor:
    """Pad a stage output to 3 YCbCr channels (missing chroma = neutral 0)."""
    full = torch.zeros(x.shape[0], 3, *x.shape[2:], device=x.device)
    full[:, channel_index(names)] = x
    return full
