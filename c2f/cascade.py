"""The cascade: one UNet + one conditioning encoder per stage.

Each stage can be given its own prompt (`stage_prompts`); a stage without one
falls back to the base prompt (`prompt`, or the label-derived prompt / the
labels themselves). With no per-stage prompts the behaviour is exactly the
original one, and checkpoints are unchanged (same modules, same state dicts).
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ALL_CHANNELS, CascadeConfig, StageConfig
from .diffusion import Diffusion
from .text import (broadcast_cond, build_encoder, labels_to_prompts, load_clip_backbone,
                   normalize_stage_prompts)
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
    def __init__(self, cfg: CascadeConfig, text_backbone=None):
        """`text_backbone`: an already-loaded `CLIPTextBackbone` to use with cond_type="clip"
        (default: load `cfg.clip_model` once via `load_clip_backbone`, reused per process)."""
        super().__init__()
        self.cfg = cfg
        # One encoder per stage so stages can be trained independently. With CLIP the
        # frozen backbone is loaded ONCE and shared; each stage only owns a small adapter.
        backbone = None
        if cfg.cond_type == "clip":
            backbone = text_backbone if text_backbone is not None else load_clip_backbone(cfg.clip_model)
        self.text_backbone = backbone  # plain attribute (not a module): never in parameters()/state_dict()
        self.encoders = nn.ModuleList(build_encoder(cfg.cond_type, cfg.cond_dim, cfg.num_classes, backbone=backbone)
                                      for _ in cfg.stages)
        self.diffusion = Diffusion(cfg.timesteps, cfg.beta_start, cfg.beta_end)
        nets = []
        for k, s in enumerate(cfg.stages):
            cond_ch = 0 if k == 0 else cfg.stages[k - 1].n_out
            nets.append(UNet(s.n_out + cond_ch, s.n_out, cfg.cond_dim, s.base_channels, s.channel_mults))
        self.stages = nn.ModuleList(nets)

    # ---- conditioning -------------------------------------------------
    def stage_condition(self, k: int, labels: Optional[torch.Tensor] = None, prompt=None,
                        stage_prompts=None, batch: Optional[int] = None):
        """Resolve the conditioning *input* of stage k (before encoding).

        Fallback rules (stage numbers in `stage_prompts` are 1-based):
          1. `stage_prompts` has an entry for stage k+1      -> use it;
          2. else the base `prompt` (str or list of B str)    -> use it (text only);
          3. else `labels`: label mode -> the labels, text mode -> labels_to_prompts(labels).
        Returns a LongTensor of B labels ("label") or a list of B strings ("text"/"clip").
        """
        ct = self.cfg.cond_type
        device = next(self.encoders[k].parameters()).device
        batch = self._batch_size(labels, prompt, stage_prompts, batch)
        if ct == "label" and prompt is not None:
            raise TypeError("label conditioning cannot take a text prompt; use cond_type='clip' or 'text' "
                            "(train.py/sample.py: --cond clip, or the offline stub --cond text)")
        override = normalize_stage_prompts(stage_prompts, len(self.cfg.stages))[k]
        if override is not None:
            return broadcast_cond(override, batch, ct, device)
        if prompt is not None:
            return broadcast_cond(prompt, batch, ct, device)
        if labels is None:
            raise ValueError(f"stage {k + 1} has no prompt: pass `labels`, a base `prompt`, "
                             f"or a prompt for every stage")
        if ct == "label":
            return broadcast_cond(labels, batch, ct, device)
        return labels_to_prompts(torch.as_tensor(labels).tolist())

    def stage_conditions(self, labels=None, prompt=None, stage_prompts=None,
                         batch: Optional[int] = None, upto: Optional[int] = None) -> list:
        """`stage_condition` for every stage (or the first `upto`)."""
        n = len(self.cfg.stages) if upto is None else upto
        return [self.stage_condition(k, labels, prompt, stage_prompts, batch) for k in range(n)]

    @staticmethod
    def _batch_size(labels, prompt, stage_prompts, batch) -> int:
        if batch is not None:
            return int(batch)
        if labels is not None:
            return int(torch.as_tensor(labels).reshape(-1).shape[0])
        cands = [prompt] + (list(stage_prompts.values()) if isinstance(stage_prompts, dict)
                            else list(stage_prompts or []))
        for p in cands:
            if isinstance(p, (list, tuple)) or (torch.is_tensor(p) and p.ndim == 1):
                return len(p)
        if any(p is not None for p in cands):
            return 1  # only single strings / scalars given
        raise ValueError("cannot infer the batch size: pass labels, a list of prompts, or batch=")

    def encode(self, k: int, labels: Optional[torch.Tensor] = None, drop_prob: float = 0.0,
               cond=None) -> torch.Tensor:
        """Embed stage k's condition with stage k's own encoder.

        `cond` is the resolved stage input (see `stage_condition`); if omitted it
        is derived from `labels` as before (backward compatible). With
        `drop_prob > 0`, each row is replaced by the encoder's null embedding
        with that probability (condition dropout for classifier-free guidance).
        """
        enc = self.encoders[k]
        if cond is None:
            cond = labels if self.cfg.cond_type == "label" else labels_to_prompts(labels.tolist())
        emb = enc(cond)
        if drop_prob > 0:
            mask = (torch.rand(emb.shape[0], device=emb.device) < drop_prob)[:, None]
            emb = torch.where(mask, enc.null(emb.shape[0], emb.device), emb)
        return emb

    def null_embedding(self, k: int, batch: int, device=None) -> torch.Tensor:
        """Stage k's unconditional embedding (what dropout trains; used by CFG later)."""
        return self.encoders[k].null(batch, device)

    def encode_stage(self, k: int, labels=None, prompt=None, stage_prompts=None,
                     drop_prob: float = 0.0, batch: Optional[int] = None) -> torch.Tensor:
        """Resolve stage k's prompt (with fallback) and embed it: (B, cond_dim)."""
        cond = self.stage_condition(k, labels, prompt, stage_prompts, batch)
        return self.encode(k, drop_prob=drop_prob, cond=cond)

    # ---- training -----------------------------------------------------
    def stage_loss(self, k: int, ycc_full: torch.Tensor, labels: Optional[torch.Tensor] = None,
                   prompt=None, stage_prompts=None) -> torch.Tensor:
        """ycc_full: (B,3,H,W) YCbCr in [-1,1] at any res >= stage res.

        Conditioning: stage k uses `stage_prompts[k+1]` if given, else `prompt`,
        else `labels` (see `stage_condition`). `stage_loss(k, x, labels)` is the
        original call and behaves exactly as before.
        """
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
        emb = self.encode_stage(k, labels, prompt, stage_prompts, self.cfg.cond_drop_prob,
                                batch=ycc_full.shape[0])
        return self.diffusion.loss(self.stages[k], target, cond_img, emb, channel_weights(s))

    # ---- sampling -----------------------------------------------------
    @torch.no_grad()
    def sample(self, labels: Optional[torch.Tensor] = None, steps: int = 50, upto: Optional[int] = None,
               prompt=None, stage_prompts=None, num_samples: Optional[int] = None) -> List[torch.Tensor]:
        """Run the cascade; returns per-stage outputs in [-1,1] (stage channels only).

        `sample(labels)` is the original call. With text conditioning, `prompt`
        is the base prompt for every stage and `stage_prompts` (dict keyed 1..n,
        or a per-stage list) overrides it for individual stages.
        """
        device = next(self.parameters()).device
        if labels is not None:
            labels = torch.as_tensor(labels).to(device)
        b = self._batch_size(labels, prompt, stage_prompts, num_samples)
        outs, prev = [], None
        n = len(self.stages) if upto is None else upto
        for k in range(n):
            s = self.cfg.stages[k]
            emb = self.encode_stage(k, labels, prompt, stage_prompts, batch=b)
            cond_img = None if prev is None else build_condition(prev, s.resolution)
            x = self.diffusion.sample(self.stages[k], (b, s.n_out, s.resolution, s.resolution),
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
