"""Configuration dataclasses.

The colour schedule lives here: each stage declares its resolution, which
YCbCr channels it *generates*, and how strongly chroma is weighted in its loss.
Stage 1 is luminance-only; chroma channels appear in later stages.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import List, Optional, Tuple

ALL_CHANNELS = ("Y", "Cb", "Cr")


@dataclass
class StageConfig:
    resolution: int
    # Channels generated at this stage, subset of ("Y", "Cb", "Cr"), Y first.
    channels: Tuple[str, ...] = ("Y",)
    # Loss weight applied to chroma channels (Y always has weight 1.0).
    chroma_weight: float = 1.0
    # If >1, the chroma *target* is average-pooled by this factor and
    # upsampled back, i.e. the stage only learns coarse chroma.
    chroma_downsample: int = 1
    # Std of Gaussian noise added to the upsampled conditioning image during
    # training ("conditioning augmentation", cf. Ho et al. 2021). 0 disables.
    cond_noise_std: float = 0.1
    base_channels: int = 32
    channel_mults: Tuple[int, ...] = (1, 2, 2)

    @property
    def n_out(self) -> int:
        return len(self.channels)


@dataclass
class CascadeConfig:
    stages: List[StageConfig] = field(default_factory=list)
    timesteps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 0.02
    # Text / label conditioning
    cond_type: str = "label"   # "label" | "text" (hashed stub) | "clip" (frozen CLIP + per-stage adapters)
    # Hugging Face model id or local path of the CLIP text model (cond_type="clip" only;
    # None = c2f.text.DEFAULT_CLIP_MODEL). Recorded in checkpoints via to_dict().
    clip_model: Optional[str] = None
    cond_dim: int = 128
    num_classes: int = 10
    # Probability of dropping the condition during training (enables
    # classifier-free guidance later; 0 disables).
    cond_drop_prob: float = 0.1

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "CascadeConfig":
        d = dict(d)
        stages = [StageConfig(**{**s, "channels": tuple(s["channels"]),
                                 "channel_mults": tuple(s["channel_mults"])})
                  for s in d.pop("stages")]
        return CascadeConfig(stages=stages, **d)


def default_config() -> CascadeConfig:
    """16x16 gray -> 32x32 gray refinement + coarse chroma -> 64x64 full chroma."""
    return CascadeConfig(stages=[
        StageConfig(resolution=16, channels=("Y",), chroma_weight=0.0, cond_noise_std=0.0),
        StageConfig(resolution=32, channels=("Y", "Cb", "Cr"), chroma_weight=0.5, chroma_downsample=4),
        StageConfig(resolution=64, channels=("Y", "Cb", "Cr"), chroma_weight=1.0, chroma_downsample=1),
    ])
