"""c2f: coarse-to-fine, luminance-first, text-conditioned diffusion cascade (prototype)."""

from .config import CascadeConfig, StageConfig, default_config
from .color import rgb_to_ycbcr, ycbcr_to_rgb
from .diffusion import Diffusion
from .unet import UNet
from .text import (LabelEncoder, HashTextEncoder, build_encoder, labels_to_prompts,
                   normalize_stage_prompts, format_prompt)
from .cascade import Cascade, build_condition

__all__ = [
    "CascadeConfig", "StageConfig", "default_config",
    "rgb_to_ycbcr", "ycbcr_to_rgb", "Diffusion", "UNet",
    "LabelEncoder", "HashTextEncoder", "build_encoder", "labels_to_prompts",
    "normalize_stage_prompts", "format_prompt",
    "Cascade", "build_condition",
]
