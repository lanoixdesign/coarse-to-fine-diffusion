"""Colour-space helpers (full-range BT.601 YCbCr). Tensors are (B, 3, H, W)."""
import torch


def rgb_to_ycbcr(rgb: torch.Tensor) -> torch.Tensor:
    """RGB in [0,1] -> YCbCr in [0,1] (Cb/Cr centred at 0.5)."""
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = 0.5 - 0.168736 * r - 0.331264 * g + 0.5 * b
    cr = 0.5 + 0.5 * r - 0.418688 * g - 0.081312 * b
    return torch.stack([y, cb, cr], dim=1)


def ycbcr_to_rgb(ycc: torch.Tensor) -> torch.Tensor:
    """YCbCr in [0,1] -> RGB in [0,1] (clamped)."""
    y, cb, cr = ycc[:, 0], ycc[:, 1] - 0.5, ycc[:, 2] - 0.5
    r = y + 1.402 * cr
    g = y - 0.344136 * cb - 0.714136 * cr
    b = y + 1.772 * cb
    return torch.stack([r, g, b], dim=1).clamp(0.0, 1.0)


def to_model_range(x: torch.Tensor) -> torch.Tensor:
    """[0,1] -> [-1,1]. For Cb/Cr, 0 then means 'neutral grey'."""
    return x * 2.0 - 1.0


def from_model_range(x: torch.Tensor) -> torch.Tensor:
    return ((x + 1.0) / 2.0).clamp(0.0, 1.0)
