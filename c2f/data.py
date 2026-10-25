"""MNIST loaders: grayscale (chroma neutral -> colour stages are a no-op) or
synthetic-colourised (class-dependent foreground colour on a tinted background)."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

from .color import rgb_to_ycbcr, to_model_range

# One RGB foreground colour per digit class, so colour is predictable from the label.
PALETTE = torch.tensor([
    [0.90, 0.10, 0.10], [0.95, 0.55, 0.05], [0.95, 0.90, 0.10], [0.20, 0.80, 0.20], [0.10, 0.70, 0.70],
    [0.15, 0.35, 0.95], [0.55, 0.20, 0.90], [0.95, 0.30, 0.70], [0.55, 0.35, 0.15], [0.90, 0.90, 0.90],
])
# Human-readable name of each PALETTE entry (for `{color}` in prompt templates).
COLOR_NAMES = ["red", "orange", "yellow", "green", "teal", "blue", "purple", "pink", "brown", "white"]


class C2FMNIST(Dataset):
    """Returns (ycbcr in [-1,1] at `size`, label)."""

    def __init__(self, root: str = "data", train: bool = True, size: int = 64, color: bool = True, download: bool = True,
                 with_gray: bool = False):
        """`with_gray=True`: items are (ycbcr, label, ycbcr_gray), where ycbcr_gray is the same digit
        rendered in grayscale (white on black, neutral chroma), e.g. to condition stage 2 on the
        grayscale structure that a gray-trained stage 1 produces."""
        from torchvision import datasets, transforms
        self.ds = datasets.MNIST(root, train=train, download=download, transform=transforms.ToTensor())
        self.size, self.color, self.with_gray = size, color, with_gray

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        img, label = self.ds[i]  # (1,28,28) in [0,1]
        img = F.interpolate(img[None], size=(self.size, self.size), mode="bilinear", align_corners=False)[0]
        if self.color:
            g = torch.Generator().manual_seed(i)
            bg = 0.15 * torch.rand(3, 1, 1, generator=g)
            rgb = bg * (1 - img) + PALETTE[label][:, None, None] * img
        else:
            rgb = img.expand(3, -1, -1)
        ycc = rgb_to_ycbcr(rgb[None])[0]
        if self.with_gray:
            gray = to_model_range(rgb_to_ycbcr(img.expand(3, -1, -1)[None])[0])
            return to_model_range(ycc), label, gray
        return to_model_range(ycc), label


def make_loader(root="data", size=64, color=True, subset=None, batch_size=64, num_workers=0, download=True,
                with_gray=False):
    ds = C2FMNIST(root, True, size, color, download, with_gray=with_gray)
    if subset:
        ds = Subset(ds, range(min(subset, len(ds))))
    return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
