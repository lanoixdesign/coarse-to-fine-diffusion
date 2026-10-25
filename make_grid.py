"""Sample the cascade once and save two figures from the SAME sampling chain.

  python make_grid.py --ckpt-dir checkpoints --steps 50 --seed 0 --out-dir docs/samples

1. grid_64x64_8x8.png    : 64 final 64x64 RGB samples, 8x8, labels = i % 10 (row-major).
2. progressive_stages.png: the first 10 samples (labels 0..9). Rows = stage 1 (16x16 Y),
   stage 2 (32x32 YCbCr), stage 3 (64x64 YCbCr), each nearest-upsampled to the same
   display size. The stage-1/2 images are the intermediate outputs of the same
   `Cascade.sample` call that produced the stage-3 images (stage k+1 conditions on them),
   not independent runs.

EMA: like sample.py, each stage uses its EMA weights if the checkpoint has them
(--no-ema forces the raw weights; --use-ema requires EMA weights).

Label-conditioned checkpoints only (prompts are not exposed here; use sample.py for those).
"""
import argparse
import os
import time

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from torchvision.utils import make_grid

from c2f import Cascade, CascadeConfig, stage_state_from_checkpoint, ycbcr_to_rgb
from c2f.cascade import to_ycbcr3
from c2f.color import from_model_range


def to_rgb(x, channels):
    return ycbcr_to_rgb(from_model_range(to_ycbcr3(x, channels))).clamp(0, 1)


def to_pil(t):
    return Image.fromarray((t.permute(1, 2, 0).numpy() * 255).round().astype("uint8"))


def font(size):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf"):
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", default="checkpoints")
    ap.add_argument("--steps", type=int, default=50, help="DDIM steps per stage")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=64, help="number of samples (8x8 grid)")
    ap.add_argument("--nrow", type=int, default=8)
    ap.add_argument("--display", type=int, default=128, help="tile size in the progressive figure")
    ap.add_argument("--out-dir", default="docs/samples")
    ema = ap.add_mutually_exclusive_group()
    ema.add_argument("--use-ema", dest="use_ema", action="store_const", const=True, default=None,
                     help="EMA weights (error if absent); default: EMA if present, else raw")
    ema.add_argument("--no-ema", dest="use_ema", action="store_const", const=False, help="raw weights")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    torch.manual_seed(args.seed)
    first = torch.load(os.path.join(args.ckpt_dir, "stage1.pt"), map_location="cpu")
    cfg = CascadeConfig.from_dict(first["config"])
    assert cfg.cond_type == "label", "make_grid.py is for label-conditioned checkpoints"
    model = Cascade(cfg)
    for k in range(len(cfg.stages)):
        state, used_ema = stage_state_from_checkpoint(
            torch.load(os.path.join(args.ckpt_dir, f"stage{k + 1}.pt"), map_location="cpu"), args.use_ema)
        model.load_stage_state(k, state)
        print(f"stage {k + 1}: {'EMA' if used_ema else 'raw'} weights")
    model.eval()

    labels = torch.arange(args.n) % 10
    t0 = time.time()
    outs = model.sample(labels, steps=args.steps)  # one sequential chain: [16x16 Y, 32x32 YCbCr, 64x64 YCbCr]
    print(f"sampled {args.n} x {len(outs)} stages, {args.steps} DDIM steps/stage, seed {args.seed}: {time.time() - t0:.1f}s")
    rgbs = [to_rgb(x, s.channels) for x, s in zip(outs, cfg.stages)]

    # 1. main grid of final outputs
    grid = make_grid(rgbs[-1], nrow=args.nrow, padding=2, pad_value=1.0)
    p1 = os.path.join(args.out_dir, "grid_64x64_8x8.png")
    to_pil(grid).save(p1)
    print("saved", p1, "labels row-major:", labels.tolist())

    # 2. progressive figure: first sample of each digit 0..9 from the same chain
    idx = [int((labels == d).nonzero()[0]) for d in range(10)]
    D, pad, left, top = args.display, 6, 170, 34
    names = [f"stage {k + 1}\n{s.resolution}x{s.resolution} {'gray' if s.channels == ('Y',) or list(s.channels) == ['Y'] else 'colour'}"
             for k, s in enumerate(cfg.stages)]
    W = left + len(idx) * (D + pad) + pad
    H = top + len(rgbs) * (D + pad) + pad
    img = Image.new("RGB", (W, H), "white")
    dr = ImageDraw.Draw(img)
    f = font(20)
    for j, i in enumerate(idx):
        dr.text((left + pad + j * (D + pad) + D // 2 - 6, 6), str(int(labels[i])), fill="black", font=f)
    for r, x in enumerate(rgbs):
        y = top + pad + r * (D + pad)
        dr.multiline_text((10, y + D // 2 - 24), names[r], fill="black", font=f, spacing=6)
        up = F.interpolate(x[idx], size=(D, D), mode="nearest")
        for j in range(len(idx)):
            img.paste(to_pil(up[j]), (left + pad + j * (D + pad), y))
    p2 = os.path.join(args.out_dir, "progressive_stages.png")
    img.save(p2)
    print("saved", p2, "sample indices:", idx)


if __name__ == "__main__":
    main()
