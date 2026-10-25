"""Sample the full cascade (16 gray -> 32 -> 64 colour) from per-stage checkpoints.

  python sample.py --labels 0 1 2 3 4 5 6 7 8 9 --steps 50 --out samples.png
"""
import argparse
import os

import torch
import torch.nn.functional as F
from torchvision.utils import save_image

from c2f import Cascade, CascadeConfig, ycbcr_to_rgb
from c2f.cascade import to_ycbcr3
from c2f.color import from_model_range


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", default="checkpoints")
    ap.add_argument("--labels", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--steps", type=int, default=50, help="DDIM steps per stage")
    ap.add_argument("--out", default="samples.png")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    first = torch.load(os.path.join(args.ckpt_dir, "stage1.pt"), map_location="cpu")
    cfg = CascadeConfig.from_dict(first["config"])
    model = Cascade(cfg)
    n = 0
    for k in range(len(cfg.stages)):
        path = os.path.join(args.ckpt_dir, f"stage{k + 1}.pt")
        if not os.path.exists(path):
            break
        model.load_stage_state(k, torch.load(path, map_location="cpu")["state"])
        n = k + 1
    print(f"loaded {n} stage(s)")
    model.eval()
    outs = model.sample(torch.tensor(args.labels), steps=args.steps, upto=n)

    # One row per stage, all shown at the final resolution (nearest upsampling).
    top = cfg.stages[n - 1].resolution
    rows = []
    for k, x in enumerate(outs):
        rgb = ycbcr_to_rgb(from_model_range(to_ycbcr3(x, cfg.stages[k].channels)))
        rows.append(F.interpolate(rgb, size=(top, top), mode="nearest"))
    save_image(torch.cat(rows), args.out, nrow=len(args.labels))
    print(f"saved {args.out}  (rows = stages 1..{n})")


if __name__ == "__main__":
    main()
