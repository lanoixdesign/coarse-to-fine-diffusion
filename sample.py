"""Sample the full cascade (16 gray -> 32 -> 64 colour) from per-stage checkpoints.

  python sample.py --labels 0 1 2 3 4 5 6 7 8 9 --steps 50 --out samples.png

Per-stage prompts (checkpoints trained with --cond clip or --cond text). The encoder
type and CLIP model id are read from the checkpoint; with CLIP the frozen text
model is loaded on startup (--clip-model overrides the id, e.g. a local path):

  python sample.py --steps 50 --num-samples 4 \
      --prompt "a handwritten digit seven" \
      --prompt-s2 "a bold red handwritten seven with thick smooth strokes on a pale blue background" \
      --prompt-s3 "a bold red handwritten seven with thick smooth strokes on a pale blue background"

Fallback: stage K uses --prompt-sK, else --prompt, else the label-derived
prompt "a handwritten digit {digit}" for each --labels entry (old behaviour).
With --labels, prompts may use the placeholders {digit}, {label} and {color}.

EMA weights (checkpoints trained with --ema-decay > 0 store them under "ema_state"):
by default each stage uses its EMA weights if its checkpoint has them, else the raw
weights (old checkpoints sample exactly as before). --use-ema requires EMA weights
for every loaded stage; --no-ema always uses the raw weights.
"""
import argparse
import os
import sys

import torch
import torch.nn.functional as F
from torchvision.utils import save_image

from c2f import (CLIPUnavailableError, Cascade, CascadeConfig, format_prompt, load_clip_backbone,
                 stage_state_from_checkpoint, ycbcr_to_rgb)
from c2f.text import DEFAULT_CLIP_MODEL
from c2f.cascade import to_ycbcr3
from c2f.color import from_model_range
from c2f.data import COLOR_NAMES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", default="checkpoints")
    ap.add_argument("--labels", type=int, nargs="+", default=None,
                    help="digit labels, one sample each (default 0..9 unless prompts are given)")
    ap.add_argument("--prompt", default=None, help="base prompt for every stage (text checkpoints)")
    for s in (1, 2, 3):
        ap.add_argument(f"--prompt-s{s}", default=None, help=f"prompt for stage {s} only (falls back to --prompt)")
    ap.add_argument("--num-samples", type=int, default=4,
                    help="number of samples when prompts are given without --labels")
    ap.add_argument("--steps", type=int, default=50, help="DDIM steps per stage")
    ap.add_argument("--out", default="samples.png")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--clip-model", default=None,
                    help="override the CLIP model id/path stored in the checkpoint (--cond clip checkpoints), "
                         "e.g. a local directory for offline use")
    ema = ap.add_mutually_exclusive_group()
    ema.add_argument("--use-ema", dest="use_ema", action="store_const", const=True, default=None,
                     help="sample with the EMA weights (error if a checkpoint has none). "
                          "Default: EMA weights when the checkpoint has them, else raw weights")
    ema.add_argument("--no-ema", dest="use_ema", action="store_const", const=False,
                     help="sample with the raw (non-EMA) weights")
    args = ap.parse_args()

    stage_args = {s: getattr(args, f"prompt_s{s}") for s in (1, 2, 3)}
    any_prompt = args.prompt is not None or any(v is not None for v in stage_args.values())
    if args.labels is None and not any_prompt:
        args.labels = list(range(10))

    torch.manual_seed(args.seed)
    first = torch.load(os.path.join(args.ckpt_dir, "stage1.pt"), map_location="cpu")
    cfg = CascadeConfig.from_dict(first["config"])
    if any_prompt and cfg.cond_type == "label":
        ap.error(f"--prompt/--prompt-sN need checkpoints trained with --cond clip or --cond text "
                 f"(these use {cfg.cond_type!r})")
    backbone = None
    if cfg.cond_type == "clip":
        if args.clip_model:
            cfg.clip_model = args.clip_model
        model_id = cfg.clip_model or DEFAULT_CLIP_MODEL
        try:
            backbone = load_clip_backbone(model_id)
        except CLIPUnavailableError as e:
            sys.exit(f"error: these checkpoints use CLIP text conditioning. {e}")
        print(f"encoder: frozen CLIP {model_id!r} + per-stage adapters (from checkpoint)")
    else:
        print(f"encoder: {cfg.cond_type}")
    model = Cascade(cfg, text_backbone=backbone)
    n = 0
    for k in range(len(cfg.stages)):
        path = os.path.join(args.ckpt_dir, f"stage{k + 1}.pt")
        if not os.path.exists(path):
            break
        try:
            state, used_ema = stage_state_from_checkpoint(torch.load(path, map_location="cpu"), args.use_ema)
        except KeyError as e:
            sys.exit(f"error: {path}: {e.args[0]}")
        model.load_stage_state(k, state)
        print(f"stage {k + 1}: {'EMA' if used_ema else 'raw'} weights")
        n = k + 1
    print(f"loaded {n} stage(s)")
    model.eval()

    labels = None if args.labels is None else torch.tensor(args.labels)
    batch = len(args.labels) if args.labels is not None else args.num_samples

    trained_color = bool(first.get("args", {}).get("color", False))

    def expand(p):  # fill {digit}/{label}/{color} per sample when labels are given
        if p is None or labels is None:
            return p
        return [format_prompt(p, l, COLOR_NAMES[l] if trained_color else "gray") for l in args.labels]

    prompt = expand(args.prompt)
    stage_prompts = {s: expand(p) for s, p in stage_args.items() if p is not None} or None
    if any_prompt:
        for k, c in enumerate(model.stage_conditions(labels, prompt, stage_prompts, batch=batch, upto=n)):
            print(f"stage {k + 1} prompt: {c[0]!r}" + (" ..." if len(set(c)) > 1 else ""))
    outs = model.sample(labels, steps=args.steps, upto=n, prompt=prompt, stage_prompts=stage_prompts,
                        num_samples=batch)

    # One row per stage, all shown at the final resolution (nearest upsampling).
    top = cfg.stages[n - 1].resolution
    rows = []
    for k, x in enumerate(outs):
        rgb = ycbcr_to_rgb(from_model_range(to_ycbcr3(x, cfg.stages[k].channels)))
        rows.append(F.interpolate(rgb, size=(top, top), mode="nearest"))
    save_image(torch.cat(rows), args.out, nrow=batch)
    print(f"saved {args.out}  (rows = stages 1..{n})")


if __name__ == "__main__":
    main()
