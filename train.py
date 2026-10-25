"""Train one stage of the cascade on MNIST.

  python train.py --stage 1 --steps 2000
  python train.py --stage 2 --steps 2000 --color

Per-stage prompts (text conditioning only, `--cond text`). Prompts are
templates filled per training image: {digit} -> "seven", {label} -> 7,
{color} -> the synthetic colour name with --color ("gray" without it).

  python train.py --stage 1 --cond text --color --prompt "a handwritten digit {digit}"
  python train.py --stage 2 --cond text --color \
      --prompt-stage2 "a {color} handwritten {digit} with thick smooth strokes"

Fallback: stage K uses --prompt-stageK, else --prompt, else the default
"a handwritten digit {digit}" (identical to the previous behaviour).
"""
import argparse
import json
import os
import time

import torch

from c2f import Cascade, default_config, format_prompt
from c2f.data import COLOR_NAMES, make_loader


def prompt_template(args, stage: int):
    """Template for this stage after fallback, or None (= original label-derived prompts)."""
    return getattr(args, f"prompt_stage{stage}") or args.prompt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, choices=[1, 2, 3], required=True)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--subset", type=int, default=0, help="use only the first N MNIST images (0 = all)")
    ap.add_argument("--color", action="store_true", help="synthetic-colourised MNIST (else grayscale; chroma = neutral)")
    ap.add_argument("--cond", choices=["label", "text"], default="label")
    ap.add_argument("--prompt", default=None,
                    help="base prompt template for every stage (--cond text); placeholders {digit} {label} {color}")
    for s in (1, 2, 3):
        ap.add_argument(f"--prompt-stage{s}", default=None,
                        help=f"prompt template for stage {s} only (falls back to --prompt)")
    ap.add_argument("--data", default="data")
    ap.add_argument("--ckpt-dir", default="checkpoints")
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    template = prompt_template(args, args.stage)
    if args.cond == "label" and (args.prompt or any(getattr(args, f"prompt_stage{s}") for s in (1, 2, 3))):
        ap.error("--prompt/--prompt-stageN need --cond text (label conditioning uses the MNIST labels)")

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = default_config()
    cfg.cond_type = args.cond
    model = Cascade(cfg).to(device)
    k = args.stage - 1
    params = list(model.stages[k].parameters()) + list(model.encoders[k].parameters())
    opt = torch.optim.AdamW(params, lr=args.lr)
    n_params = sum(p.numel() for p in params)
    top = cfg.stages[-1].resolution
    loader = make_loader(args.data, size=top, color=args.color, subset=args.subset or None, batch_size=args.batch_size)
    print(f"stage {args.stage}: res={cfg.stages[k].resolution} channels={cfg.stages[k].channels} "
          f"params={n_params:,} device={device}")
    if template is not None:
        print(f"stage {args.stage} prompt template: {template!r}  "
              f"(e.g. {format_prompt(template, 7, COLOR_NAMES[7] if args.color else 'gray')!r})")

    step, t0, losses = 0, time.time(), []
    model.train()
    while step < args.steps:
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            if template is None:
                loss = model.stage_loss(k, x, y)
            else:
                prompts = [format_prompt(template, l, COLOR_NAMES[l] if args.color else "gray") for l in y.tolist()]
                loss = model.stage_loss(k, x, y, stage_prompts={args.stage: prompts})
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(loss.item())
            step += 1
            if step % args.log_every == 0 or step == 1 or step == args.steps:
                print(f"step {step:6d}  loss {loss.item():.4f}  ({time.time() - t0:.1f}s)", flush=True)
            if step >= args.steps:
                break

    os.makedirs(args.ckpt_dir, exist_ok=True)
    path = os.path.join(args.ckpt_dir, f"stage{args.stage}.pt")
    torch.save({"config": cfg.to_dict(), "stage": args.stage, "state": model.stage_state(k),
                "args": vars(args), "prompt_template": template}, path)
    with open(os.path.join(args.ckpt_dir, f"stage{args.stage}_losses.json"), "w") as f:
        json.dump(losses, f)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
