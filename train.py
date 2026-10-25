"""Train one stage of the cascade on MNIST.

  python train.py --stage 1 --steps 2000
  python train.py --stage 2 --steps 2000 --color

Text conditioning: `--cond clip` (frozen CLIP text encoder, downloaded from the
Hugging Face Hub on first use; `--clip-model` picks the model id or a local path)
or `--cond text` (alias `stub`: hashed bag-of-words stub, offline, no download).

Per-stage prompts (text conditioning only). Prompts are
templates filled per training image: {digit} -> "seven", {label} -> 7,
{color} -> the synthetic colour name with --color ("gray" without it).

  python train.py --stage 1 --cond clip --color --prompt "a handwritten digit {digit}"
  python train.py --stage 2 --cond clip --color \
      --prompt-stage2 "a {color} handwritten {digit} with thick smooth strokes"

Fallback: stage K uses --prompt-stageK, else --prompt, else the default
"a handwritten digit {digit}" (identical to the previous behaviour).

Long runs (all off by default; with the defaults training is identical to before):

  python train.py --stage 1 --steps 30000 --batch-size 128 --ema-decay 0.999 \
      --warmup-steps 500 --lr-schedule cosine --sample-every 5000 --save-every 5000 \
      --ckpt-dir checkpoints_long --sample-dir samples_long
  python train.py --stage 1 --steps 30000 ... --resume          # continue from checkpoints_long/stage1.pt

  --ema-decay D     EMA of the stage weights (UNet + encoder), decay warmup min(D, (1+n)/(10+n)).
                    Saved under "ema_state"; "state" stays the raw weights.
  --warmup-steps N  linear LR warmup from 0; --lr-schedule constant|cosine|linear decays to
                    lr * --min-lr-ratio at --steps.
  --sample-every N  every N steps (and at the end) sample 2 images per digit 0-9 with the EMA
                    weights (raw if no EMA), --sample-steps DDIM steps, fixed noise seed, and save
                    a nearest-upscaled PNG to --sample-dir. Stages 2/3 condition on the real
                    low-res images of 20 fixed training images (top row of the PNG).
  --save-every N    save the resumable checkpoint (raw + EMA + optimizer) every N steps, plus a
                    weights-only snapshot stageK_stepNNNNNN.pt.
  Loss log: ckpt-dir/stageK_log.json (step, loss, mean loss over the last --log-every steps, lr).
"""
import argparse
import json
import math
import os
import sys
import time

import torch
import torch.nn.functional as F
from torchvision.utils import save_image

from c2f import (CLIPUnavailableError, DEFAULT_CLIP_MODEL, Cascade, StageEMA, default_config, format_prompt,
                 load_clip_backbone, ycbcr_to_rgb)
from c2f.cascade import build_condition, stage_target, to_ycbcr3
from c2f.color import from_model_range
from c2f.text import encoder_info
from c2f.data import COLOR_NAMES, make_loader


def prompt_template(args, stage: int):
    """Template for this stage after fallback, or None (= original label-derived prompts)."""
    return getattr(args, f"prompt_stage{stage}") or args.prompt


def lr_factor(step: int, total: int, warmup: int, schedule: str, min_ratio: float) -> float:
    """LR multiplier for optimiser step `step` (1-based). constant + warmup 0 -> always 1.0."""
    if warmup > 0 and step <= warmup:
        return step / warmup
    if schedule == "constant":
        return 1.0
    frac = min(1.0, (step - warmup) / max(1, total - warmup))
    if schedule == "cosine":
        return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * frac))
    return min_ratio + (1 - min_ratio) * (1 - frac)  # linear


def sample_grid_inputs(model, k, dataset, per_label=2, num_classes=10):
    """Labels [0..9] * per_label, and for stage k > 0 the real stage-(k-1) images of fixed
    training images with those labels (the conditioning for stage k)."""
    labels = torch.arange(num_classes).repeat(per_label)
    if k == 0:
        return labels, None
    found = {c: [] for c in range(num_classes)}
    for i in range(len(dataset)):
        x, y = dataset[i]
        if len(found[y]) < per_label:
            found[y].append(x)
        if all(len(v) == per_label for v in found.values()):
            break
    labels = torch.tensor([c for r in range(per_label) for c in range(num_classes) if len(found[c]) > r])
    xs = torch.stack([found[c][r] for r in range(per_label) for c in range(num_classes) if len(found[c]) > r])
    p = model.cfg.stages[k - 1]
    return labels, stage_target(F.interpolate(xs, size=(p.resolution,) * 2, mode="area"), p)


@torch.no_grad()
def save_sample_grid(model, k, labels, prev_real, path, steps, seed, tile, prompts=None):
    """Sample stage k for `labels` (fixed noise seed; the training RNG is untouched)."""
    s = model.cfg.stages[k]
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        emb = model.encode_stage(k, labels.to(device), stage_prompts=None if prompts is None else {k + 1: prompts})
        cond = None if prev_real is None else build_condition(prev_real.to(device), s.resolution)
        x = model.diffusion.sample(model.stages[k], (len(labels), s.n_out, s.resolution, s.resolution),
                                   cond, emb, steps=steps, device=device)
    model.train(was_training)

    def rgb(img, ch):
        return F.interpolate(ycbcr_to_rgb(from_model_range(to_ycbcr3(img.cpu(), ch))), size=(tile, tile),
                             mode="nearest")
    rows = [rgb(x, s.channels)]
    if prev_real is not None:  # first row: the real low-res conditioning images
        rows.insert(0, rgb(prev_real, model.cfg.stages[k - 1].channels))
    save_image(torch.cat(rows), path, nrow=10, padding=2, pad_value=0.5)
    return x


def atomic_save(obj, path):
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, choices=[1, 2, 3], required=True)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--subset", type=int, default=0, help="use only the first N MNIST images (0 = all)")
    ap.add_argument("--color", action="store_true", help="synthetic-colourised MNIST (else grayscale; chroma = neutral)")
    ap.add_argument("--cond", choices=["label", "clip", "text", "stub"], default="label",
                    help="label: MNIST class embedding; clip: frozen CLIP text encoder + per-stage adapter "
                         "(recommended for text prompts); text/stub: offline hashed bag-of-words stub")
    ap.add_argument("--clip-model", default=DEFAULT_CLIP_MODEL,
                    help="Hugging Face model id or local directory of the CLIP text model (--cond clip)")
    ap.add_argument("--prompt", default=None,
                    help="base prompt template for every stage (--cond clip/text); placeholders {digit} {label} {color}")
    for s in (1, 2, 3):
        ap.add_argument(f"--prompt-stage{s}", default=None,
                        help=f"prompt template for stage {s} only (falls back to --prompt)")
    ap.add_argument("--data", default="data")
    ap.add_argument("--ckpt-dir", default="checkpoints")
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    # long-run options (defaults = previous behaviour)
    ap.add_argument("--ema-decay", type=float, default=0.0,
                    help="EMA decay of the stage weights (0 = off; e.g. 0.999). Saved as 'ema_state'")
    ap.add_argument("--no-ema-warmup", action="store_true",
                    help="use the constant EMA decay from step 1 (default: min(decay, (1+n)/(10+n)))")
    ap.add_argument("--warmup-steps", type=int, default=0, help="linear LR warmup steps (0 = off)")
    ap.add_argument("--lr-schedule", choices=["constant", "cosine", "linear"], default="constant",
                    help="LR decay after warmup, down to lr * --min-lr-ratio at --steps")
    ap.add_argument("--min-lr-ratio", type=float, default=0.0)
    ap.add_argument("--sample-every", type=int, default=0,
                    help="every N steps sample 2 images per digit (EMA weights) to --sample-dir (0 = off)")
    ap.add_argument("--sample-dir", default="samples")
    ap.add_argument("--sample-steps", type=int, default=50, help="DDIM steps for --sample-every grids")
    ap.add_argument("--sample-tile", type=int, default=64, help="tile size (nearest upscaling) in sample PNGs")
    ap.add_argument("--save-every", type=int, default=0,
                    help="save a resumable checkpoint every N steps (0 = only at the end)")
    ap.add_argument("--resume", nargs="?", const="auto", default=None,
                    help="resume from a checkpoint (default path: ckpt-dir/stageK.pt)")
    args = ap.parse_args()
    if args.cond == "stub":
        args.cond = "text"

    template = prompt_template(args, args.stage)
    if args.cond == "label" and (args.prompt or any(getattr(args, f"prompt_stage{s}") for s in (1, 2, 3))):
        ap.error("--prompt/--prompt-stageN need --cond clip or --cond text (label conditioning uses the MNIST labels)")

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = default_config()
    cfg.cond_type = args.cond
    backbone = None
    if args.cond == "clip":
        cfg.clip_model = args.clip_model
        t_load = time.time()
        try:
            backbone = load_clip_backbone(args.clip_model)
        except CLIPUnavailableError as e:
            sys.exit(f"error: {e}")
        print(f"loaded frozen CLIP text encoder {args.clip_model!r} ({backbone.out_dim}-d, "
              f"{time.time() - t_load:.1f}s); only the stage-{args.stage} adapter is trained")
    model = Cascade(cfg, text_backbone=backbone).to(device)
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

    ema = StageEMA(model, k, args.ema_decay, warmup=not args.no_ema_warmup) if args.ema_decay > 0 else None
    scheduled = args.warmup_steps > 0 or args.lr_schedule != "constant"
    os.makedirs(args.ckpt_dir, exist_ok=True)
    path = os.path.join(args.ckpt_dir, f"stage{args.stage}.pt")
    log_path = os.path.join(args.ckpt_dir, f"stage{args.stage}_log.json")
    log = {"args": vars(args), "records": [], "samples": []}

    step, t0, losses = 0, time.time(), []
    elapsed0 = 0.0
    if args.resume:
        rpath = path if args.resume == "auto" else args.resume
        ck = torch.load(rpath, map_location="cpu")
        if ck.get("stage") != args.stage or "optimizer" not in ck:
            sys.exit(f"error: {rpath} is not a resumable stage-{args.stage} checkpoint")
        model.load_stage_state(k, ck["state"])
        opt.load_state_dict(ck["optimizer"])
        step = int(ck["step"])
        if ema is not None:
            if ck.get("ema_state") is not None:
                ema.load(ck["ema_state"], ck.get("ema"))
            else:
                print("warning: checkpoint has no EMA weights; EMA restarts from the current raw weights")
        lp = os.path.join(args.ckpt_dir, f"stage{args.stage}_losses.json")
        if os.path.exists(lp):
            with open(lp) as f:
                losses = json.load(f)[:step]
        if os.path.exists(log_path):
            with open(log_path) as f:
                old = json.load(f)
            log["records"] = [r for r in old.get("records", []) if r["step"] <= step]
            log["samples"] = [r for r in old.get("samples", []) if r["step"] <= step]
        elapsed0 = float(ck.get("elapsed", 0.0))
        torch.manual_seed(args.seed + step)  # new data order after resuming (not bit-identical to a single run)
        print(f"resumed from {rpath} at step {step}")

    grid_inputs = None
    if args.sample_every > 0:
        os.makedirs(args.sample_dir, exist_ok=True)
        with torch.random.fork_rng(devices=[]):
            grid_inputs = sample_grid_inputs(model, k, loader.dataset)
        grid_prompts = None
        if template is not None:
            grid_prompts = [format_prompt(template, l, COLOR_NAMES[l] if args.color else "gray")
                            for l in grid_inputs[0].tolist()]

    def checkpoint(final: bool):
        ck = {"config": cfg.to_dict(), "stage": args.stage, "state": model.stage_state(k),
              "args": vars(args), "prompt_template": template, "encoder": encoder_info(cfg),
              "step": step, "optimizer": opt.state_dict(), "elapsed": elapsed0 + time.time() - t0}
        if ema is not None:
            ck["ema_state"], ck["ema"] = ema.state_dict(), ema.meta()
        atomic_save(ck, path)
        with open(os.path.join(args.ckpt_dir, f"stage{args.stage}_losses.json"), "w") as f:
            json.dump(losses, f)
        write_log()
        if not final:
            snap = {key: v for key, v in ck.items() if key != "optimizer"}
            atomic_save(snap, os.path.join(args.ckpt_dir, f"stage{args.stage}_step{step:06d}.pt"))
            print(f"checkpoint step {step}: {path}", flush=True)

    def write_log():
        with open(log_path + ".tmp", "w") as f:
            json.dump(log, f)
        os.replace(log_path + ".tmp", log_path)

    def do_sample():
        out = os.path.join(args.sample_dir, f"stage{args.stage}_step{step:06d}.png")
        ts = time.time()
        if ema is not None:
            with ema.swapped_in():
                x = save_sample_grid(model, k, *grid_inputs, out, args.sample_steps, args.seed + 1234,
                                     args.sample_tile, grid_prompts)
        else:
            x = save_sample_grid(model, k, *grid_inputs, out, args.sample_steps, args.seed + 1234,
                                 args.sample_tile, grid_prompts)
        log["samples"].append({"step": step, "path": out, "weights": "ema" if ema is not None else "raw",
                               "ddim_steps": args.sample_steps, "mean": x.mean().item(), "std": x.std().item()})
        write_log()
        print(f"samples step {step}: {out} ({'EMA' if ema is not None else 'raw'} weights, "
              f"{args.sample_steps} DDIM steps, {time.time() - ts:.1f}s)", flush=True)

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
            if scheduled:
                for g in opt.param_groups:
                    g["lr"] = args.lr * lr_factor(step + 1, args.steps, args.warmup_steps, args.lr_schedule,
                                                  args.min_lr_ratio)
            opt.step()
            if ema is not None:
                ema.update()
            losses.append(loss.item())
            step += 1
            if step % args.log_every == 0 or step == 1 or step == args.steps:
                recent = losses[-args.log_every:]
                avg = sum(recent) / len(recent)
                lr_now = opt.param_groups[0]["lr"]
                el = elapsed0 + time.time() - t0
                extra = f"  avg{len(recent)} {avg:.4f}  lr {lr_now:.2e}" if (scheduled or ema or args.sample_every
                                                                             or args.save_every) else ""
                print(f"step {step:6d}  loss {loss.item():.4f}{extra}  ({time.time() - t0:.1f}s)", flush=True)
                log["records"].append({"step": step, "loss": loss.item(), "loss_avg": avg, "lr": lr_now,
                                       "elapsed_s": round(el, 1)})
                if extra:
                    write_log()
            if args.save_every > 0 and step % args.save_every == 0 and step < args.steps:
                checkpoint(final=False)
            if args.sample_every > 0 and step % args.sample_every == 0:
                do_sample()
            if step >= args.steps:
                break

    checkpoint(final=True)
    if args.sample_every > 0 and step % args.sample_every != 0:
        do_sample()
    if backbone is not None:
        info = backbone.cache_info()
        print(f"CLIP cache: {info['entries']} unique prompts, {info['forward_passes']} CLIP forward passes")
    print(f"saved {path}")


if __name__ == "__main__":
    main()
