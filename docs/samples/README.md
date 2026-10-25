# Samples: label-conditioned cascade, 2000 steps/stage (2026-10-25)

Two figures from **one** sampling chain (`make_grid.py`, repo root):

- `grid_64x64_8x8.png` — 64 final 64x64 RGB samples (stage-3 output), 8x8, 2 px white padding.
  Labels are `i % 10` in row-major order: row 1 is 0 1 2 3 4 5 6 7, row 2 is 8 9 0 1 2 3 4 5, and so on (the last sample is label 3).
- `progressive_stages.png` — the first 10 samples of that same chain (labels 0..9, one per column).
  Rows: stage 1 (16x16 grayscale Y), stage 2 (32x32 YCbCr), stage 3 (64x64 YCbCr), each converted to RGB and
  **nearest-neighbour** upsampled to 128x128 so the pixel size of each stage stays visible.
  The stage-1/2 images are the actual intermediate outputs that stage 2/3 were conditioned on (`Cascade.sample` returns every stage's output from one sequential run). They are not independent runs.

## Command

```bash
OMP_NUM_THREADS=8 python make_grid.py --ckpt-dir checkpoints --steps 100 --seed 0 --out-dir docs/samples
```

- Sampler: deterministic DDIM (eta = 0), **100 steps per stage**, T = 1000, no classifier-free guidance.
- Seed: `torch.manual_seed(0)` once before sampling, batch of 64 (labels `arange(64) % 10`).
- Runtime: about 35 s of sampling for 64 x 3 stages on CPU (8 threads), about 38 s wall time in total.
- Equivalent check with the stock script (stages as rows, unlabelled, batch of 10):
  `python sample.py --ckpt-dir checkpoints --labels 0 1 2 3 4 5 6 7 8 9 --steps 100 --seed 0 --out samples.png`.
  Stage 1 is identical to the progressive figure, because the first 10 initial noises match. Stages 2/3 differ slightly, because the batch size changes the later noise draws.

## Checkpoints (not in the repo)

- `--cond label` (MNIST class embedding), `default_config()`: 16x16 Y -> 32x32 Y+CbCr (chroma_weight 0.5, chroma low-passed x4) -> 64x64 Y+CbCr.
- Tiny UNets, about 690k parameters per stage, no attention.
- 2000 training steps per stage, batch 64, lr 2e-4, seed 0, full MNIST train set.
  Stage 1 was trained on gray MNIST (it is luminance-only). Stages 2 and 3 were trained with `--color`, i.e. synthetic-colourised MNIST: a fixed foreground colour per class (0 red, 1 orange, 2 yellow, 3 green, 4 teal, 5 blue, 6 purple, 7 pink, 8 brown, 9 white) on a near-black, slightly tinted background.
- Final training loss, averaged over the last 100 steps (from `stageK_losses.json`): stage 1 ≈ 0.039, stage 2 ≈ 0.0125, stage 3 ≈ 0.0085.
  These are epsilon-MSE values for different targets, so they are not comparable across stages and are not quality metrics.

## What the samples look like (visual inspection only, no metrics)

See the 2026-10-25 entry "First real samples" in [EXPERIMENTS.md](../../EXPERIMENTS.md).

> Note: the two PNGs were generated but not uploaded in this commit, because the GitHub connector used for the push only accepts text content. Regenerate them with the command above, or add them in a separate commit.
