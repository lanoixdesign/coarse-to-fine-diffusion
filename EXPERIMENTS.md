# Experiments log

Only record what was actually run and observed. No numbers that were not measured.

## Template

```
### YYYY-MM-DD — <short title>
- Stage(s):
- Config: (resolution, channels, chroma_weight, chroma_downsample, cond_noise_std, base_channels, timesteps)
- Dataset: (mnist gray / mnist --color / subset size)
- Command(s):
- Hardware:
- Loss curve notes: (see checkpoints/stageK_losses.json)
- Samples: (path / description)
- Observations:
- Next steps:
```

## 2026-10-25 — Smoke test (pipeline only, not a training run)

- Stage(s): 1, 2, 3, then full-cascade sampling.
- Config: `default_config()` — 16x16 Y; 32x32 Y+CbCr (chroma_weight 0.5, chroma_downsample 4); 64x64 Y+CbCr (chroma_weight 1.0). base_channels 32, T = 1000, label conditioning.
- Dataset: synthetic-colourised MNIST (`--color`), first 512 training images, downloaded via torchvision.
- Commands:
  ```
  python train.py --stage {1,2,3} --steps 20 --batch-size 16 --subset 512 --color
  python sample.py --steps 5 --labels 0 1 2 3
  python train.py --stage 2 --steps 5 --batch-size 16 --subset 512 --cond text   # text-stub path
  pytest -q tests                                                                 # 4 passed
  ```
- Hardware: CPU only (torch 2.x CPU build, Python 3.13).
- Loss curve notes: 20 steps per stage. The printed loss went down within those steps (stage 1: 1.07 → 0.23, stage 2: 1.13 → 0.61, stage 3: 1.10 → 0.49 at steps 1 → 20). With this few steps that only shows that gradients flow; it says nothing about model quality.
- Samples: the 4×3 grid (rows = stages) is unstructured noise, which is expected after 20 steps and 5 DDIM steps. Stage 1 is grey, stages 2–3 are coloured noise.
- Observations: every stage trains, saves a checkpoint, and the cascade chains 16 → 32 → 64 with the right shapes. The text-stub conditioning path also runs. No quality claims.
- Next steps: train stage 1 properly (thousands of steps, ideally on GPU) and look at 16x16 samples; then stage 2 conditioned on ground-truth 16x16 inputs; compare grayscale and synthetic-colour MNIST.
