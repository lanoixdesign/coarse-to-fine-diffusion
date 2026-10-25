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

## 2026-10-25 — Per-stage prompts (interface change, no training results)

- What: `Cascade.stage_loss` / `Cascade.sample` take an optional base `prompt` and per-stage `stage_prompts` (fallback: stage prompt → base prompt → labels). `train.py --prompt/--prompt-stage{1,2,3}`, `sample.py --prompt/--prompt-s{1,2,3}`. See README "Per-stage prompts".
- This is an **interface change only**. No training run, no quality or metric results. The text encoder is still the hashed bag-of-words stub, so per-stage prompts only exercise the plumbing.
- Checks run (CPU, small subset, temporary checkpoint dirs):
  - `pytest -q tests`: 16 passed (4 existing + 12 new in `tests/test_stage_prompts.py`).
  - The README smoke test (`train.py --stage {1,2,3} --steps 20 --batch-size 16 --subset 512 --color`, then `sample.py --steps 5 --labels 0 1 2 3`) gives loss histories identical to the previous code, and the sample PNG is byte-identical. `--cond text` without prompt flags is identical to the previous code too.
  - Text mode with per-stage prompt templates (20 steps per stage), then `sample.py --steps 5` with `--prompt` plus `--prompt-s2/--prompt-s3`. It runs end to end. A base prompt alone gives a PNG byte-identical to the same prompt passed to every stage.

## 2026-10-25 — CLIP text encoder (encoder/interface change, plumbing smoke test only)

- What: new `--cond clip` mode (`c2f.text.CLIPTextEncoder`). A frozen `openai/clip-vit-base-patch32` text tower (`CLIPTextModelWithProjection`, L2-normalised projected `text_embeds`, 512-d) is loaded **once** and shared by all stages. Each stage gets its own trainable `LayerNorm(512) -> Linear(512, 128)` adapter. The null embedding is CLIP(`""`) mapped by the stage adapter. Features are cached per unique prompt string. Checkpoints record the encoder type and model id, and store adapters only (no CLIP weights, about 3 MB per stage). `--cond text` (hashed stub) and `--cond label` are unchanged. See README "CLIP text encoder".
- This is an **encoder/interface change only**. Nothing was trained with CLIP beyond a few steps, and there are **no quality claims**.
- Checks run (CPU, `OMP_NUM_THREADS=2`, Python 3.13, torch 2.14.1+cpu, transformers 5.20.2; also tests with transformers 4.46.3):
  - `pytest -q tests`: 27 passed (16 existing + 11 new in `tests/test_clip.py`). With `HF_HUB_OFFLINE=1` and an empty HF cache (what CI does): 19 passed, 8 skipped (the CLIP-model tests, with a reason).
  - Smoke test (temporary checkpoint dir, 64-image `--color` subset, batch 16): `train.py --stage {1,2,3} --steps 5 --cond clip` with per-stage prompt templates, then `sample.py --steps 3` with `--prompt` / `--prompt-s2` / `--prompt-s3`, and with `--labels 0 1 2 3`. All commands completed and checkpoints saved and reloaded. Each stage used 11 CLIP forward passes in total (10 digit prompts + `""`) thanks to the cache. Printed losses over 5 steps (stage 1: 1.02 → 0.64, stage 2: 1.22 → 0.96, stage 3: 1.11 → 0.96) only show that gradients flow. The samples are noise, as expected.
  - Observed CLIP cosine similarities (L2-normalised `text_embeds`): "a photo of a red car" vs "a picture of a red automobile" 0.97, vs "a bowl of soup" 0.68; automobile vs soup 0.71.
  - Older label/stub checkpoints (written by the previous code) load and sample with the new `sample.py`.
- Next steps: an actual training run with `--cond clip` and per-stage prompts, compared against `--cond label`.

## 2026-10-25 — First real samples: label-conditioned cascade, 2000 steps/stage

- Stage(s): full cascade sampling only (no training in this entry).
- Checkpoints: `--cond label`, `default_config()`, about 690k-parameter UNets per stage, 2000 steps per stage (batch 64, lr 2e-4, seed 0). Stage 1 used gray MNIST. Stages 2 and 3 used `--color` (synthetic-colourised MNIST). Final training loss averaged over the last 100 steps: 0.039 / 0.0125 / 0.0085 (stages 1/2/3; different targets, not comparable).
- Command: `OMP_NUM_THREADS=8 python make_grid.py --ckpt-dir checkpoints --steps 100 --seed 0 --out-dir docs/samples`. DDIM eta = 0, 100 steps per stage, no guidance, 64 samples with labels `i % 10`. The first 10 samples (labels 0..9) are reused for the progressive figure, so all stages come from the same chain.
- Hardware / time: CPU, 8 threads. About 35 s of sampling (64 samples x 3 stages), about 38 s wall time.
- Samples: `docs/samples/grid_64x64_8x8.png` (8x8 final 64x64), `docs/samples/progressive_stages.png` (rows = stage 1/2/3, nearest upsampled). Details in `docs/samples/README.md`.
- Observations (by eye, no metrics):
  - **Digit identity is weak.** Every sample is a stroke-like, digit-ish shape on a background, but most do not read as the conditioned digit. By a rough count, about 8–10 of the 64 clearly look like the correct label (several 1s, 5s, 0s, a 6, 7s and a 9), and roughly another 8 are borderline. The rest are wrong, ambiguous or broken into disconnected stroke fragments. Label conditioning is visibly only partly learned after 2000 steps.
  - **Stage 1 (16x16 gray)** gives clean black backgrounds with blocky white strokes, so it is the least noisy stage. Its shapes are already wrong for most labels (in the progressive figure, the 0 looks like "b"/"6", the 1 is two dashes, the 4 is an arch). Only the 3 and the 9 are roughly plausible.
  - **Stages 2/3 follow stage 1's shape closely.** They mostly keep the layout and smooth/thicken the strokes. They do not fix wrong shapes, and sometimes they lose thin parts (e.g. the 5's top stroke fades away, leaving two blobs). So they neither clearly refine nor badly corrupt the structure. They mostly copy it, plus noise.
  - **Colour is not plausible relative to the training data.** Training digits have one fixed colour per class on a near-black background. Instead, the samples have a noisy mid-grey/greenish background (dark-green speckle at stage 2, grey speckle at stage 3) and multicoloured strokes, often pink, green and yellow within the same digit. The colour does not reliably match the class palette (e.g. label 5 should be blue and comes out pink/green). The chroma stages have not converged.
  - **Noise:** strong per-pixel colour speckle at stages 2 and 3, in the background and on the strokes. Stage 3 is grainy rather than blocky. Stage 1 is visibly blocky (16x16), as expected.
- Next steps: train longer (stage 1 first, since stage 1 sets the shape and the later stages copy it); add classifier-free guidance; check stage 2/3 on ground-truth low-res inputs to separate their errors from stage 1's; compare against gray-only stages 2/3.

## 2026-10-25 — EMA, periodic samples, sampler sanity check; long stage-1 run launched (no results yet)

- Code: `train.py --ema-decay` (EMA of the stage weights, decay warmup, saved as `"ema_state"`; `"state"` stays raw), `--warmup-steps`, `--lr-schedule {constant,cosine,linear}`, `--sample-every/--sample-dir/--sample-steps`, `--save-every`, `--resume`, JSON loss log `stageK_log.json`. `sample.py` / `make_grid.py`: EMA weights when the checkpoint has them, `--use-ema` / `--no-ema`. All new options default to off.
- Backward compatibility checks (CPU): with default options, `train.py --stage 1 --steps 30 --batch-size 16 --subset 512` gives a `stage1_losses.json` byte-identical to the previous code and identical weights. `sample.py` on the old 2000-step checkpoints gives a PNG byte-identical to the previous code (raw weights, no EMA in those checkpoints).
- Tests: `pytest -q tests`: 35 passed (27 existing + 6 in `tests/test_ema.py` + 2 in `tests/test_sampler.py`), with transformers 5.20.2 and the CLIP model cached. Without transformers: 26 passed, 9 skipped (CLIP).
- Sampling-pipeline sanity check before the long run, on the old 2000-step stage-1 checkpoint and 256 real gray MNIST images at 16x16. **No bug found.**
  - (a) Training targets are in [-1, 1] (min -1.0, max 1.0; background = -1). The sampler clamps x0 to [-1, 1], and `from_model_range` maps [-1, 1] to [0, 1]. The ranges match.
  - (b) The DDIM update uses the same `alphas_cumprod[t]` as `q_sample`. Timesteps go from 999 to 0 (50 unique values), the same integer range the UNet's time embedding sees in training. With an oracle ε, each DDIM step lands on `q_sample(x0, t_next, ε)` and the output equals x0 (now a test, `tests/test_sampler.py`).
  - (c) At sampling, label embeddings equal `encoder(labels)`. The null token (index 10) is only used by training-time condition dropout (p = 0.1). No guidance is used.
  - (d) ε-MSE of the old checkpoint on noised real images: t=50: 0.143, t=200: 0.069, t=500: 0.012, t=800: 0.0013. Random t: 0.032 (the training loss over the last 100 steps was 0.039). With a wrong label the ε-MSE is higher (t=200: 0.091 vs 0.069 correct, 0.077 null), so the label is used, but only weakly. Denoising from t=300 with 25 DDIM steps of the sampler's update gives MSE 0.073 to the clean image, vs 0.62 for the rescaled noisy input. From t=500: 0.16 vs 1.30. The wrong shapes of the earlier stage-1 samples are therefore not explained by a range, schedule, indexing or conditioning mismatch. Old results are unaffected (no code fix was needed).
- Run launched (stage 1 only, no results yet):
  ```
  OMP_NUM_THREADS=8 python train.py --stage 1 --steps 30000 --batch-size 128 --lr 2e-4 --ema-decay 0.999 \
      --warmup-steps 500 --sample-every 5000 --save-every 5000 --ckpt-dir checkpoints_long --sample-dir samples_long
  ```
  Gray MNIST, label conditioning, `default_config()` (16x16 Y, base_channels 32, about 688k parameters), constant LR after warmup, EMA 0.999, sample grids with EMA weights and 50 DDIM steps every 5000 steps. Hardware: CPU, 8 threads. Measured speed: about 0.083 s/step at batch 128, so roughly 42 min for 30000 steps. Stages 2 and 3 were not retrained.
- Results: none yet. They will be added in a later entry.

