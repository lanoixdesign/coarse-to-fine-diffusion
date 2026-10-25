"""Classifier-free guidance sweep on the stage-1 (16x16 Y) model, no retraining.

For every (guidance scale w, DDIM steps) setting, with the SAME noise seeds:
  * a 10x2 grid (labels 0..9 twice, seed 1234 = the training-time sample grid),
  * a 10x5 grid (columns = digits 0..9, 5 rows, seed 1235),
  * N samples per digit (seed 2026) scored by a separately trained 16x16 MNIST
    classifier (tools/mnist16_classifier.py): accuracy = share recognised as the
    conditioning label (a PROXY for legibility, not a quality benchmark), mean
    confidence, intra-class diversity (mean pairwise L2 in pixel space, vs real
    test digits) and a Frechet distance on the classifier's 256-d penultimate
    features against real test digits (small-sample proxy, lower = closer).

  python tools/cfg_eval.py --ckpt checkpoints_long/stage1.pt --scales 1 2 3 4 6 --steps 50 100
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from torchvision.utils import make_grid, save_image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from c2f import Cascade, CascadeConfig, stage_state_from_checkpoint  # noqa: E402
from mnist16_classifier import Net, mnist16  # noqa: E402


@torch.no_grad()
def sample_stage1(model, labels, w, steps, seed, bs=500, rederive=False):
    out = []
    g_state = torch.random.get_rng_state()
    torch.manual_seed(seed)
    for i in range(0, len(labels), bs):
        lab = labels[i:i + bs]
        emb = model.encode_stage(0, lab)
        null = model.null_embedding(0, len(lab)) if w != 1.0 else None
        out.append(model.diffusion.sample(model.stages[0], (len(lab), 1, 16, 16), None, emb, steps=steps,
                                          guidance_scale=w, null_emb=null, rederive_eps=rederive))
    torch.random.set_rng_state(g_state)
    return torch.cat(out)


def feats(net, x):
    net.eval()
    with torch.no_grad():
        h = net.f[:-1](x)  # up to the last Linear (256-d after ReLU/Dropout(eval))
    return h.double().numpy()


def frechet(a, b):
    mu1, mu2 = a.mean(0), b.mean(0)
    s1, s2 = np.cov(a, rowvar=False), np.cov(b, rowvar=False)
    # Tr sqrt(s1 s2) = sum sqrt(eig(sqrt(s1) s2 sqrt(s1))), symmetric PSD -> eigh (no scipy needed)
    e, v = np.linalg.eigh(s1)
    r1 = (v * np.sqrt(np.clip(e, 0, None))) @ v.T
    tr_covmean = np.sqrt(np.clip(np.linalg.eigvalsh(r1 @ s2 @ r1), 0, None)).sum()
    return float(((mu1 - mu2) ** 2).sum() + np.trace(s1) + np.trace(s2) - 2 * tr_covmean)


def intra_div(x, y):
    d = []
    for c in range(10):
        xc = x[y == c].flatten(1)
        dist = torch.cdist(xc, xc)
        n = len(xc)
        d.append(dist.sum().item() / (n * (n - 1)))
    return float(np.mean(d))


def tile(x, nrow, size=48):
    return make_grid(F.interpolate((x + 1) / 2, size=(size, size), mode="nearest").clamp(0, 1),
                     nrow=nrow, padding=2, pad_value=0.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints_long/stage1.pt")
    ap.add_argument("--clf", default="checkpoints_long/mnist16_classifier.pt")
    ap.add_argument("--scales", type=float, nargs="+", default=[1, 2, 3, 4, 6])
    ap.add_argument("--steps", type=int, nargs="+", default=[50])
    ap.add_argument("--per-class", type=int, default=200)
    ap.add_argument("--out-dir", default="samples_long/cfg_sweep")
    ap.add_argument("--data", default="data")
    ap.add_argument("--json-name", default="cfg_sweep.json")
    ap.add_argument("--rederive", type=int, nargs="+", default=[0], help="0/1: re-derive eps after the x0 clamp")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    ck = torch.load(a.ckpt, map_location="cpu")
    cfg = CascadeConfig.from_dict(ck["config"])
    print(f"cond_type={cfg.cond_type} cond_drop_prob={cfg.cond_drop_prob} (train step {ck.get('step')})")
    model = Cascade(cfg)
    state, used_ema = stage_state_from_checkpoint(ck, None)
    model.load_stage_state(0, state)
    model.eval()
    print("weights:", "EMA" if used_ema else "raw")

    net = Net()
    c = torch.load(a.clf, map_location="cpu")
    net.load_state_dict(c["state"])
    net.eval()
    xte, yte = mnist16(a.data, False)
    with torch.no_grad():
        real_acc = (net(xte).argmax(1) == yte).float().mean().item()
    # real reference set with the same class balance as the generated set
    idx = torch.cat([(yte == d).nonzero()[:a.per_class, 0] for d in range(10)])
    xr, yr = xte[idx], yte[idx]
    fr = feats(net, xr)
    real_div = intra_div(xr, yr)
    print(f"classifier on real test: acc {real_acc:.4f}; real intra-class diversity {real_div:.3f}")

    labels = torch.arange(10).repeat_interleave(a.per_class)
    results = {"real": {"acc": real_acc, "intra_div": real_div, "n_ref": len(xr)}, "settings": []}
    big = {}
    for rd in a.rederive:
     for steps in a.steps:
        for w in a.scales:
            t0 = time.time()
            x = sample_stage1(model, labels, w, steps, seed=2026, rederive=bool(rd))
            dt = time.time() - t0
            with torch.no_grad():
                p = torch.cat([F.softmax(net(x[i:i + 1000]), 1) for i in range(0, len(x), 1000)])
            pred = p.argmax(1)
            acc = (pred == labels).float().mean().item()
            per = [(pred[labels == d] == d).float().mean().item() for d in range(10)]
            conf = p.gather(1, labels[:, None]).mean().item()
            fd = frechet(feats(net, x), fr)
            div = intra_div(x, labels)
            r = {"w": w, "ddim_steps": steps, "rederive_eps": bool(rd), "n": len(x), "acc": acc, "per_digit_acc": per,
                 "mean_p_label": conf, "intra_div": div, "frechet_clf_feats": fd, "sample_s": round(dt, 1)}
            results["settings"].append(r)
            print(f"w={w:g} steps={steps} rederive={rd}: acc {acc:.4f}  p(label) {conf:.3f}  div {div:.3f}  "
                  f"FD {fd:.2f}  ({dt:.1f}s)  per-digit " + " ".join(f"{v:.2f}" for v in per), flush=True)
            tag = f"w{w:g}_ddim{steps}" + ("_rederive" if rd else "")
            g2 = sample_stage1(model, torch.arange(10).repeat(2), w, steps, seed=1234, bs=20, rederive=bool(rd))
            save_image(tile(g2, 10, 64), os.path.join(a.out_dir, f"grid10x2_{tag}.png"))
            g5 = sample_stage1(model, torch.arange(10).repeat(5), w, steps, seed=1235, bs=50, rederive=bool(rd))
            save_image(tile(g5, 10, 48), os.path.join(a.out_dir, f"grid10x5_{tag}.png"))
            big[(w, steps, rd)] = g5
    with open(os.path.join(a.out_dir, a.json_name), "w") as f:
        json.dump(results, f, indent=1)

    # side-by-side montage of the 10x5 grids, one panel per setting
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18) \
        if os.path.exists("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf") else ImageFont.load_default()
    for rd, steps in [(rd, st) for rd in a.rederive for st in a.steps]:
        panels = []
        for w in a.scales:
            g = tile(big[(w, steps, rd)], 10, 32)
            im = Image.fromarray((g.permute(1, 2, 0).numpy() * 255).round().astype("uint8"))
            r = next(r for r in results["settings"] if r["w"] == w and r["ddim_steps"] == steps
                     and r["rederive_eps"] == bool(rd))
            canvas = Image.new("RGB", (im.width, im.height + 26), "white")
            canvas.paste(im, (0, 26))
            ImageDraw.Draw(canvas).text((4, 3), f"w={w:g}  acc={r['acc']:.3f}", fill="black", font=font)
            panels.append(canvas)
        W = sum(p.width for p in panels) + 8 * (len(panels) - 1)
        m = Image.new("RGB", (W, panels[0].height), "white")
        x0 = 0
        for p in panels:
            m.paste(p, (x0, 0)); x0 += p.width + 8
        m.save(os.path.join(a.out_dir, f"montage_10x5_ddim{steps}" + ("_rederive" if rd else "") + ".png"))
    print("saved", a.out_dir)


if __name__ == "__main__":
    main()
