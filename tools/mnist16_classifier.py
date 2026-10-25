"""Small MNIST classifier at 16x16 (stage-1 resolution), used as a *proxy* metric:
the share of generated stage-1 samples it recognises as the conditioning label.

  python tools/mnist16_classifier.py --out checkpoints_long/mnist16_classifier.pt

Preprocessing is exactly the stage-1 training target: MNIST [0,1] -> bilinear 64x64
(as C2FMNIST with size=64, gray) -> area 16x16 -> [-1, 1].
"""
import argparse
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def mnist16(root="data", train=True):
    from torchvision import datasets
    ds = datasets.MNIST(root, train=train, download=False)
    x = ds.data.float().div(255)[:, None]
    x = F.interpolate(x, size=(64, 64), mode="bilinear", align_corners=False)
    x = F.interpolate(x, size=(16, 16), mode="area")
    return x * 2 - 1, ds.targets.clone()


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.f = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),   # 8x8
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),  # 4x4
            nn.Flatten(), nn.Dropout(0.3), nn.Linear(128 * 16, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 10))

    def forward(self, x):
        return self.f(x)


def shift(x, m=1):
    """Random integer translation by up to m pixels (background = -1)."""
    p = F.pad(x, (m, m, m, m), value=-1.0)
    dx, dy = torch.randint(0, 2 * m + 1, (2,)).tolist()
    return p[:, :, dy:dy + x.shape[2], dx:dx + x.shape[3]]


@torch.no_grad()
def predict(net, x, bs=1000):
    net.eval()
    return torch.cat([F.softmax(net(x[i:i + bs]), 1) for i in range(0, len(x), bs)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--out", default="checkpoints_long/mnist16_classifier.pt")
    a = ap.parse_args()
    torch.manual_seed(0)
    xtr, ytr = mnist16(a.data, True)
    xte, yte = mnist16(a.data, False)
    net = Net()
    opt = torch.optim.AdamW(net.parameters(), 2e-3, weight_decay=1e-4)
    bs, n = 256, len(xtr)
    total = a.epochs * (n // bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=total)
    t0 = time.time()
    for ep in range(a.epochs):
        net.train()
        perm = torch.randperm(n)
        for i in range(n // bs):
            idx = perm[i * bs:(i + 1) * bs]
            loss = F.cross_entropy(net(shift(xtr[idx])), ytr[idx])
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        acc = (predict(net, xte).argmax(1) == yte).float().mean().item()
        print(f"epoch {ep + 1}: test acc {acc:.4f} ({time.time() - t0:.0f}s)", flush=True)
    torch.save({"state": net.state_dict(), "test_acc": acc, "epochs": a.epochs}, a.out)
    print("saved", a.out)


if __name__ == "__main__":
    main()
