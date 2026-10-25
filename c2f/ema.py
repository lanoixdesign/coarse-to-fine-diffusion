"""Exponential moving average (EMA) of one cascade stage's weights.

The EMA tracks the stage's UNet *and* its conditioning encoder (the same two
modules `Cascade.stage_state(k)` saves), so `EMA.state_dict()` has exactly the
layout of `stage_state(k)` and can be passed to `Cascade.load_stage_state`.

Decay warmup (default on): the effective decay at update n is
    min(decay, (1 + n) / (10 + n)),
so early in training the average follows the raw weights closely instead of
being dominated by the random initialisation.

Checkpoints: train.py stores the raw weights under "state" (unchanged) and the
EMA weights under "ema_state" (only when --ema-decay > 0). See
`stage_state_from_checkpoint` for how sample.py / make_grid.py choose.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Dict, Optional

import torch


def _flat(state: dict) -> Dict[str, torch.Tensor]:
    return {f"{part}.{k}": v for part, sd in state.items() for k, v in sd.items()}


class StageEMA:
    def __init__(self, model, k: int, decay: float = 0.999, warmup: bool = True):
        if not 0.0 < decay < 1.0:
            raise ValueError(f"EMA decay must be in (0, 1), got {decay}")
        self.model, self.k, self.decay, self.warmup = model, k, float(decay), warmup
        self.num_updates = 0
        self.shadow = {part: {n: t.detach().clone() for n, t in sd.items()}
                       for part, sd in model.stage_state(k).items()}

    def _live(self):
        return {"unet": self.model.stages[self.k], "encoder": self.model.encoders[self.k]}

    def current_decay(self) -> float:
        if not self.warmup:
            return self.decay
        return min(self.decay, (1.0 + self.num_updates) / (10.0 + self.num_updates))

    @torch.no_grad()
    def update(self) -> float:
        d = self.current_decay()
        for part, mod in self._live().items():
            live = mod.state_dict()
            sh = self.shadow[part]
            fl = [n for n, t in live.items() if t.is_floating_point()]
            if fl:
                torch._foreach_lerp_([sh[n] for n in fl], [live[n].detach() for n in fl], 1.0 - d)
            for n, t in live.items():  # integer buffers (if any): copy
                if not t.is_floating_point():
                    sh[n].copy_(t)
        self.num_updates += 1
        return d

    def state_dict(self) -> dict:
        """Same layout as `Cascade.stage_state(k)`: {"unet": ..., "encoder": ...}."""
        return {part: {n: t.clone() for n, t in sd.items()} for part, sd in self.shadow.items()}

    def meta(self) -> dict:
        return {"decay": self.decay, "warmup": self.warmup, "num_updates": self.num_updates}

    def load(self, state: dict, meta: Optional[dict] = None) -> None:
        for part, sd in state.items():
            for n, t in sd.items():
                self.shadow[part][n].copy_(t)
        if meta:
            self.num_updates = int(meta.get("num_updates", 0))

    @contextmanager
    def swapped_in(self):
        """Temporarily load the EMA weights into the live model (restored on exit)."""
        raw = self.model.stage_state(self.k)
        raw = {part: {n: t.detach().clone() for n, t in sd.items()} for part, sd in raw.items()}
        self.model.load_stage_state(self.k, self.shadow)
        try:
            yield
        finally:
            self.model.load_stage_state(self.k, raw)


def stage_state_from_checkpoint(ckpt: dict, use_ema: Optional[bool] = None):
    """Pick the weights to sample with from a train.py checkpoint.

    use_ema=None (default, "auto"): EMA weights if the checkpoint has "ema_state",
    else the raw "state" (old checkpoints behave exactly as before).
    use_ema=True: EMA weights, error if absent. use_ema=False: raw weights.
    Returns (state, used_ema: bool).
    """
    has = ckpt.get("ema_state") is not None
    if use_ema is None:
        use_ema = has
    if use_ema:
        if not has:
            raise KeyError("checkpoint has no 'ema_state' (trained without --ema-decay); "
                           "sample without --use-ema")
        return ckpt["ema_state"], True
    return ckpt["state"], False
