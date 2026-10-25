"""Conditioning encoders (one instance per stage of the cascade) and per-stage prompt helpers.

Interface: an `nn.Module` with attribute `dim` and
    forward(cond) -> (B, dim) tensor
where `cond` is a LongTensor of labels (LabelEncoder) or a list of strings
(text encoders). `null(batch)` returns the "unconditional" embedding used
for condition dropout / classifier-free guidance.

To plug in CLIP or T5: subclass `TextEncoder`, run the frozen pretrained
encoder in `forward`, pool to one vector per prompt (or project its pooled
output) to `dim`. See `PretrainedTextEncoder` below for the intended shape.

Per-stage prompts: every stage encodes its *own* prompt with its own encoder
(`Cascade.encoders[k]`). `normalize_stage_prompts` / `broadcast_cond` below
implement the fallback rules (a stage without a prompt uses the base prompt).
"""
from __future__ import annotations

import hashlib
from typing import List, Sequence, Union

import torch
import torch.nn as nn

DIGIT_NAMES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]


def labels_to_prompts(labels: Sequence[int]) -> List[str]:
    return [f"a handwritten digit {DIGIT_NAMES[int(l)]}" for l in labels]


class TextEncoder(nn.Module):
    dim: int

    def null(self, batch: int, device=None) -> torch.Tensor:
        raise NotImplementedError


class LabelEncoder(TextEncoder):
    """Class-label embedding; index `num_classes` is the null/unconditional token."""

    def __init__(self, num_classes: int = 10, dim: int = 128):
        super().__init__()
        self.dim, self.num_classes = dim, num_classes
        self.emb = nn.Embedding(num_classes + 1, dim)

    def forward(self, labels: torch.Tensor) -> torch.Tensor:
        return self.emb(labels)

    def null(self, batch: int, device=None) -> torch.Tensor:
        idx = torch.full((batch,), self.num_classes, dtype=torch.long, device=device or self.emb.weight.device)
        return self.emb(idx)


class HashTextEncoder(TextEncoder):
    """Stub text encoder: hashed bag-of-words -> trainable EmbeddingBag.

    Not a language model. It exists so the cascade is exercised with
    string prompts end-to-end; replace with CLIP/T5 for real text.
    """

    def __init__(self, dim: int = 128, buckets: int = 4096):
        super().__init__()
        self.dim, self.buckets = dim, buckets
        self.bag = nn.EmbeddingBag(buckets + 1, dim, mode="mean")  # last bucket = null

    def _hash(self, word: str) -> int:
        return int(hashlib.md5(word.encode()).hexdigest(), 16) % self.buckets

    def forward(self, prompts: Union[List[str], torch.Tensor]) -> torch.Tensor:
        device = self.bag.weight.device
        ids, offsets = [], []
        for p in prompts:
            offsets.append(len(ids))
            toks = p.lower().split() or ["<empty>"]
            ids.extend(self._hash(w) for w in toks)
        return self.bag(torch.tensor(ids, device=device), torch.tensor(offsets, device=device))

    def null(self, batch: int, device=None) -> torch.Tensor:
        device = device or self.bag.weight.device
        ids = torch.full((batch,), self.buckets, dtype=torch.long, device=device)
        return self.bag(ids, torch.arange(batch, device=device))


class PretrainedTextEncoder(TextEncoder):
    """Placeholder for a frozen CLIP / T5 encoder (not implemented).

    Expected implementation: load tokenizer + encoder (e.g. via
    `transformers`), freeze it, pool token states to one vector, and apply a
    trainable `nn.Linear(hidden, dim)`. Keep a learned null vector for CFG.
    """

    def __init__(self, name: str, dim: int = 128):
        super().__init__()
        raise NotImplementedError(f"Pretrained text encoder '{name}' is not wired up yet (see roadmap).")


def build_encoder(cond_type: str, dim: int, num_classes: int = 10) -> TextEncoder:
    if cond_type == "label":
        return LabelEncoder(num_classes, dim)
    if cond_type == "text":
        return HashTextEncoder(dim)
    raise ValueError(f"unknown cond_type {cond_type!r}")


# ---------------------------------------------------------------------------
# Per-stage prompts
# ---------------------------------------------------------------------------
# A "stage condition" is what one stage's encoder consumes: a LongTensor of
# labels (LabelEncoder) or a list of B strings (text encoders).
StageCond = Union[str, Sequence[str], Sequence[int], int, torch.Tensor, None]


def normalize_stage_prompts(stage_prompts, n_stages: int) -> List[StageCond]:
    """Turn a per-stage prompt spec into a list of length `n_stages` (None = use base).

    Accepted forms:
      * None                          -> no overrides
      * dict {stage: prompt}          -> stage keys are 1-based ints (1..n) or "s1"/"stage1"
      * list/tuple [p1, p2, p3]       -> positional, index 0 = stage 1; None entries and
                                         a shorter list mean "fall back to the base prompt"
    Each prompt is a str (broadcast to the batch), a list of B strings, or (label
    conditioning only) an int / LongTensor of B labels.
    """
    out: List[StageCond] = [None] * n_stages
    if stage_prompts is None:
        return out
    if isinstance(stage_prompts, str):
        raise TypeError("stage_prompts must be a dict or a per-stage list; "
                        "pass a single prompt for all stages as `prompt=` instead")
    if isinstance(stage_prompts, dict):
        for key, p in stage_prompts.items():
            k = key
            if isinstance(k, str):
                s = k.lower()
                for pre in ("stage", "s"):
                    if s.startswith(pre) and s[len(pre):].isdigit():
                        s = s[len(pre):]
                        break
                if not s.isdigit():
                    raise KeyError(f"bad stage key {key!r}; use 1..{n_stages} or 's1'..'s{n_stages}'")
                k = int(s)
            if not (isinstance(k, int) and 1 <= k <= n_stages):
                raise KeyError(f"stage key {key!r} out of range 1..{n_stages} (stages are 1-based)")
            out[k - 1] = p
        return out
    stage_prompts = list(stage_prompts)
    if len(stage_prompts) > n_stages:
        raise ValueError(f"got {len(stage_prompts)} stage prompts for {n_stages} stages")
    for i, p in enumerate(stage_prompts):
        out[i] = p
    return out


def _is_text(p) -> bool:
    return isinstance(p, str) or (isinstance(p, (list, tuple)) and len(p) > 0 and all(isinstance(x, str) for x in p))


def broadcast_cond(p: StageCond, batch: int, cond_type: str, device=None):
    """Validate one stage condition and expand it to the batch.

    Returns a list of `batch` strings ("text") or a LongTensor of `batch` labels ("label").
    """
    if cond_type == "text":
        if not _is_text(p):
            raise TypeError(f"text conditioning expects a str or a list of str, got {type(p).__name__}")
        prompts = [p] * batch if isinstance(p, str) else list(p)
        if len(prompts) != batch:
            raise ValueError(f"got {len(prompts)} prompts for a batch of {batch}")
        return prompts
    if cond_type == "label":
        if _is_text(p):
            raise TypeError("label conditioning cannot take string prompts; use cond_type='text' "
                            "(train.py/sample.py: --cond text) for prompts")
        t = torch.as_tensor(p, dtype=torch.long, device=device)
        if t.ndim == 0:
            t = t.expand(batch)
        if t.shape != (batch,):
            raise ValueError(f"got labels of shape {tuple(t.shape)} for a batch of {batch}")
        return t
    raise ValueError(f"unknown cond_type {cond_type!r}")


def format_prompt(template: str, label: int, color_name: str = "") -> str:
    """Fill `{digit}`, `{label}` and `{color}` placeholders (used by train.py / sample.py)."""
    return template.format(digit=DIGIT_NAMES[int(label)], label=int(label), color=color_name)
