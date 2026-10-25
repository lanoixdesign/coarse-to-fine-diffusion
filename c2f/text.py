"""Conditioning encoders shared by every stage of the cascade.

Interface: an `nn.Module` with attribute `dim` and
    forward(cond) -> (B, dim) tensor
where `cond` is a LongTensor of labels (LabelEncoder) or a list of strings
(text encoders). `null(batch)` returns the "unconditional" embedding used
for condition dropout / classifier-free guidance.

To plug in CLIP or T5: subclass `TextEncoder`, run the frozen pretrained
encoder in `forward`, pool to one vector per prompt (or project its pooled
output) to `dim`. See `PretrainedTextEncoder` below for the intended shape.
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
