"""Conditioning encoders (one instance per stage of the cascade) and per-stage prompt helpers.

Interface: an `nn.Module` with attribute `dim` and
    forward(cond) -> (B, dim) tensor
where `cond` is a LongTensor of labels (LabelEncoder) or a list of strings
(text encoders). `null(batch)` returns the "unconditional" embedding used
for condition dropout / classifier-free guidance.

Encoders (`CascadeConfig.cond_type`):
  * "label" -> LabelEncoder     MNIST class embedding.
  * "text"  -> HashTextEncoder  stub hashed bag-of-words (offline / CI fallback, not a language model).
  * "clip"  -> CLIPTextEncoder  frozen pretrained CLIP text tower (Hugging Face `transformers`)
                                + a small trainable per-stage adapter to `dim`.

CLIP design: ONE frozen `CLIPTextBackbone` (tokenizer + CLIPTextModelWithProjection,
eval mode, requires_grad=False) is loaded once and shared by every stage's
`CLIPTextEncoder`. The backbone is *not* a registered submodule, so it is not in
`Cascade.parameters()` (never optimised) or `state_dict()` (never saved in
checkpoints); each stage only owns and saves its adapter. The backbone embeds
every prompt string on its own (batch of one, so the result does not depend on
what else is in the batch) and caches the result per unique string.

Per-stage prompts: every stage encodes its *own* prompt with its own encoder
(`Cascade.encoders[k]`). `normalize_stage_prompts` / `broadcast_cond` below
implement the fallback rules (a stage without a prompt uses the base prompt).
"""
from __future__ import annotations

import hashlib
import os
from collections import OrderedDict
from typing import Dict, List, Optional, Sequence, Union

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


DEFAULT_CLIP_MODEL = "openai/clip-vit-base-patch32"
TEXT_COND_TYPES = ("text", "clip")  # cond types that take string prompts
TRANSFORMERS_MIN_VERSION = "4.46"  # tested: 4.46.3 and 5.20.2


class CLIPUnavailableError(RuntimeError):
    """CLIP could not be loaded (transformers missing, or the model could not be downloaded/found)."""


def _clip_help(model_id: str) -> str:
    return (
        f"  * install the dependency:  pip install 'transformers>={TRANSFORMERS_MIN_VERSION}'  "
        f"(or: pip install -r requirements.txt)\n"
        f"  * the first run downloads '{model_id}' from the Hugging Face Hub (needs network access);\n"
        f"    offline, pre-download it once (`hf download {model_id}`, or `huggingface-cli download` with older\n"
        f"    huggingface_hub, or just one online run) and then set HF_HUB_OFFLINE=1,\n"
        f"    or point --clip-model at a local directory containing the model and tokenizer files;\n"
        f"  * or use the offline stub text encoder instead:  --cond text")


class CLIPTextBackbone:
    """Frozen CLIP text tower shared by all stages (deliberately *not* an nn.Module).

    Embedding: the projected, pooled text embedding `text_embeds` of
    `CLIPTextModelWithProjection` (EOS-token hidden state after the final layer norm,
    times CLIP's text projection; 512-d for clip-vit-base-patch32), L2-normalised.
    This is the vector CLIP aligns with images.

    `embed(prompts)` -> (B, out_dim) float32 tensor on CPU (or the backbone's device). Each
    unique prompt is tokenised (truncated to 77 tokens) and run through CLIP *on its own*,
    and the result is cached by string, so repeated prompts (templated training prompts) cost
    a dict lookup. The cache is bounded (`max_cache` entries, least-recently-used dropped).
    The model is in eval mode under `torch.no_grad()`, so the output is deterministic.
    """

    def __init__(self, model_id: str = DEFAULT_CLIP_MODEL, device="cpu", max_cache: int = 100_000):
        self.model_id = model_id
        try:
            import transformers
            from transformers import CLIPTextModelWithProjection, CLIPTokenizer
        except ImportError as e:  # transformers missing (or broken install)
            raise CLIPUnavailableError(
                f"--cond clip needs the Hugging Face `transformers` package, which could not be imported "
                f"({e}).\n" + _clip_help(model_id)) from e
        try:
            verbosity = transformers.logging.get_verbosity()
            transformers.logging.set_verbosity_error()  # silence "unused vision weights" notices
            transformers.logging.disable_progress_bar()
            try:
                self.tokenizer = CLIPTokenizer.from_pretrained(model_id)
                self.model = CLIPTextModelWithProjection.from_pretrained(model_id)
            finally:
                transformers.logging.set_verbosity(verbosity)
                transformers.logging.enable_progress_bar()
        except Exception as e:  # OSError (not found / offline cache miss), HTTP/connection errors, bad config
            lines = str(e).strip().splitlines()
            detail = f"{type(e).__name__}: {lines[0] if lines else ''}"
            if model_id.startswith((".", "/", "~")) and not os.path.isdir(os.path.expanduser(model_id)):
                detail = f"local directory {model_id!r} does not exist"
            elif os.environ.get("HF_HUB_OFFLINE") or os.environ.get("TRANSFORMERS_OFFLINE"):
                detail += " [HF_HUB_OFFLINE is set and the model is not in the local Hugging Face cache]"
            raise CLIPUnavailableError(
                f"could not load the CLIP text model '{model_id}' ({detail})"
                + ".\n" + _clip_help(model_id)) from e
        self.model.eval().to(device)
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.out_dim = int(self.model.config.projection_dim)
        self.max_length = int(min(self.tokenizer.model_max_length, self.model.config.max_position_embeddings))
        self.max_cache = max_cache
        self._cache: "OrderedDict[str, torch.Tensor]" = OrderedDict()
        self.n_forward = 0  # number of CLIP forward passes (cache misses), for tests / logging

    @property
    def device(self):
        return next(self.model.parameters()).device

    def to(self, device) -> "CLIPTextBackbone":
        self.model.to(device)
        self._cache.clear()
        return self

    def parameters(self):
        return self.model.parameters()

    @torch.no_grad()
    def _encode_one(self, prompt: str) -> torch.Tensor:
        tok = self.tokenizer([prompt], padding=True, truncation=True, max_length=self.max_length,
                             return_tensors="pt")
        tok = {k: v.to(self.device) for k, v in tok.items()}
        self.n_forward += 1
        emb = self.model(**tok).text_embeds[0].float()
        return torch.nn.functional.normalize(emb, dim=-1).clone()

    def embed(self, prompts: Sequence[str]) -> torch.Tensor:
        out = []
        for p in prompts:
            if not isinstance(p, str):
                raise TypeError(f"CLIP prompts must be strings, got {type(p).__name__}")
            e = self._cache.get(p)
            if e is None:
                e = self._encode_one(p)
                self._cache[p] = e
                if len(self._cache) > self.max_cache:
                    self._cache.popitem(last=False)
            else:
                self._cache.move_to_end(p)
            out.append(e)
        return torch.stack(out)

    def cache_info(self) -> dict:
        return {"entries": len(self._cache), "forward_passes": self.n_forward}


_BACKBONES: Dict[tuple, CLIPTextBackbone] = {}


def load_clip_backbone(model_id: Optional[str] = None, device="cpu", reuse: bool = True) -> CLIPTextBackbone:
    """Load (or reuse, per process) the frozen CLIP text backbone. Raises CLIPUnavailableError."""
    model_id = model_id or DEFAULT_CLIP_MODEL
    key = (model_id, str(device))
    if reuse and key in _BACKBONES:
        return _BACKBONES[key]
    bb = CLIPTextBackbone(model_id, device=device)
    if reuse:
        _BACKBONES[key] = bb
    return bb


class CLIPTextEncoder(TextEncoder):
    """One stage's CLIP conditioning: shared frozen backbone + this stage's trainable adapter.

    forward(prompts) = adapter(backbone.embed(prompts)), adapter = LayerNorm(512) -> Linear(512, dim).
    null(batch)      = adapter(backbone.embed([""])): the embedding of the empty prompt "",
                       a fixed CLIP vector (deterministic), mapped by this stage's adapter.
    Only the adapter is a parameter / in the state dict; the backbone is never saved or trained.
    """

    NULL_PROMPT = ""

    def __init__(self, backbone: CLIPTextBackbone, dim: int = 128):
        super().__init__()
        self.dim = dim
        object.__setattr__(self, "backbone", backbone)  # plain attribute: not a submodule
        self.adapter = nn.Sequential(nn.LayerNorm(backbone.out_dim), nn.Linear(backbone.out_dim, dim))

    def _adapt(self, feats: torch.Tensor) -> torch.Tensor:
        w = self.adapter[1].weight
        return self.adapter(feats.to(device=w.device, dtype=w.dtype))

    def forward(self, prompts: Union[List[str], torch.Tensor]) -> torch.Tensor:
        if torch.is_tensor(prompts):
            raise TypeError("CLIPTextEncoder takes a list of prompt strings, not a tensor")
        return self._adapt(self.backbone.embed(list(prompts)))

    def null(self, batch: int, device=None) -> torch.Tensor:
        e = self._adapt(self.backbone.embed([self.NULL_PROMPT] * batch))  # == forward([""] * batch)
        return e if device is None else e.to(device)


def encoder_info(cfg) -> dict:
    """What a checkpoint records about its conditioning encoder."""
    info = {"type": cfg.cond_type}
    if cfg.cond_type == "clip":
        info.update(model_id=cfg.clip_model or DEFAULT_CLIP_MODEL,
                    embedding="CLIPTextModelWithProjection.text_embeds (L2-normalised)",
                    null_prompt=CLIPTextEncoder.NULL_PROMPT, backbone_saved=False)
    return info


def build_encoder(cond_type: str, dim: int, num_classes: int = 10,
                  backbone: Optional[CLIPTextBackbone] = None, clip_model: Optional[str] = None) -> TextEncoder:
    if cond_type == "label":
        return LabelEncoder(num_classes, dim)
    if cond_type == "text":
        return HashTextEncoder(dim)
    if cond_type == "clip":
        return CLIPTextEncoder(backbone if backbone is not None else load_clip_backbone(clip_model), dim)
    raise ValueError(f"unknown cond_type {cond_type!r} (expected 'label', 'text' or 'clip')")


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

    Returns a list of `batch` strings ("text" / "clip") or a LongTensor of `batch` labels ("label").
    """
    if cond_type in TEXT_COND_TYPES:
        if not _is_text(p):
            raise TypeError(f"text conditioning expects a str or a list of str, got {type(p).__name__}")
        prompts = [p] * batch if isinstance(p, str) else list(p)
        if len(prompts) != batch:
            raise ValueError(f"got {len(prompts)} prompts for a batch of {batch}")
        return prompts
    if cond_type == "label":
        if _is_text(p):
            raise TypeError("label conditioning cannot take string prompts; use cond_type='clip' or 'text' "
                            "(train.py/sample.py: --cond clip, or the offline stub --cond text) for prompts")
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
