"""CLIP text conditioning (`cond_type="clip"`).

Tests that need the pretrained model skip cleanly (with the reason) when `transformers`
is not installed or the model cannot be downloaded / is not in the local Hugging Face
cache (CI runs with HF_HUB_OFFLINE=1, so they skip there unless the model is cached).
The import-error / load-error message tests always run.
"""
import builtins
import os

import pytest
import torch
import torch.nn.functional as F

from c2f import (CLIPTextEncoder, CLIPUnavailableError, Cascade, DEFAULT_CLIP_MODEL, default_config,
                 labels_to_prompts, load_clip_backbone)
from c2f.text import CLIPTextBackbone, encoder_info

BASE = "a handwritten digit seven"
DETAIL = "a bold red handwritten seven with thick smooth strokes on a pale blue background"
SHAPES = [(2, 1, 16, 16), (2, 3, 32, 32), (2, 3, 64, 64)]


@pytest.fixture(scope="module")
def backbone():
    try:
        return load_clip_backbone(DEFAULT_CLIP_MODEL)
    except CLIPUnavailableError as e:
        pytest.skip(f"CLIP model {DEFAULT_CLIP_MODEL!r} unavailable (transformers missing or model "
                    f"not downloadable/cached): {str(e).splitlines()[0]}")


def clip_model(backbone, seed=0):
    torch.manual_seed(seed)
    cfg = default_config()
    cfg.cond_type = "clip"
    cfg.timesteps = 50
    for s in cfg.stages:
        s.base_channels = 16
    return Cascade(cfg, text_backbone=backbone).eval()


def sample(model, seed=0, **kw):
    torch.manual_seed(seed)
    return model.sample(steps=2, **kw)


# ---- (a) determinism ---------------------------------------------------------
def test_clip_embeddings_deterministic(backbone):
    a = backbone.embed([BASE, DETAIL])
    b = backbone.embed([BASE, DETAIL])
    assert torch.equal(a, b)
    backbone._cache.clear()  # recompute through CLIP, not the cache
    assert torch.equal(a, backbone.embed([BASE, DETAIL]))
    # the result for a prompt does not depend on the other prompts in the batch
    assert torch.equal(backbone.embed([DETAIL, "x", BASE])[2], a[0])
    # a second, independently loaded backbone gives identical embeddings
    fresh = load_clip_backbone(DEFAULT_CLIP_MODEL, reuse=False)
    assert fresh is not backbone
    assert torch.equal(fresh.embed([BASE, DETAIL]), a)
    # two fresh encoders with the same adapter weights give identical conditioning
    torch.manual_seed(0)
    e1 = CLIPTextEncoder(backbone, 32)
    torch.manual_seed(0)
    e2 = CLIPTextEncoder(fresh, 32)
    assert torch.equal(e1([BASE, DETAIL]), e2([BASE, DETAIL]))
    assert torch.equal(e1.null(3), e2.null(3))


def test_null_is_empty_prompt(backbone):
    enc = CLIPTextEncoder(backbone, 32)
    null = enc.null(4)
    assert null.shape == (4, 32)
    assert torch.equal(null, enc([""] * 4))
    assert torch.equal(enc.null(2), enc.null(2))


def test_cache_avoids_recomputing(backbone):
    backbone._cache.clear()
    n0 = backbone.n_forward
    prompts = labels_to_prompts([1, 2, 1, 2, 1, 2, 2, 1])
    backbone.embed(prompts)
    backbone.embed(prompts)
    assert backbone.n_forward - n0 == 2  # two unique strings -> two CLIP passes in total


# ---- (b) non-random, semantically sensible -------------------------------------
def test_clip_embeddings_are_semantic(backbone):
    car, auto, soup = backbone.embed(["a photo of a red car", "a picture of a red automobile", "a bowl of soup"])
    assert not torch.allclose(car, auto) and not torch.allclose(car, soup)
    cos = lambda u, v: F.cosine_similarity(u, v, dim=0).item()
    # observed with openai/clip-vit-base-patch32: car~auto 0.97, car~soup 0.68, auto~soup 0.71
    assert cos(car, auto) > cos(car, soup) + 0.1
    assert cos(car, auto) > cos(auto, soup) + 0.1
    seven, three = backbone.embed(labels_to_prompts([7, 3]))
    assert not torch.allclose(seven, three)
    assert torch.allclose(car.norm(), torch.tensor(1.0), atol=1e-5)  # L2-normalised


# ---- (c) per-stage prompts -----------------------------------------------------
def test_per_stage_prompts_encoded_separately(backbone):
    model = clip_model(backbone)
    sp = {2: DETAIL, 3: DETAIL}
    e_base = [model.encode_stage(k, prompt=BASE, batch=2) for k in range(3)]
    e_sp = [model.encode_stage(k, prompt=BASE, stage_prompts=sp, batch=2) for k in range(3)]
    assert all(e.shape == (2, model.cfg.cond_dim) for e in e_sp)
    assert torch.equal(e_base[0], e_sp[0])  # stage 1 keeps the base prompt
    assert not torch.allclose(e_base[1], e_sp[1])
    assert not torch.allclose(e_base[2], e_sp[2])
    # each stage's prompt string goes through CLIP on its own
    seen = []
    orig = backbone._encode_one
    backbone._encode_one = lambda p: (seen.append(p), orig(p))[1]
    try:
        backbone._cache.clear()
        for k in range(3):
            model.encode_stage(k, prompt=BASE, stage_prompts=sp, batch=2)
    finally:
        del backbone._encode_one
    assert seen == [BASE, DETAIL]  # one CLIP pass per unique string, stages 2/3 share the cached DETAIL
    # raw CLIP features of the stage prompts differ
    assert not torch.allclose(*backbone.embed([BASE, DETAIL]))


def test_clip_fallback_rules_unchanged(backbone):
    model = clip_model(backbone)
    a = sample(model, prompt=BASE, num_samples=2)
    assert [o.shape for o in a] == SHAPES
    b = sample(model, prompt=BASE, stage_prompts={1: BASE, 2: BASE, 3: BASE}, num_samples=2)
    c = sample(model, stage_prompts=[BASE, BASE, BASE], num_samples=2)
    for xs in (b, c):
        assert all(torch.equal(u, v) for u, v in zip(a, xs))
    # partial override falls back to the base prompt
    for k in (0, 2):
        assert torch.equal(model.encode_stage(k, prompt=BASE, stage_prompts={2: DETAIL}, batch=2),
                           model.encode_stage(k, prompt=BASE, batch=2))
    # labels -> "a handwritten digit {name}" as the default base prompt
    y = torch.tensor([7, 7])
    d = sample(model, labels=y)
    assert all(torch.equal(u, v) for u, v in zip(a, d))
    with pytest.raises(ValueError):
        model.stage_condition(1, stage_prompts={1: BASE}, batch=2)
    # condition dropout swaps in that stage's null ("" prompt) embedding
    for k in range(3):
        assert torch.equal(model.encode_stage(k, prompt=BASE, drop_prob=1.0, batch=2), model.null_embedding(k, 2))


# ---- (d) shared, frozen backbone ------------------------------------------------
def test_backbone_shared_and_frozen(backbone):
    model = clip_model(backbone)
    assert all(enc.backbone is backbone for enc in model.encoders)
    assert model.text_backbone is backbone
    assert all(not p.requires_grad for p in backbone.parameters())
    assert not backbone.model.training
    clip_ids = {id(p) for p in backbone.parameters()}
    model_params = list(model.parameters())
    assert not clip_ids & {id(p) for p in model_params}  # CLIP not trainable through the cascade
    # nothing of CLIP in the state dict / per-stage checkpoints: only the small adapters
    assert {k.split(".")[0] for k in model.state_dict()} == {"encoders", "stages"}
    for k in range(3):
        enc_state = model.stage_state(k)["encoder"]
        assert set(enc_state) == {"adapter.0.weight", "adapter.0.bias", "adapter.1.weight", "adapter.1.bias"}
    n_clip = sum(p.numel() for p in backbone.parameters())
    n_adapters = sum(p.numel() for e in model.encoders for p in e.parameters())
    assert n_adapters < n_clip / 100
    # training a stage: gradients reach the adapter, never CLIP
    x = torch.rand(2, 3, 64, 64) * 2 - 1
    model.train()
    loss = model.stage_loss(1, x, torch.tensor([3, 7]), stage_prompts={2: DETAIL})
    loss.backward()
    assert model.encoders[1].adapter[1].weight.grad is not None
    assert all(p.grad is None for p in backbone.parameters())
    # a second cascade reuses the same backbone by default (loaded once per process)
    cfg = default_config()
    cfg.cond_type = "clip"
    assert Cascade(cfg).text_backbone is load_clip_backbone(DEFAULT_CLIP_MODEL)


def test_clip_checkpoint_roundtrip(backbone, tmp_path):
    model = clip_model(backbone)
    torch.save({"config": model.cfg.to_dict(), "state": model.stage_state(1), "encoder": encoder_info(model.cfg)},
               tmp_path / "stage2.pt")
    ck = torch.load(tmp_path / "stage2.pt")
    assert ck["encoder"]["type"] == "clip" and ck["encoder"]["model_id"] == DEFAULT_CLIP_MODEL
    assert (tmp_path / "stage2.pt").stat().st_size < 10_000_000  # adapters + UNet only, no CLIP weights
    from c2f import CascadeConfig
    m2 = Cascade(CascadeConfig.from_dict(ck["config"]), text_backbone=backbone).eval()
    m2.load_stage_state(1, ck["state"])
    assert torch.equal(m2.encode_stage(1, prompt=DETAIL, batch=2), model.encode_stage(1, prompt=DETAIL, batch=2))


# ---- (e) helpful errors ---------------------------------------------------------
def test_missing_transformers_error_message(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *a, **kw):
        if name == "transformers" or name.startswith("transformers."):
            raise ImportError("No module named 'transformers'")
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(CLIPUnavailableError) as ei:
        CLIPTextBackbone("openai/clip-vit-base-patch32")
    msg = str(ei.value)
    for needle in ("transformers", "pip install", "openai/clip-vit-base-patch32", "HF_HUB_OFFLINE",
                   "--clip-model", "--cond text"):
        assert needle in msg, needle
    # and the cascade surfaces the same error when built with cond_type="clip" (no reuse cache hit)
    cfg = default_config()
    cfg.cond_type, cfg.clip_model = "clip", "some-org/not-loaded-yet"
    with pytest.raises(CLIPUnavailableError):
        Cascade(cfg)


def test_bad_model_error_message(monkeypatch, tmp_path):
    pytest.importorskip("transformers", reason="transformers not installed")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    missing = str(tmp_path / "no-such-clip")
    with pytest.raises(CLIPUnavailableError) as ei:
        load_clip_backbone(missing, reuse=False)
    msg = str(ei.value)
    assert missing in msg and "HF_HUB_OFFLINE" in msg and "--clip-model" in msg and "pip install" in msg


def test_stub_and_label_modes_need_no_transformers(monkeypatch):
    real_import = builtins.__import__
    monkeypatch.setattr(builtins, "__import__", lambda name, *a, **kw: (_ for _ in ()).throw(ImportError(name))
                        if name.startswith("transformers") else real_import(name, *a, **kw))
    for cond in ("label", "text"):
        cfg = default_config()
        cfg.cond_type = cond
        for s in cfg.stages:
            s.base_channels = 16
        assert Cascade(cfg).text_backbone is None
    assert encoder_info(cfg) == {"type": "text"}
