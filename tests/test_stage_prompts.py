"""Per-stage prompt conditioning: shapes, fallback rules, backward compatibility."""
import pytest
import torch

from c2f import Cascade, default_config, labels_to_prompts, normalize_stage_prompts

BASE = "a handwritten digit seven"
DETAIL = "a bold red handwritten seven with thick smooth strokes on a pale blue background"
SHAPES = [(2, 1, 16, 16), (2, 3, 32, 32), (2, 3, 64, 64)]


def small_model(cond="text", seed=0):
    torch.manual_seed(seed)
    cfg = default_config()
    cfg.cond_type = cond
    cfg.timesteps = 50
    for s in cfg.stages:
        s.base_channels = 16
    return Cascade(cfg).eval()


def x_batch(b=2):
    g = torch.Generator().manual_seed(123)
    return torch.rand(b, 3, 64, 64, generator=g) * 2 - 1


def sample(model, seed=0, **kw):
    torch.manual_seed(seed)
    return model.sample(steps=2, **kw)


def losses(model, x, seed=0, **kw):
    out = []
    for k in range(3):
        torch.manual_seed(seed)
        out.append(model.stage_loss(k, x, **kw))
    return out


# ---- shapes -----------------------------------------------------------------
def test_per_stage_prompt_shapes_sample_and_loss():
    model = small_model()
    sp = {2: DETAIL, 3: [DETAIL, DETAIL + " glossy"]}
    outs = sample(model, prompt=BASE, stage_prompts=sp, num_samples=2)
    assert [o.shape for o in outs] == SHAPES
    for k in range(3):
        emb = model.encode_stage(k, prompt=BASE, stage_prompts=sp, batch=2)
        assert emb.shape == (2, model.cfg.cond_dim)
        loss = model.stage_loss(k, x_batch(), prompt=BASE, stage_prompts=sp)
        assert loss.ndim == 0 and torch.isfinite(loss)
        loss.backward()


def test_all_stage_prompts_without_base_and_list_form():
    model = small_model()
    outs = sample(model, stage_prompts=[BASE, DETAIL, DETAIL], num_samples=3)
    assert [o.shape[0] for o in outs] == [3, 3, 3]
    # batch inferred from a list of per-sample prompts
    outs = sample(model, prompt=[BASE, BASE])
    assert [o.shape for o in outs] == SHAPES


def test_missing_prompt_raises():
    model = small_model()
    with pytest.raises(ValueError):
        model.stage_condition(1, stage_prompts={1: BASE}, batch=2)  # stage 2 has nothing to fall back to


# ---- fallback equivalence ---------------------------------------------------
def test_base_prompt_equals_same_prompt_on_every_stage():
    model = small_model()
    a = sample(model, prompt=BASE, num_samples=2)
    b = sample(model, prompt=BASE, stage_prompts={1: BASE, 2: BASE, 3: BASE}, num_samples=2)
    c = sample(model, stage_prompts=[BASE, BASE, BASE], num_samples=2)
    d = sample(model, prompt="something else", stage_prompts={"s1": BASE, "stage2": BASE, 3: BASE}, num_samples=2)
    for xs in (b, c, d):
        for u, v in zip(a, xs):
            assert torch.equal(u, v)
    x = x_batch()
    model.train()  # includes condition dropout (drop_prob 0.1) -> same RNG path in both
    la = losses(model, x, prompt=BASE)
    lb = losses(model, x, stage_prompts={1: BASE, 2: BASE, 3: BASE})
    assert all(torch.equal(u, v) for u, v in zip(la, lb))


def test_partial_override_falls_back_to_base():
    model = small_model()
    for k in (0, 2):
        assert torch.equal(model.encode_stage(k, prompt=BASE, stage_prompts={2: DETAIL}, batch=2),
                           model.encode_stage(k, prompt=BASE, batch=2))


def test_labels_are_the_default_base_prompt_in_text_mode():
    model = small_model("text")
    y = torch.tensor([7, 7])
    a = sample(model, labels=y)
    b = sample(model, prompt=labels_to_prompts([7, 7]))
    assert labels_to_prompts([7]) == [BASE]
    for u, v in zip(a, b):
        assert torch.equal(u, v)


# ---- per-stage prompts actually change the stage's conditioning -------------
def test_different_stage_prompt_changes_only_that_stage_embedding():
    model = small_model()
    sp = {2: DETAIL, 3: DETAIL}
    e_base = [model.encode_stage(k, prompt=BASE, batch=2) for k in range(3)]
    e_sp = [model.encode_stage(k, prompt=BASE, stage_prompts=sp, batch=2) for k in range(3)]
    assert torch.equal(e_base[0], e_sp[0])
    assert not torch.allclose(e_base[1], e_sp[1])
    assert not torch.allclose(e_base[2], e_sp[2])
    # stage 1 unchanged -> identical 16x16 output; later stages differ
    a = sample(model, prompt=BASE, num_samples=2)
    b = sample(model, prompt=BASE, stage_prompts=sp, num_samples=2)
    assert torch.equal(a[0], b[0])
    assert not torch.allclose(a[2], b[2])


def test_null_embedding_path_per_stage():
    model = small_model()
    for k in range(3):
        null = model.null_embedding(k, 2)
        assert null.shape == (2, model.cfg.cond_dim)
        # drop_prob=1 -> every row is that stage's null embedding, whatever the prompt
        emb = model.encode_stage(k, prompt=BASE, stage_prompts={k + 1: DETAIL}, drop_prob=1.0, batch=2)
        assert torch.equal(emb, null)


# ---- label conditioning / backward compatibility ----------------------------
def test_old_call_signatures_unchanged():
    for cond in ("label", "text"):
        model = small_model(cond)
        y = torch.tensor([3, 7])
        x = x_batch()
        outs = sample(model, labels=y)
        torch.manual_seed(0)
        outs_pos = model.sample(y, 2)  # positional, as in the original API
        assert [o.shape for o in outs_pos] == SHAPES
        assert all(torch.equal(u, v) for u, v in zip(outs, outs_pos))
        torch.manual_seed(0)
        assert torch.isfinite(model.stage_loss(1, x, y))
        assert model.encode(0, y).shape == (2, model.cfg.cond_dim)
        assert model.encode(0, y, 0.5).shape == (2, model.cfg.cond_dim)


def test_label_mode_rejects_text_but_allows_per_stage_labels():
    model = small_model("label")
    y = torch.tensor([3, 7])
    with pytest.raises(TypeError):
        model.stage_condition(0, labels=y, prompt=BASE)
    with pytest.raises(TypeError):
        model.stage_condition(1, labels=y, stage_prompts={2: DETAIL})
    cond = model.stage_condition(1, labels=y, stage_prompts={2: torch.tensor([1, 1])})
    assert cond.tolist() == [1, 1]
    assert model.stage_condition(0, labels=y, stage_prompts={2: 5}).tolist() == [3, 7]
    outs = sample(model, labels=y, stage_prompts={3: 9})
    assert [o.shape for o in outs] == SHAPES


def test_state_dict_keys_unchanged():
    model = small_model("text")
    keys = set(model.state_dict())
    assert {k.split(".")[0] for k in keys} == {"encoders", "stages"}
    assert set(model.stage_state(0)) == {"unet", "encoder"}


def test_normalize_stage_prompts():
    assert normalize_stage_prompts(None, 3) == [None, None, None]
    assert normalize_stage_prompts({2: "a"}, 3) == [None, "a", None]
    assert normalize_stage_prompts({"s3": "c", "stage1": "a"}, 3) == ["a", None, "c"]
    assert normalize_stage_prompts(["a", None], 3) == ["a", None, None]
    with pytest.raises(KeyError):
        normalize_stage_prompts({0: "a"}, 3)  # stages are 1-based
    with pytest.raises(TypeError):
        normalize_stage_prompts("a", 3)
    with pytest.raises(ValueError):
        normalize_stage_prompts(["a", "b", "c", "d"], 3)
