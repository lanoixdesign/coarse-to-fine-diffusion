"""EMA of stage weights (c2f.ema) and the train.py LR schedule helper."""
import importlib.util
import os

import pytest
import torch

from c2f import Cascade, StageEMA, default_config, stage_state_from_checkpoint


def small_model(cond="label"):
    cfg = default_config()
    cfg.cond_type = cond
    cfg.timesteps = 50
    for s in cfg.stages:
        s.base_channels = 16
    return Cascade(cfg)


def train_steps(model, k, ema, n=5):
    params = list(model.stages[k].parameters()) + list(model.encoders[k].parameters())
    opt = torch.optim.AdamW(params, lr=1e-2)
    for _ in range(n):
        loss = model.stage_loss(k, torch.rand(4, 3, 64, 64) * 2 - 1, torch.tensor([1, 2, 3, 4]))
        opt.zero_grad()
        loss.backward()
        opt.step()
        ema.update()


def test_ema_state_layout_matches_stage_state():
    torch.manual_seed(0)
    model = small_model()
    for k in range(3):
        ema = StageEMA(model, k, 0.999)
        raw, avg = model.stage_state(k), ema.state_dict()
        assert raw.keys() == avg.keys() == {"unet", "encoder"}
        for part in raw:
            assert raw[part].keys() == avg[part].keys()
            for n in raw[part]:
                assert raw[part][n].shape == avg[part][n].shape and raw[part][n].dtype == avg[part][n].dtype
                assert torch.equal(raw[part][n], avg[part][n])  # starts as a copy
        model.load_stage_state(k, avg)  # loadable with the existing API


def test_ema_differs_from_raw_after_steps_and_warmup():
    torch.manual_seed(0)
    model = small_model()
    init = {n: t.clone() for n, t in model.stages[0].state_dict().items()}
    ema = StageEMA(model, 0, 0.999)
    assert ema.current_decay() == pytest.approx(0.1)  # warmup: (1+0)/(10+0)
    train_steps(model, 0, ema, n=5)
    assert ema.num_updates == 5
    raw, avg = model.stage_state(0)["unet"], ema.state_dict()["unet"]
    w = "inp.weight"
    assert not torch.allclose(raw[w], avg[w])
    assert not torch.allclose(init[w], avg[w])  # it moved towards the raw weights
    # with a constant 0.999 decay (no warmup) the average barely moves away from init
    torch.manual_seed(0)
    model2 = small_model()
    ema2 = StageEMA(model2, 0, 0.999, warmup=False)
    train_steps(model2, 0, ema2, n=5)
    d_warm = (avg[w] - init[w]).abs().mean()
    d_const = (ema2.state_dict()["unet"][w] - init[w]).abs().mean()
    assert d_const < d_warm


def test_ema_swapped_in_restores_raw_weights():
    torch.manual_seed(0)
    model = small_model()
    ema = StageEMA(model, 0, 0.99)
    train_steps(model, 0, ema, n=3)
    raw = {n: t.clone() for n, t in model.stages[0].state_dict().items()}
    with ema.swapped_in():
        assert torch.equal(model.stages[0].state_dict()["inp.weight"], ema.shadow["unet"]["inp.weight"])
    for n, t in model.stages[0].state_dict().items():
        assert torch.equal(t, raw[n])


def test_checkpoint_weight_selection_backward_compatible():
    model = small_model()
    raw = model.stage_state(0)
    old = {"state": raw}  # checkpoint written before EMA existed
    assert stage_state_from_checkpoint(old) == (raw, False)
    assert stage_state_from_checkpoint(old, use_ema=False) == (raw, False)
    with pytest.raises(KeyError):
        stage_state_from_checkpoint(old, use_ema=True)
    avg = StageEMA(model, 0, 0.999).state_dict()
    new = {"state": raw, "ema_state": avg}
    assert stage_state_from_checkpoint(new)[1] is True  # default: EMA when present
    assert stage_state_from_checkpoint(new, use_ema=False) == (raw, False)


def test_ema_decay_validation():
    with pytest.raises(ValueError):
        StageEMA(small_model(), 0, 1.0)


def _train_module():
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "train.py")
    spec = importlib.util.spec_from_file_location("c2f_train", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_lr_factor_defaults_and_schedules():
    f = _train_module().lr_factor
    assert all(f(s, 100, 0, "constant", 0.0) == 1.0 for s in range(1, 101))  # default: unchanged LR
    assert f(5, 100, 10, "constant", 0.0) == pytest.approx(0.5)
    assert f(10, 100, 10, "cosine", 0.0) == pytest.approx(1.0)
    assert f(100, 100, 10, "cosine", 0.1) == pytest.approx(0.1)
    assert f(55, 100, 10, "linear", 0.0) == pytest.approx(0.5)
