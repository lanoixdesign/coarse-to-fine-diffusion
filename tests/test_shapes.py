import torch

from c2f import Cascade, UNet, default_config, rgb_to_ycbcr, ycbcr_to_rgb
from c2f.cascade import stage_target, to_ycbcr3


def small_cfg(cond="label"):
    cfg = default_config()
    cfg.cond_type = cond
    cfg.timesteps = 50
    for s in cfg.stages:
        s.base_channels = 16
    return cfg


def test_color_roundtrip():
    rgb = torch.rand(4, 3, 8, 8)
    assert torch.allclose(ycbcr_to_rgb(rgb_to_ycbcr(rgb)), rgb, atol=1e-4)


def test_unet_shapes():
    for res in (16, 32, 64):
        net = UNet(in_channels=4, out_channels=3, cond_dim=32, base_channels=16)
        y = net(torch.randn(2, 4, res, res), torch.randint(0, 1000, (2,)), torch.randn(2, 32))
        assert y.shape == (2, 3, res, res)


def test_stage_targets():
    cfg = small_cfg()
    x = torch.randn(2, 3, 32, 32)
    assert stage_target(x, cfg.stages[0]).shape == (2, 1, 32, 32)
    assert stage_target(x, cfg.stages[1]).shape == (2, 3, 32, 32)


def test_cascade_loss_and_sample():
    for cond in ("label", "text"):
        torch.manual_seed(0)
        cfg = small_cfg(cond)
        model = Cascade(cfg)
        x = torch.rand(2, 3, 64, 64) * 2 - 1
        y = torch.tensor([3, 7])
        for k in range(3):
            loss = model.stage_loss(k, x, y)
            assert loss.ndim == 0 and torch.isfinite(loss)
            loss.backward()
        outs = model.sample(y, steps=2)
        assert [o.shape for o in outs] == [(2, 1, 16, 16), (2, 3, 32, 32), (2, 3, 64, 64)]
        assert to_ycbcr3(outs[0], cfg.stages[0].channels).shape == (2, 3, 16, 16)
