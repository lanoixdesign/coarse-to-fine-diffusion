"""Classifier-free guidance in the DDIM sampler and Cascade.sample, and --cond-source gray."""
import torch

from c2f import Cascade, Diffusion, default_config
from c2f.data import C2FMNIST


def _model(seed=0):
    torch.manual_seed(seed)
    m = Cascade(default_config())
    m.eval()
    return m


def test_guidance_one_is_unguided_and_zero_is_unconditional():
    m = _model()
    labels = torch.tensor([1, 7, 3])
    torch.manual_seed(5)
    ref = m.sample(labels, steps=4, upto=1)[0]
    torch.manual_seed(5)
    same = m.sample(labels, steps=4, upto=1, guidance_scale=1.0)[0]
    assert torch.equal(ref, same)
    # w = 0 -> pure unconditional branch = sampling with the null embedding as condition
    d, net = m.diffusion, m.stages[0]
    emb, null = m.encode_stage(0, labels), m.null_embedding(0, 3)
    torch.manual_seed(5)
    w0 = d.sample(net, (3, 1, 16, 16), None, emb, steps=4, guidance_scale=0.0, null_emb=null)
    torch.manual_seed(5)
    unc = d.sample(net, (3, 1, 16, 16), None, null, steps=4)
    assert torch.allclose(w0, unc, atol=1e-3)  # batched vs unbatched float rounding, amplified by 1/sqrt(ab)


def test_guidance_formula():
    d = Diffusion(1000)
    c, u = torch.ones(2, 4), torch.zeros(2, 4)

    def fake(x, t, emb):  # eps_c = 0.1, eps_u = -0.1 (per-row, from the embedding)
        return 0.1 * (2 * emb[:, :1, None, None] - 1).expand_as(x)

    seen = {}
    orig = d.alphas_cumprod
    torch.manual_seed(0)
    x = d.sample(fake, (2, 1, 2, 2), None, c, steps=1, guidance_scale=3.0, null_emb=u)
    # single DDIM step to t=0 with eps = -0.1 + 3*(0.1-(-0.1)) = 0.5: x0 = (x_T - sqrt(1-ab) eps)/sqrt(ab)
    torch.manual_seed(0)
    xT = torch.randn(2, 1, 2, 2)
    ab = orig[999]
    x0 = ((xT - (1 - ab).sqrt() * 0.5) / ab.sqrt()).clamp(-1, 1)
    assert torch.allclose(x, x0, atol=1e-5)


def test_per_stage_guidance_and_shapes():
    m = _model()
    outs = m.sample(torch.tensor([0, 9]), steps=2, guidance_scale=[2.0, 1.0, 1.5])
    assert [tuple(o.shape) for o in outs] == [(2, 1, 16, 16), (2, 3, 32, 32), (2, 3, 64, 64)]


def test_cond_source_changes_only_the_conditioning():
    m = _model()
    x = torch.rand(2, 3, 64, 64) * 2 - 1
    y = torch.tensor([2, 5])
    torch.manual_seed(1)
    a = m.stage_loss(1, x, y)
    torch.manual_seed(1)
    b = m.stage_loss(1, x, y, cond_source=x)
    assert torch.equal(a, b)
    torch.manual_seed(1)
    c = m.stage_loss(1, x, y, cond_source=-x)
    assert not torch.equal(a, c)


def test_gray_rendering_is_neutral(tmp_path):
    try:
        ds = C2FMNIST("data", True, 64, True, download=False, with_gray=True)
    except Exception:  # MNIST not available offline (CI)
        import pytest
        pytest.skip("MNIST not downloaded")
    xc, label, xg = ds[0]
    assert xg.shape == xc.shape == (3, 64, 64)
    assert xg[1:].abs().max() < 1e-5  # neutral chroma
    assert xg[0].max() > 0.9 and xg[0].min() < -0.99  # white strokes on black
