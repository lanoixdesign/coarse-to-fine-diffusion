"""DDIM sampler vs the training forward process (regression guard for the sanity check)."""
import torch

from c2f import Diffusion


def test_ddim_with_oracle_eps_follows_forward_process():
    """With the exact noise as the model output, every DDIM step must land on
    q_sample(x0, t_next, eps): same alpha_bar indexing as training, final output = x0."""
    torch.manual_seed(0)
    d = Diffusion(1000)
    x0 = torch.rand(3, 1, 16, 16) * 2 - 1
    seen = []

    def oracle(x, t, cond):
        ab = d.alphas_cumprod[t].view(-1, 1, 1, 1)
        eps = (x - ab.sqrt() * x0) / (1 - ab).sqrt()
        seen.append((int(t[0]), x.clone(), eps))
        return eps

    out = d.sample(oracle, x0.shape, None, None, steps=20)
    assert torch.allclose(out, x0, atol=1e-5)
    assert seen[0][0] == 999 and seen[-1][0] == 0  # timestep range = training range [0, T-1]
    eps_T = seen[0][2]
    for t, x, _ in seen[1:]:
        assert torch.allclose(x, d.q_sample(x0, torch.full((3,), t), eps_T), atol=1e-4)


def test_loss_uses_same_forward_process():
    d = Diffusion(1000)
    x0 = torch.zeros(2, 1, 4, 4)
    noise = torch.randn_like(x0)
    t = torch.tensor([0, 999])
    xt = d.q_sample(x0, t, noise)
    ab = d.alphas_cumprod[t].view(-1, 1, 1, 1)
    assert torch.allclose(xt, (1 - ab).sqrt() * noise)
