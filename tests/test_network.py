import torch

from pinn_bloodflow.pinn.network import PINN


def test_output_shapes():
    model = PINN(in_dim=2, hidden_layers=3, hidden_units=8)
    x = torch.rand(10, 1)
    y = torch.rand(10, 1)
    u, v, p = model(x, y)
    assert u.shape == (10, 1)
    assert v.shape == (10, 1)
    assert p.shape == (10, 1)


def test_three_input_dims_for_phase2():
    model = PINN(in_dim=3, hidden_layers=2, hidden_units=8)
    x, y, t = torch.rand(5, 1), torch.rand(5, 1), torch.rand(5, 1)
    u, v, p = model(x, y, t)
    assert u.shape == (5, 1)


def test_gradients_flow_to_all_parameters():
    model = PINN(in_dim=2, hidden_layers=3, hidden_units=8)
    x, y = torch.rand(4, 1), torch.rand(4, 1)
    u, v, p = model(x, y)
    (u.sum() + v.sum() + p.sum()).backward()
    assert all(p.grad is not None for p in model.parameters())


def test_fourier_time_features_are_exactly_periodic():
    # freq = k/period for integer k -- sin/cos of 2*pi*freq*t must repeat
    # exactly every `period`, so the network's output should too, even
    # though it was never explicitly trained on t and t+period together.
    period = 1.3
    model = PINN(in_dim=3, hidden_layers=2, hidden_units=8, fourier_time_freqs=[1 / period, 2 / period])
    x, y = torch.rand(6, 1), torch.rand(6, 1)
    t = torch.rand(6, 1)
    u0, v0, p0 = model(x, y, t)
    u1, v1, p1 = model(x, y, t + period)
    assert torch.allclose(u0, u1, atol=1e-5)
    assert torch.allclose(v0, v1, atol=1e-5)
    assert torch.allclose(p0, p1, atol=1e-5)


def test_fourier_time_features_still_allow_gradients():
    model = PINN(in_dim=3, hidden_layers=2, hidden_units=8, fourier_time_freqs=[1.0, 2.0])
    x, y = torch.rand(4, 1), torch.rand(4, 1)
    t = torch.rand(4, 1, requires_grad=True)
    u, v, p = model(x, y, t)
    u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    assert u_t.shape == t.shape
    (u.sum() + v.sum() + p.sum()).backward()
    assert all(param.grad is not None for param in model.parameters())
