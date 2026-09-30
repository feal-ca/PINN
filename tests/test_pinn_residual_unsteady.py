"""Manufactured-solution test for the Phase 2 unsteady residual (M11).

Extends the Phase 1 polynomial test with a time term: u = x^2*y + t,
v = -x*y^2 (unchanged, so continuity is still exactly zero), p = x+y.
Hand-derived:

  u_t = 1
  momentum_u = 2 + x^3*y^2 + 2*x*y*t - 2*nu*y
  momentum_v = x^2*y^3 - t*y^2 + 1 + 2*nu*x
"""

import torch

from pinn_bloodflow.pinn.physics import unsteady_ns_residuals


def polynomial_model_t(x, y, t):
    u = x**2 * y + t
    v = -x * y**2
    p = x + y
    return u, v, p


def test_unsteady_manufactured_solution_matches_hand_derived_residuals():
    torch.manual_seed(0)
    x = torch.rand(20, 1) * 2 + 0.1
    y = torch.rand(20, 1) * 2 + 0.1
    t = torch.rand(20, 1) * 2 + 0.1
    nu = 0.05

    continuity, momentum_u, momentum_v = unsteady_ns_residuals(polynomial_model_t, x, y, t, nu)

    x_np, y_np, t_np = x.detach(), y.detach(), t.detach()
    expected_continuity = torch.zeros_like(x_np)
    expected_momentum_u = 2 + x_np**3 * y_np**2 + 2 * x_np * y_np * t_np - 2 * nu * y_np
    expected_momentum_v = x_np**2 * y_np**3 - t_np * y_np**2 + 1 + 2 * nu * x_np

    assert torch.allclose(continuity, expected_continuity, atol=1e-5)
    assert torch.allclose(momentum_u, expected_momentum_u, atol=1e-4)
    assert torch.allclose(momentum_v, expected_momentum_v, atol=1e-4)


def test_unsteady_residuals_require_grad_for_training():
    x = torch.rand(5, 1, requires_grad=True)
    y = torch.rand(5, 1, requires_grad=True)
    t = torch.rand(5, 1, requires_grad=True)
    continuity, momentum_u, momentum_v = unsteady_ns_residuals(polynomial_model_t, x, y, t, nu=0.01)
    loss = (continuity**2).mean() + (momentum_u**2).mean() + (momentum_v**2).mean()
    loss.backward()  # should not raise
