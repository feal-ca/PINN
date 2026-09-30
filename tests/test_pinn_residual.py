"""Manufactured-solution test for the autograd NS residual machinery.

Uses u = x^2*y, v = -x*y^2, p = x+y as a stand-in "model" (not a trained
network) because it's divergence-free (u_x + v_y = 2xy - 2xy = 0) and its
momentum residuals can be worked out by hand:

  momentum_u = x^3*y^2 + 1 - 2*nu*y
  momentum_v = x^2*y^3 + 1 + 2*nu*x

If `ns_residuals` has an indexing, sign, or chain-rule bug, this is exactly
the kind of test that catches it -- it checks the actual numbers, not just
"does it run without crashing."
"""

import torch

from pinn_bloodflow.pinn.physics import ns_residuals


def polynomial_model(x, y):
    u = x**2 * y
    v = -x * y**2
    p = x + y
    return u, v, p


def test_manufactured_solution_matches_hand_derived_residuals():
    torch.manual_seed(0)
    x = torch.rand(20, 1) * 2 + 0.1
    y = torch.rand(20, 1) * 2 + 0.1
    nu = 0.05

    continuity, momentum_u, momentum_v = ns_residuals(polynomial_model, x, y, nu)

    x_np, y_np = x.detach(), y.detach()
    expected_continuity = torch.zeros_like(x_np)
    expected_momentum_u = x_np**3 * y_np**2 + 1 - 2 * nu * y_np
    expected_momentum_v = x_np**2 * y_np**3 + 1 + 2 * nu * x_np

    assert torch.allclose(continuity, expected_continuity, atol=1e-5)
    assert torch.allclose(momentum_u, expected_momentum_u, atol=1e-4)
    assert torch.allclose(momentum_v, expected_momentum_v, atol=1e-4)


def test_residuals_require_grad_for_training():
    # The residuals must stay attached to the model's parameters (via x, y's
    # graph) so a training loop can backprop through them.
    x = torch.rand(5, 1, requires_grad=True)
    y = torch.rand(5, 1, requires_grad=True)
    continuity, momentum_u, momentum_v = ns_residuals(polynomial_model, x, y, nu=0.01)
    loss = (continuity**2).mean() + (momentum_u**2).mean() + (momentum_v**2).mean()
    loss.backward()  # should not raise
