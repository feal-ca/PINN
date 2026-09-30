"""Steady incompressible Navier-Stokes residuals, via autograd.

Given a callable `model(x, y) -> (u, v, p)`, this computes how badly that
prediction violates the governing equations at each (x, y) collocation
point:

  continuity:   u_x + v_y = 0
  momentum-x:   u*u_x + v*u_y + p_x - nu*(u_xx + u_yy) = 0
  momentum-y:   u*v_x + v*v_y + p_y - nu*(v_xx + v_yy) = 0

Training pushes these toward zero everywhere in the fluid domain, which is
what lets physics -- rather than more data -- fill in the field far from
the sparse sensors.
"""

import torch


def _grad(outputs: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
    """d(outputs)/d(inputs), keeping the graph so second derivatives work.

    `allow_unused` + `materialize_grads` handle the (rare but real) case
    where `outputs` is analytically independent of `inputs` -- e.g. taking
    d/dy of something that only depends on x -- which otherwise makes
    autograd raise instead of returning zero.
    """
    return torch.autograd.grad(
        outputs, inputs, grad_outputs=torch.ones_like(outputs),
        create_graph=True, retain_graph=True, allow_unused=True, materialize_grads=True,
    )[0]


def ns_residuals(
    model, x: torch.Tensor, y: torch.Tensor, nu: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x = x.clone().requires_grad_(True)
    y = y.clone().requires_grad_(True)
    u, v, p = model(x, y)

    u_x, u_y = _grad(u, x), _grad(u, y)
    v_x, v_y = _grad(v, x), _grad(v, y)
    p_x, p_y = _grad(p, x), _grad(p, y)
    u_xx, u_yy = _grad(u_x, x), _grad(u_y, y)
    v_xx, v_yy = _grad(v_x, x), _grad(v_y, y)

    continuity = u_x + v_y
    momentum_u = u * u_x + v * u_y + p_x - nu * (u_xx + u_yy)
    momentum_v = u * v_x + v * v_y + p_y - nu * (v_xx + v_yy)
    return continuity, momentum_u, momentum_v


def unsteady_ns_residuals(
    model, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor, nu: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Phase 2's residuals: same as `ns_residuals` plus the local
    acceleration terms (u_t, v_t) that a steady flow doesn't have.
    Continuity has no explicit time derivative -- incompressibility is
    still an instantaneous, per-snapshot constraint, not a statement about
    how divergence evolves.
    """
    x = x.clone().requires_grad_(True)
    y = y.clone().requires_grad_(True)
    t = t.clone().requires_grad_(True)
    u, v, p = model(x, y, t)

    u_x, u_y, u_t = _grad(u, x), _grad(u, y), _grad(u, t)
    v_x, v_y, v_t = _grad(v, x), _grad(v, y), _grad(v, t)
    p_x, p_y = _grad(p, x), _grad(p, y)
    u_xx, u_yy = _grad(u_x, x), _grad(u_y, y)
    v_xx, v_yy = _grad(v_x, x), _grad(v_y, y)

    continuity = u_x + v_y
    momentum_u = u_t + u * u_x + v * u_y + p_x - nu * (u_xx + u_yy)
    momentum_v = v_t + u * v_x + v * v_y + p_y - nu * (v_xx + v_yy)
    return continuity, momentum_u, momentum_v
