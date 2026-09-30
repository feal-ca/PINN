"""The individual loss terms, and how they're combined.

Loss-term weighting is a genuinely open question in the PINN literature
(fixed weights vs. gradient-norm/NTK-based adaptive schemes) -- these
starting weights come from common practice in the inverse-blood-flow
papers this project is built on, not a resolved optimum. Worth watching
the per-term curves during training rather than trusting them blindly.
"""

import torch

from pinn_bloodflow.pinn.physics import ns_residuals, unsteady_ns_residuals


def data_loss(model, x: torch.Tensor, y: torch.Tensor, u_target: torch.Tensor, v_target: torch.Tensor) -> torch.Tensor:
    u_pred, v_pred, _ = model(x, y)
    return torch.mean((u_pred - u_target) ** 2) + torch.mean((v_pred - v_target) ** 2)


def data_loss_t(
    model, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor, u_target: torch.Tensor, v_target: torch.Tensor
) -> torch.Tensor:
    u_pred, v_pred, _ = model(x, y, t)
    return torch.mean((u_pred - u_target) ** 2) + torch.mean((v_pred - v_target) ** 2)


def pde_loss(model, x: torch.Tensor, y: torch.Tensor, nu: torch.Tensor) -> torch.Tensor:
    continuity, momentum_u, momentum_v = ns_residuals(model, x, y, nu)
    return torch.mean(continuity**2) + torch.mean(momentum_u**2) + torch.mean(momentum_v**2)


def pde_loss_t(model, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor, nu: torch.Tensor) -> torch.Tensor:
    continuity, momentum_u, momentum_v = unsteady_ns_residuals(model, x, y, t, nu)
    return torch.mean(continuity**2) + torch.mean(momentum_u**2) + torch.mean(momentum_v**2)


def no_slip_loss(model, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    u, v, _ = model(x, y)
    return torch.mean(u**2) + torch.mean(v**2)


def no_slip_loss_t(model, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    u, v, _ = model(x, y, t)
    return torch.mean(u**2) + torch.mean(v**2)


def inlet_loss(model, x: torch.Tensor, y: torch.Tensor, u_profile: torch.Tensor, v_profile: torch.Tensor) -> torch.Tensor:
    u, v, _ = model(x, y)
    return torch.mean((u - u_profile) ** 2) + torch.mean((v - v_profile) ** 2)


def inlet_loss_t(
    model, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor, u_profile: torch.Tensor, v_profile: torch.Tensor
) -> torch.Tensor:
    u, v, _ = model(x, y, t)
    return torch.mean((u - u_profile) ** 2) + torch.mean((v - v_profile) ** 2)


def outlet_loss(model, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """The outflow condition the CFD solver uses: zero pressure and zero
    streamwise velocity gradient on the outlet line."""
    x = x.clone().requires_grad_(True)
    u, v, p = model(x, y)
    ones = torch.ones_like(u)
    u_x = torch.autograd.grad(u, x, grad_outputs=ones, create_graph=True)[0]
    v_x = torch.autograd.grad(v, x, grad_outputs=ones, create_graph=True)[0]
    return torch.mean(p**2) + torch.mean(u_x**2) + torch.mean(v_x**2)


def pressure_gauge_loss(model, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # Incompressible pressure is only defined up to an additive constant --
    # every PDE/BC term above only ever sees dp/dx or dp/dy. Without pinning
    # an absolute level somewhere, the network could learn p + const and
    # satisfy every other loss term just as well.
    _, _, p = model(x, y)
    return torch.mean(p**2)


def pressure_gauge_loss_t(model, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    _, _, p = model(x, y, t)
    return torch.mean(p**2)


def periodicity_loss(model, x: torch.Tensor, y: torch.Tensor, t0: torch.Tensor, t1: torch.Tensor) -> torch.Tensor:
    """Soft constraint that the flow at the same (x, y) is the same at t0
    and t0+period (t1). This is what lets the PINN fit one representative
    cycle instead of a multi-cycle transient: it tells the network "assume
    a repeating limit cycle" directly, rather than making it rediscover
    periodicity purely from data spread across a cycle it's only shown
    once. Only checked on velocity, matching `data_loss` -- pressure has no
    sensor target either, so there's nothing anchoring its overall level
    at t0 vs. t1 beyond the shared gauge point.
    """
    u0, v0, _ = model(x, y, t0)
    u1, v1, _ = model(x, y, t1)
    return torch.mean((u0 - u1) ** 2) + torch.mean((v0 - v1) ** 2)


def total_loss(terms: dict[str, torch.Tensor], weights: dict[str, float]) -> torch.Tensor:
    return sum(weights[name] * value for name, value in terms.items())
