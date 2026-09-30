"""The Phase 1 (steady) PINN training loop.

Boundary and inlet points are sampled once and held fixed (they're exact,
known geometry, so there's nothing to gain from resampling them). Collocation
points, by contrast, are resampled periodically so the network doesn't
overfit to one particular random point cloud when approximating "the PDE
residual should be zero everywhere in the fluid domain."

Optional config keys (all default to the original behaviour):

  input_norm        rescale (x, y) to [-1, 1] inside the network
  nu_init_factor    initial viscosity as a multiple of the true value (4.0)
  nu_init_abs       initial viscosity as an absolute value (overrides the factor)
  lr_final_frac     cosine-decay the learning rate down to this fraction
  rar_frac          fraction of collocation points picked where the residual
                    is largest (residual-based adaptive refinement)
  outlet_bc         also impose the solver's outflow condition (p = 0, zero
                    streamwise gradient) instead of a single pressure gauge point
  bc_points         points per boundary segment (100)
  snapshot_epochs   epochs at which to keep a copy of the weights (for animations)
  checkpoint_path   write a resumable checkpoint here every checkpoint_every
                    epochs, and resume from it if it exists
"""

import math
import os
from dataclasses import dataclass, field

import numpy as np
import torch

from pinn_bloodflow.geometry import StenosisParams, is_fluid, wall_curve_points
from pinn_bloodflow.pinn.losses import (
    data_loss, inlet_loss, no_slip_loss, outlet_loss, pde_loss, pressure_gauge_loss, total_loss,
)
from pinn_bloodflow.pinn.network import PINN
from pinn_bloodflow.pinn.physics import ns_residuals
from pinn_bloodflow.solver.chorin import inlet_profile


def _sample_collocation_points(params: StenosisParams, n_points: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    xs = np.empty(n_points)
    ys = np.empty(n_points)
    filled = 0
    while filled < n_points:
        batch = (n_points - filled) * 3
        x_c = rng.uniform(0, params.Lx, size=batch)
        y_c = rng.uniform(0, params.H, size=batch)
        valid = is_fluid(x_c, y_c, params)
        x_ok, y_ok = x_c[valid], y_c[valid]
        take = min(len(x_ok), n_points - filled)
        xs[filled:filled + take] = x_ok[:take]
        ys[filled:filled + take] = y_ok[:take]
        filled += take
    return xs, ys


def _to_col(array: np.ndarray) -> torch.Tensor:
    return torch.tensor(array, dtype=torch.float32).reshape(-1, 1)


@dataclass
class TrainingHistory:
    epoch: list[int] = field(default_factory=list)
    total: list[float] = field(default_factory=list)
    data: list[float] = field(default_factory=list)
    pde: list[float] = field(default_factory=list)
    bc: list[float] = field(default_factory=list)
    nu: list[float] = field(default_factory=list)
    snapshots: list = field(default_factory=list)   # (epoch, state_dict, nu) at cfg['snapshot_epochs']


def build_model(cfg: dict, params: StenosisParams) -> PINN:
    bounds = [(0.0, params.Lx), (0.0, params.H)] if cfg.get("input_norm") else None
    return PINN(in_dim=2, hidden_layers=cfg["hidden_layers"], hidden_units=cfg["hidden_units"], input_bounds=bounds)


def train_pinn(
    sensor_data,
    params: StenosisParams,
    U0: float,
    nu_true_hint: float,
    cfg: dict,
) -> tuple[PINN, torch.nn.Parameter, TrainingHistory]:
    torch.manual_seed(cfg["seed"])
    if cfg.get("threads"):
        torch.set_num_threads(int(cfg["threads"]))

    model = build_model(cfg, params)

    # Learn log(nu) rather than nu: it stays positive by construction and sits
    # on a scale Adam can move sensibly, however small the viscosity is.
    nu0 = cfg["nu_init_abs"] if cfg.get("nu_init_abs") else cfg.get("nu_init_factor", 4.0) * nu_true_hint
    log_nu = torch.nn.Parameter(torch.tensor(float(np.log(nu0))))

    # -- fixed boundary/inlet points -----------------------------------------
    n_bc = cfg.get("bc_points", 100)
    x_bottom, y_bottom = wall_curve_points(params, n=n_bc)
    x_top = np.linspace(0, params.Lx, n_bc)
    y_top = np.full(n_bc, params.H)
    x_wall = np.concatenate([x_bottom, x_top])
    y_wall = np.concatenate([y_bottom, y_top])

    y_inlet = np.linspace(0, params.H, n_bc)
    x_inlet = np.zeros(n_bc)
    u_inlet_profile = inlet_profile(y_inlet, U0, params.H)
    v_inlet_profile = np.zeros(n_bc)

    x_gauge, y_gauge = np.array([params.Lx * 0.98]), np.array([params.H * 0.5])

    wall_t = (_to_col(x_wall), _to_col(y_wall))
    inlet_t = (_to_col(x_inlet), _to_col(y_inlet), _to_col(u_inlet_profile), _to_col(v_inlet_profile))
    gauge_t = (_to_col(x_gauge), _to_col(y_gauge))
    outlet_t = (_to_col(np.full(n_bc, params.Lx)), _to_col(np.linspace(0, params.H, n_bc)))
    use_outlet = bool(cfg.get("outlet_bc"))

    sensor_x = _to_col(sensor_data.x)
    sensor_y = _to_col(sensor_data.y)
    sensor_u = _to_col(sensor_data.u_meas)
    sensor_v = _to_col(sensor_data.v_meas)

    weights = cfg["loss_weights"]
    history = TrainingHistory()
    n_colloc = cfg["collocation_points"]
    rar_frac = cfg.get("rar_frac", 0.0)

    def sample_new_collocation(block: int):
        # Seeded by block number so a resumed run draws the same points.
        rng = np.random.default_rng(cfg["seed"] * 100003 + block)
        n_rar = int(n_colloc * rar_frac)
        x_c, y_c = _sample_collocation_points(params, n_colloc - n_rar, rng)
        if n_rar > 0:
            px, py = _sample_collocation_points(params, n_rar * 6, rng)
            tx, ty = _to_col(px), _to_col(py)
            cont, mu, mv = ns_residuals(model, tx, ty, torch.exp(log_nu).detach())
            score = (cont**2 + mu**2 + mv**2).detach().ravel().numpy()
            keep = np.argsort(score)[-n_rar:]
            x_c, y_c = np.concatenate([x_c, px[keep]]), np.concatenate([y_c, py[keep]])
        return _to_col(x_c), _to_col(y_c)

    colloc_x, colloc_y = sample_new_collocation(0)

    def compute_losses():
        nu = torch.exp(log_nu)
        terms = {
            "data": data_loss(model, sensor_x, sensor_y, sensor_u, sensor_v),
            "pde": pde_loss(model, colloc_x, colloc_y, nu),
            "bc": (
                no_slip_loss(model, *wall_t)
                + inlet_loss(model, inlet_t[0], inlet_t[1], inlet_t[2], inlet_t[3])
                + (outlet_loss(model, *outlet_t) if use_outlet else pressure_gauge_loss(model, *gauge_t))
            ),
        }
        return terms, total_loss(terms, weights)

    def record(epoch: int, terms: dict, loss: torch.Tensor) -> None:
        history.epoch.append(epoch)
        history.total.append(loss.item())
        history.data.append(terms["data"].item())
        history.pde.append(terms["pde"].item())
        history.bc.append(terms["bc"].item())
        history.nu.append(torch.exp(log_nu).item())

    # -- Adam phase --------------------------------------------------------
    # log_nu gets its own learning rate, scaled from the network's.
    nu_lr_scale = cfg.get("nu_lr_scale", 0.3)
    optimizer = torch.optim.Adam([
        {"params": model.parameters(), "lr": cfg["adam_lr"]},
        {"params": [log_nu], "lr": cfg["adam_lr"] * nu_lr_scale},
    ])
    base_lrs = [g["lr"] for g in optimizer.param_groups]
    lr_final = cfg.get("lr_final_frac")
    n_adam = cfg["adam_epochs"]

    ckpt_path = cfg.get("checkpoint_path")
    ckpt_every = cfg.get("checkpoint_every", 1000)
    start_epoch = 0
    if ckpt_path and os.path.exists(ckpt_path):
        state = torch.load(ckpt_path, weights_only=False)
        model.load_state_dict(state["model"])
        log_nu.data = state["log_nu"]
        optimizer.load_state_dict(state["optimizer"])
        history = state["history"]
        start_epoch = state["epoch"]
        colloc_x, colloc_y = sample_new_collocation(start_epoch // cfg["resample_every"])
        print(f"  resumed from epoch {start_epoch}", flush=True)

    snap_epochs = set(cfg.get("snapshot_epochs", []))
    for epoch in range(start_epoch, n_adam):
        if epoch > 0 and epoch % cfg["resample_every"] == 0:
            colloc_x, colloc_y = sample_new_collocation(epoch // cfg["resample_every"])
        if lr_final:
            frac = lr_final + (1 - lr_final) * 0.5 * (1 + math.cos(math.pi * epoch / n_adam))
            for g, b in zip(optimizer.param_groups, base_lrs):
                g["lr"] = b * frac

        optimizer.zero_grad()
        terms, loss = compute_losses()
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0 or epoch == n_adam - 1:
            record(epoch, terms, loss)
        if epoch % 1000 == 0:
            print(
                f"  [adam] epoch={epoch:6d} total={loss.item():.4f} "
                f"data={terms['data'].item():.5f} pde={terms['pde'].item():.5f} "
                f"bc={terms['bc'].item():.5f} nu={torch.exp(log_nu).item():.5f}",
                flush=True,
            )
        if epoch in snap_epochs:
            history.snapshots.append((epoch, {k: v.detach().clone() for k, v in model.state_dict().items()},
                                      torch.exp(log_nu).item()))
        if ckpt_path and (epoch + 1) % ckpt_every == 0:
            torch.save({"model": model.state_dict(), "log_nu": log_nu.data, "optimizer": optimizer.state_dict(),
                        "history": history, "epoch": epoch + 1}, ckpt_path)

    # -- L-BFGS polishing phase ----------------------------------------------
    # log_nu is frozen here on purpose: L-BFGS polishes the field's fit now
    # that Adam has settled on a viscosity estimate, instead of moving that
    # estimate under a very different step-size rule.
    if cfg.get("lbfgs_iters", 0) > 0:
        log_nu.requires_grad_(False)
        lbfgs = torch.optim.LBFGS(
            list(model.parameters()),
            max_iter=cfg["lbfgs_iters"], line_search_fn="strong_wolfe",
        )

        def closure():
            lbfgs.zero_grad()
            _, loss = compute_losses()
            loss.backward()
            return loss

        lbfgs.step(closure)
    terms, loss = compute_losses()
    record(n_adam + cfg.get("lbfgs_iters", 0), terms, loss)
    if snap_epochs:
        history.snapshots.append((n_adam + cfg.get("lbfgs_iters", 0), {k: v.detach().clone() for k, v in model.state_dict().items()},
                                  torch.exp(log_nu).item()))

    return model, log_nu, history
