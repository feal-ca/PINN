"""The Phase 2 (pulsatile) PINN training loop (M11/M12).

Structurally the same two-stage Adam -> L-BFGS recipe as Phase 1's
`train.py`, extended with:
  - a third input (t) everywhere geometry is sampled;
  - the unsteady residual (adds u_t, v_t to the momentum equations);
  - a periodicity loss, so the network is told directly "assume the flow
    repeats every period T" instead of having to infer that purely from
    data that only ever shows it one cycle;
  - an optional warm start from the converged Phase 1 (steady) network:
    Phase 1's first layer maps (x, y) -> hidden units; Phase 2's first
    layer maps (x, y, t) -> hidden units, one extra input column. We copy
    the trained (x, y) columns and every deeper layer verbatim, leaving
    only the new t-column at its random Xavier initialization -- the
    network starts already knowing the flow's spatial structure and only
    has to learn how that structure is modulated over the cycle.

Unlike Phase 1, viscosity is NOT re-inferred here: it's passed in already
recovered from Phase 1's denser, purely-spatial dataset. Re-fitting a
single scalar parameter is exactly the kind of estimate that benefits from
the most data available, and Phase 2's sensor set is *more* dilute in
space per unit of new information (spread across snapshots), not less.
"""

from dataclasses import dataclass, field

import numpy as np
import torch

from pinn_bloodflow.geometry import StenosisParams, is_fluid, stenosis_height, wall_curve_points
from pinn_bloodflow.pinn.losses import (
    data_loss_t, inlet_loss_t, no_slip_loss_t, pde_loss_t, periodicity_loss, pressure_gauge_loss_t, total_loss,
)
from pinn_bloodflow.pinn.network import PINN
from pinn_bloodflow.solver.chorin import inlet_profile
from pinn_bloodflow.solver.waveform import cardiac_pulse


def _sample_fluid_xy(params: StenosisParams, n_points: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
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


def _sample_near_wall_xy(
    params: StenosisParams, n_points: int, rng: np.random.Generator, margin: float = 0.05, throat_sigma: float = 0.4,
) -> tuple[np.ndarray, np.ndarray]:
    """Points within `margin` of a wall (bottom stenosis curve or flat top),
    concentrated near the throat (`x0`) via a Gaussian-distributed x rather
    than uniform over the whole channel length. Used to give the PDE
    residual enough support to constrain wall-shear-stress specifically at
    the throat, where it's evaluated and where the ground-truth curve
    peaks. A first attempt spread near-wall points uniformly along x, which
    mostly landed far from the throat and *hurt* both field and WSS error
    (dev-log 2026-08-28) -- diluting bulk-flow coverage without adding
    resolution where it was actually needed.
    """
    xs = np.empty(n_points)
    ys = np.empty(n_points)
    filled = 0
    while filled < n_points:
        batch = (n_points - filled) * 3
        x_c = np.clip(rng.normal(params.x0, throat_sigma, size=batch), 0.0, params.Lx)
        bottom = rng.random(batch) < 0.5
        offset = rng.uniform(0.0, margin, size=batch)
        y_c = np.where(bottom, stenosis_height(x_c, params) + offset, params.H - offset)
        valid = is_fluid(x_c, y_c, params)
        x_ok, y_ok = x_c[valid], y_c[valid]
        take = min(len(x_ok), n_points - filled)
        xs[filled:filled + take] = x_ok[:take]
        ys[filled:filled + take] = y_ok[:take]
        filled += take
    return xs, ys


def _sample_collocation_points_t(
    params: StenosisParams, n_points: int, t_min: float, t_max: float, rng: np.random.Generator,
    near_wall_frac: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_wall = int(round(n_points * near_wall_frac))
    n_bulk = n_points - n_wall
    x_bulk, y_bulk = _sample_fluid_xy(params, n_bulk, rng)
    if n_wall > 0:
        x_wall, y_wall = _sample_near_wall_xy(params, n_wall, rng)
        xs, ys = np.concatenate([x_bulk, x_wall]), np.concatenate([y_bulk, y_wall])
    else:
        xs, ys = x_bulk, y_bulk
    ts = rng.uniform(t_min, t_max, size=n_points)
    return xs, ys, ts


def _to_col(array: np.ndarray) -> torch.Tensor:
    return torch.tensor(np.asarray(array, dtype=np.float64), dtype=torch.float32).reshape(-1, 1)


@dataclass
class TrainingHistoryPhase2:
    epoch: list[int] = field(default_factory=list)
    total: list[float] = field(default_factory=list)
    data: list[float] = field(default_factory=list)
    pde: list[float] = field(default_factory=list)
    bc: list[float] = field(default_factory=list)
    periodicity: list[float] = field(default_factory=list)


def _warm_start_from_phase1(model: PINN, phase1_checkpoint_path: str) -> None:
    ckpt = torch.load(phase1_checkpoint_path, weights_only=False)
    p1_state = ckpt["model_state"]
    state = model.state_dict()
    first_layer_key = "net.0.weight"
    if p1_state[first_layer_key].shape[0] != state[first_layer_key].shape[0]:
        raise ValueError(
            "Phase 1 and Phase 2 hidden_units must match to warm-start "
            f"(phase1={p1_state[first_layer_key].shape[0]}, phase2={state[first_layer_key].shape[0]})"
        )
    for key, value in p1_state.items():
        if key == first_layer_key:
            state[key][:, :2] = value  # copy the (x, y) columns; leave the new t-column at its random init
        else:
            state[key] = value
    model.load_state_dict(state)


def train_pinn_phase2(
    sensor_data,
    params: StenosisParams,
    U0: float,
    nu: float,
    period: float,
    amplitude_fundamental: float,
    amplitude_harmonic: float,
    record_start_t: float,
    cfg: dict,
    phase1_checkpoint_path: str | None = None,
) -> tuple[PINN, TrainingHistoryPhase2]:
    torch.manual_seed(cfg["seed"])
    rng = np.random.default_rng(cfg["seed"])

    # Fourier time features (dev-log 2026-08-28): give the network sin/cos
    # at the cardiac fundamental + as many harmonics as the waveform
    # actually has energy at (2, matching `cardiac_pulse`'s two-harmonic
    # construction), instead of a raw t it has to learn periodicity of
    # from scratch.
    n_harmonics = cfg.get("fourier_harmonics", 0)
    fourier_time_freqs = [k / period for k in range(1, n_harmonics + 1)] if n_harmonics else None

    model = PINN(
        in_dim=3, hidden_layers=cfg["hidden_layers"], hidden_units=cfg["hidden_units"],
        fourier_time_freqs=fourier_time_freqs,
    )
    if cfg.get("warm_start_from_phase1") and phase1_checkpoint_path:
        _warm_start_from_phase1(model, phase1_checkpoint_path)

    nu_t = torch.tensor(float(nu))
    t_min, t_max = record_start_t, record_start_t + period

    # -- fixed boundary/inlet points, replicated across several times ------
    n_bc_space = 60
    n_bc_time = 20
    x_bottom, y_bottom = wall_curve_points(params, n=n_bc_space)
    x_top = np.linspace(0, params.Lx, n_bc_space)
    y_top = np.full(n_bc_space, params.H)
    x_wall_space = np.concatenate([x_bottom, x_top])
    y_wall_space = np.concatenate([y_bottom, y_top])
    t_bc = np.linspace(t_min, t_max, n_bc_time, endpoint=False)

    x_wall = np.tile(x_wall_space, n_bc_time)
    y_wall = np.tile(y_wall_space, n_bc_time)
    t_wall = np.repeat(t_bc, len(x_wall_space))
    wall_t = (_to_col(x_wall), _to_col(y_wall), _to_col(t_wall))

    y_inlet_space = np.linspace(0, params.H, n_bc_space)
    base_u_inlet = inlet_profile(y_inlet_space, U0, params.H)
    x_inlet = np.tile(np.zeros(n_bc_space), n_bc_time)
    y_inlet = np.tile(y_inlet_space, n_bc_time)
    t_inlet = np.repeat(t_bc, n_bc_space)
    amp = cardiac_pulse(
        t_inlet, period=period, amplitude_fundamental=amplitude_fundamental, amplitude_harmonic=amplitude_harmonic,
    )
    u_inlet_profile = np.tile(base_u_inlet, n_bc_time) * amp
    v_inlet_profile = np.zeros_like(u_inlet_profile)
    inlet_t = (
        _to_col(x_inlet), _to_col(y_inlet), _to_col(t_inlet), _to_col(u_inlet_profile), _to_col(v_inlet_profile),
    )

    x_gauge, y_gauge = np.array([params.Lx * 0.98]), np.array([params.H * 0.5])
    t_gauge = np.full_like(x_gauge, t_min)
    gauge_t = (_to_col(x_gauge), _to_col(y_gauge), _to_col(t_gauge))

    # -- periodicity points: same (x, y), evaluated at t0 vs. t0+T ----------
    n_periodic = 800
    x_per, y_per = _sample_fluid_xy(params, n_periodic, rng)
    t0_col = _to_col(np.full(n_periodic, t_min))
    t1_col = _to_col(np.full(n_periodic, t_max))
    x_per_col, y_per_col = _to_col(x_per), _to_col(y_per)

    # -- sensor data: flatten (n_snapshots, n_points) to matching columns --
    n_snap, n_pts = sensor_data.u_meas.shape
    sensor_x = _to_col(np.broadcast_to(sensor_data.x[None, :], (n_snap, n_pts)))
    sensor_y = _to_col(np.broadcast_to(sensor_data.y[None, :], (n_snap, n_pts)))
    sensor_t = _to_col(np.broadcast_to(sensor_data.t[:, None], (n_snap, n_pts)))
    sensor_u = _to_col(sensor_data.u_meas)
    sensor_v = _to_col(sensor_data.v_meas)

    weights = cfg["loss_weights"]
    history = TrainingHistoryPhase2()

    near_wall_frac = cfg.get("near_wall_collocation_frac", 0.0)

    def sample_new_collocation():
        x_c, y_c, t_c = _sample_collocation_points_t(
            params, cfg["collocation_points"], t_min, t_max, rng, near_wall_frac=near_wall_frac,
        )
        return _to_col(x_c), _to_col(y_c), _to_col(t_c)

    colloc = sample_new_collocation()

    def compute_losses():
        terms = {
            "data": data_loss_t(model, sensor_x, sensor_y, sensor_t, sensor_u, sensor_v),
            "pde": pde_loss_t(model, *colloc, nu_t),
            "bc": (
                no_slip_loss_t(model, *wall_t)
                + inlet_loss_t(model, *inlet_t)
                + pressure_gauge_loss_t(model, *gauge_t)
            ),
            "periodicity": periodicity_loss(model, x_per_col, y_per_col, t0_col, t1_col),
        }
        return terms, total_loss(terms, weights)

    def record(epoch: int, terms: dict, loss: torch.Tensor) -> None:
        history.epoch.append(epoch)
        history.total.append(loss.item())
        history.data.append(terms["data"].item())
        history.pde.append(terms["pde"].item())
        history.bc.append(terms["bc"].item())
        history.periodicity.append(terms["periodicity"].item())

    # -- Adam phase ----------------------------------------------------------
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["adam_lr"])
    for epoch in range(cfg["adam_epochs"]):
        if epoch > 0 and epoch % cfg["resample_every"] == 0:
            colloc = sample_new_collocation()

        optimizer.zero_grad()
        terms, loss = compute_losses()
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0 or epoch == cfg["adam_epochs"] - 1:
            record(epoch, terms, loss)
        if epoch % 1000 == 0:
            print(
                f"  [adam] epoch={epoch:6d} total={loss.item():.4f} "
                f"data={terms['data'].item():.5f} pde={terms['pde'].item():.5f} "
                f"bc={terms['bc'].item():.5f} periodicity={terms['periodicity'].item():.5f}",
                flush=True,
            )

    # -- L-BFGS polishing phase -----------------------------------------------
    lbfgs = torch.optim.LBFGS(list(model.parameters()), max_iter=cfg["lbfgs_iters"], line_search_fn="strong_wolfe")

    def closure():
        lbfgs.zero_grad()
        _, loss = compute_losses()
        loss.backward()
        return loss

    lbfgs.step(closure)
    terms, loss = compute_losses()
    record(cfg["adam_epochs"] + cfg["lbfgs_iters"], terms, loss)

    return model, history
