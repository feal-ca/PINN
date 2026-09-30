"""Run a bounded slice of Phase 2 Adam training (or the final L-BFGS
polish), saving/loading a resumable state file between calls.

Exists only as a workaround for background training runs being killed
externally mid-flight (dev-log 2026-08-28): it lets the same training
recipe as `train_pinn_phase2` run as a sequence of short *foreground*
calls, each well under the tool's per-call timeout, instead of one long
backgrounded process. Not meant to be a permanent addition -- it
duplicates `train_pinn_phase2`'s inner loop rather than refactoring it to
be resumable, which would be the right fix if this needed to be a
long-term feature.

State saved between chunks is the model weights, the Adam optimizer state,
and training history -- NOT the numpy Generator's internal state, so the
exact sequence of resampled collocation points differs slightly across a
resume boundary. That's a cosmetic gap (collocation resampling is already
a stochastic regularizer, not something the physics depends on
reproducing exactly), not a correctness one.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pinn_bloodflow.geometry import StenosisParams  # noqa: E402
from pinn_bloodflow.pinn.losses import (  # noqa: E402
    data_loss_t, inlet_loss_t, no_slip_loss_t, pde_loss_t, periodicity_loss, pressure_gauge_loss_t, total_loss,
)
from pinn_bloodflow.pinn.network import PINN  # noqa: E402
from pinn_bloodflow.pinn.train_phase2 import (  # noqa: E402
    TrainingHistoryPhase2, _sample_collocation_points_t, _sample_fluid_xy, _to_col, _warm_start_from_phase1,
)
from pinn_bloodflow.sensors.sampling import SensorDataPulsatile  # noqa: E402
from pinn_bloodflow.solver.chorin import inlet_profile  # noqa: E402
from pinn_bloodflow.solver.waveform import cardiac_pulse  # noqa: E402
from pinn_bloodflow.geometry import wall_curve_points  # noqa: E402


def build_fixed_tensors(sensor_data, params, U0, period, amplitude_fundamental, amplitude_harmonic, record_start_t, rng):
    t_min, t_max = record_start_t, record_start_t + period
    n_bc_space, n_bc_time = 60, 20
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
    amp = cardiac_pulse(t_inlet, period=period, amplitude_fundamental=amplitude_fundamental, amplitude_harmonic=amplitude_harmonic)
    u_inlet_profile = np.tile(base_u_inlet, n_bc_time) * amp
    v_inlet_profile = np.zeros_like(u_inlet_profile)
    inlet_t = (_to_col(x_inlet), _to_col(y_inlet), _to_col(t_inlet), _to_col(u_inlet_profile), _to_col(v_inlet_profile))

    x_gauge, y_gauge = np.array([params.Lx * 0.98]), np.array([params.H * 0.5])
    t_gauge = np.full_like(x_gauge, t_min)
    gauge_t = (_to_col(x_gauge), _to_col(y_gauge), _to_col(t_gauge))

    n_periodic = 800
    x_per, y_per = _sample_fluid_xy(params, n_periodic, rng)
    t0_col = _to_col(np.full(n_periodic, t_min))
    t1_col = _to_col(np.full(n_periodic, t_max))
    periodic_t = (_to_col(x_per), _to_col(y_per), t0_col, t1_col)

    n_snap, n_pts = sensor_data.u_meas.shape
    sensor_x = _to_col(np.broadcast_to(sensor_data.x[None, :], (n_snap, n_pts)))
    sensor_y = _to_col(np.broadcast_to(sensor_data.y[None, :], (n_snap, n_pts)))
    sensor_t = _to_col(np.broadcast_to(sensor_data.t[:, None], (n_snap, n_pts)))
    sensor_u, sensor_v = _to_col(sensor_data.u_meas), _to_col(sensor_data.v_meas)
    sensor_t_tuple = (sensor_x, sensor_y, sensor_t, sensor_u, sensor_v)

    return wall_t, inlet_t, gauge_t, periodic_t, sensor_t_tuple, t_min, t_max


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/phase2_pulsatile.yaml")
    p.add_argument("--sensors", default="data/sensors/phase2_sensors.npz")
    p.add_argument("--phase1-checkpoint", default="data/checkpoints/phase1_pinn.pt")
    p.add_argument("--state-in", default=None)
    p.add_argument("--state-out", required=True)
    p.add_argument("--end-epoch", type=int, required=True, help="run Adam up to (exclusive) this epoch")
    p.add_argument("--final", action="store_true", help="after reaching adam_epochs, also run L-BFGS and finish")
    p.add_argument("--out", default="data/checkpoints/phase2_pinn.pt", help="final checkpoint path (--final only)")
    args = p.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    geo, flow_cfg, pulse_cfg, pinn_cfg = cfg["geometry"], cfg["flow"], cfg["pulsatile"], cfg["pinn"]
    params = StenosisParams(H=geo["H"], Lx=geo["Lx"], x0=geo["stenosis_x0"], pct_ds=geo["stenosis_pct_ds"], sigma=geo["stenosis_sigma"], eccentric=geo["eccentric"])

    raw = np.load(args.sensors)
    sensor_data = SensorDataPulsatile(x=raw["x"], y=raw["y"], t=raw["t"], u_true=raw["u_true"], v_true=raw["v_true"], u_meas=raw["u_meas"], v_meas=raw["v_meas"], noise_level=float(raw["noise_level"]))

    phase1_ckpt = torch.load(args.phase1_checkpoint, weights_only=False)
    nu = torch.exp(phase1_ckpt["log_nu"]).item()
    record_start_t = float(pulse_cfg["record_from_cycle"]) * pulse_cfg["period_T"]

    n_harmonics = pinn_cfg.get("fourier_harmonics", 0)
    period = pulse_cfg["period_T"]
    fourier_time_freqs = [k / period for k in range(1, n_harmonics + 1)] if n_harmonics else None

    torch.manual_seed(pinn_cfg["seed"])
    rng = np.random.default_rng(pinn_cfg["seed"])
    model = PINN(in_dim=3, hidden_layers=pinn_cfg["hidden_layers"], hidden_units=pinn_cfg["hidden_units"], fourier_time_freqs=fourier_time_freqs)
    if pinn_cfg.get("warm_start_from_phase1"):
        _warm_start_from_phase1(model, args.phase1_checkpoint)
    optimizer = torch.optim.Adam(model.parameters(), lr=pinn_cfg["adam_lr"])
    history = TrainingHistoryPhase2()
    start_epoch = 0

    if args.state_in and Path(args.state_in).exists():
        state = torch.load(args.state_in, weights_only=False)
        model.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        history = TrainingHistoryPhase2(**state["history"])
        start_epoch = state["next_epoch"]
        print(f"Resumed from {args.state_in} at epoch {start_epoch}")

    wall_t, inlet_t, gauge_t, periodic_t, sensor_t, t_min, t_max = build_fixed_tensors(
        sensor_data, params, flow_cfg["U0"], period, pulse_cfg["amplitude_fundamental"], pulse_cfg["amplitude_harmonic"], record_start_t, rng,
    )
    weights = pinn_cfg["loss_weights"]
    near_wall_frac = pinn_cfg.get("near_wall_collocation_frac", 0.0)

    def sample_new_collocation():
        x_c, y_c, t_c = _sample_collocation_points_t(params, pinn_cfg["collocation_points"], t_min, t_max, rng, near_wall_frac=near_wall_frac)
        return _to_col(x_c), _to_col(y_c), _to_col(t_c)

    colloc = sample_new_collocation()

    def compute_losses():
        terms = {
            "data": data_loss_t(model, *sensor_t),
            "pde": pde_loss_t(model, *colloc, torch.tensor(float(nu))),
            "bc": no_slip_loss_t(model, *wall_t) + inlet_loss_t(model, *inlet_t) + pressure_gauge_loss_t(model, *gauge_t),
            "periodicity": periodicity_loss(model, *periodic_t),
        }
        return terms, total_loss(terms, weights)

    end_epoch = min(args.end_epoch, pinn_cfg["adam_epochs"])
    for epoch in range(start_epoch, end_epoch):
        if epoch > 0 and epoch % pinn_cfg["resample_every"] == 0:
            colloc = sample_new_collocation()
        optimizer.zero_grad()
        terms, loss = compute_losses()
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0 or epoch == pinn_cfg["adam_epochs"] - 1:
            history.epoch.append(epoch)
            history.total.append(loss.item())
            history.data.append(terms["data"].item())
            history.pde.append(terms["pde"].item())
            history.bc.append(terms["bc"].item())
            history.periodicity.append(terms["periodicity"].item())
        if epoch % 1000 == 0:
            print(f"  [adam] epoch={epoch:6d} total={loss.item():.4f} data={terms['data'].item():.5f} pde={terms['pde'].item():.5f} bc={terms['bc'].item():.5f} periodicity={terms['periodicity'].item():.5f}", flush=True)

    print(f"Reached epoch {end_epoch}/{pinn_cfg['adam_epochs']}")

    if end_epoch >= pinn_cfg["adam_epochs"] and args.final:
        lbfgs = torch.optim.LBFGS(list(model.parameters()), max_iter=pinn_cfg["lbfgs_iters"], line_search_fn="strong_wolfe")

        def closure():
            lbfgs.zero_grad()
            _, loss = compute_losses()
            loss.backward()
            return loss

        lbfgs.step(closure)
        terms, loss = compute_losses()
        history.epoch.append(pinn_cfg["adam_epochs"] + pinn_cfg["lbfgs_iters"])
        history.total.append(loss.item())
        history.data.append(terms["data"].item())
        history.pde.append(terms["pde"].item())
        history.bc.append(terms["bc"].item())
        history.periodicity.append(terms["periodicity"].item())
        print(f"Final loss terms: data={terms['data'].item():.5f} pde={terms['pde'].item():.5f} bc={terms['bc'].item():.5f} periodicity={terms['periodicity'].item():.5f}")

        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state": model.state_dict(),
            "hidden_layers": pinn_cfg["hidden_layers"],
            "hidden_units": pinn_cfg["hidden_units"],
            "fourier_time_freqs": model.fourier_time_freqs,
            "history": vars(history),
            "nu": nu,
            "period": period,
            "record_start_t": record_start_t,
        }, out)
        print("Saved final checkpoint to", out)
    else:
        Path(args.state_out).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "history": vars(history),
            "next_epoch": end_epoch,
        }, args.state_out)
        print("Saved resumable state to", args.state_out)


if __name__ == "__main__":
    main()
