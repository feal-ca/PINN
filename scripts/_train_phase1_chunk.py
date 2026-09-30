"""Resumable, chunked Phase 1 training, plus the ablation switches.

Same recipe as `train_pinn`, split into short foreground calls because
long background runs get killed in this environment (dev-log 2026-08-28).
Adds two flags the library trainer doesn't have:

  --freeze-nu     hold viscosity at the true value instead of learning it
  --data-weight   override the data-loss weight (0 disables the sensors)

Together those give the zero-sensor ablation: if the field is already
determined by the exact inlet, the no-slip walls and the PDE residual,
then training with no sensor data at all should reproduce most of the
reported accuracy, and the sensors' real job is identifying viscosity.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pinn_bloodflow.geometry import StenosisParams, wall_curve_points  # noqa: E402
from pinn_bloodflow.pinn.losses import (  # noqa: E402
    data_loss, inlet_loss, no_slip_loss, pde_loss, pressure_gauge_loss, total_loss,
)
from pinn_bloodflow.pinn.network import PINN  # noqa: E402
from pinn_bloodflow.pinn.train import TrainingHistory, _sample_collocation_points, _to_col  # noqa: E402
from pinn_bloodflow.solver.chorin import inlet_profile  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/phase1_steady.yaml")
    p.add_argument("--sensors", default="data/sensors/phase1_sensors.npz")
    p.add_argument("--ground-truth", default="data/ground_truth/phase1_steady.npz")
    p.add_argument("--state-in", default=None)
    p.add_argument("--state-out", required=True)
    p.add_argument("--end-epoch", type=int, required=True)
    p.add_argument("--adam-epochs", type=int, default=None, help="override cfg adam_epochs")
    p.add_argument("--final", action="store_true", help="run L-BFGS and write the checkpoint")
    p.add_argument("--out", default="data/checkpoints/phase1_pinn.pt")
    p.add_argument("--freeze-nu", action="store_true")
    p.add_argument("--data-weight", type=float, default=None)
    p.add_argument("--save-every", type=int, default=2000,
                   help="write resumable state this often, so an interrupted run loses at most this many epochs")
    args = p.parse_args()

    cfg = yaml.safe_load(open(args.config))
    geo, flow, pinn_cfg = cfg["geometry"], cfg["flow"], cfg["pinn"]
    params = StenosisParams(
        H=geo["H"], Lx=geo["Lx"], x0=geo["stenosis_x0"],
        pct_ds=geo["stenosis_pct_ds"], sigma=geo["stenosis_sigma"], eccentric=geo["eccentric"],
    )
    adam_epochs = args.adam_epochs or pinn_cfg["adam_epochs"]
    U0 = flow["U0"]
    gt = np.load(args.ground_truth)
    nu_true = float(gt["nu"])

    weights = dict(pinn_cfg["loss_weights"])
    if args.data_weight is not None:
        weights["data"] = args.data_weight

    torch.manual_seed(pinn_cfg["seed"])
    rng = np.random.default_rng(pinn_cfg["seed"])
    model = PINN(in_dim=2, hidden_layers=pinn_cfg["hidden_layers"], hidden_units=pinn_cfg["hidden_units"])

    if args.freeze_nu:
        log_nu = torch.tensor(float(np.log(nu_true)))
        nu_params = []
    else:
        log_nu = torch.nn.Parameter(torch.tensor(float(np.log(4.0 * nu_true))))
        nu_params = [log_nu]

    n_bc = 100
    x_bottom, y_bottom = wall_curve_points(params, n=n_bc)
    x_wall = np.concatenate([x_bottom, np.linspace(0, params.Lx, n_bc)])
    y_wall = np.concatenate([y_bottom, np.full(n_bc, params.H)])
    y_inlet = np.linspace(0, params.H, n_bc)
    wall_t = (_to_col(x_wall), _to_col(y_wall))
    inlet_t = (_to_col(np.zeros(n_bc)), _to_col(y_inlet),
               _to_col(inlet_profile(y_inlet, U0, params.H)), _to_col(np.zeros(n_bc)))
    gauge_t = (_to_col(np.array([params.Lx * 0.98])), _to_col(np.array([params.H * 0.5])))

    s = np.load(args.sensors)
    sx, sy = _to_col(s["x"]), _to_col(s["y"])
    su, sv = _to_col(s["u_meas"]), _to_col(s["v_meas"])

    opt_groups = [{"params": model.parameters(), "lr": pinn_cfg["adam_lr"]}]
    if nu_params:
        opt_groups.append({"params": nu_params, "lr": pinn_cfg["adam_lr"] * pinn_cfg["nu_lr_scale"]})
    optimizer = torch.optim.Adam(opt_groups)
    history = TrainingHistory()
    start_epoch = 0

    if args.state_in and Path(args.state_in).exists():
        st = torch.load(args.state_in, weights_only=False)
        model.load_state_dict(st["model_state"])
        optimizer.load_state_dict(st["optimizer_state"])
        history = TrainingHistory(**st["history"])
        start_epoch = st["next_epoch"]
        if not args.freeze_nu:
            with torch.no_grad():
                log_nu.copy_(torch.tensor(st["log_nu"]))
        print(f"Resumed at epoch {start_epoch}")

    colloc = None

    def resample():
        xc, yc = _sample_collocation_points(params, pinn_cfg["collocation_points"], rng)
        return _to_col(xc), _to_col(yc)

    colloc = resample()

    def losses():
        nu = torch.exp(log_nu)
        terms = {
            "data": data_loss(model, sx, sy, su, sv),
            "pde": pde_loss(model, colloc[0], colloc[1], nu),
            "bc": (no_slip_loss(model, *wall_t)
                   + inlet_loss(model, *inlet_t)
                   + pressure_gauge_loss(model, *gauge_t)),
        }
        return terms, total_loss(terms, weights)

    def save_state(next_epoch: int) -> None:
        Path(args.state_out).parent.mkdir(parents=True, exist_ok=True)
        tmp = args.state_out + ".tmp"
        torch.save({
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "history": vars(history),
            "next_epoch": next_epoch,
            "log_nu": float(log_nu),
        }, tmp)
        Path(tmp).replace(args.state_out)  # atomic, so a kill mid-write can't corrupt it

    end = min(args.end_epoch, adam_epochs)
    for epoch in range(start_epoch, end):
        if epoch > 0 and epoch % pinn_cfg["resample_every"] == 0:
            colloc = resample()
        optimizer.zero_grad()
        terms, loss = losses()
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0 or epoch == adam_epochs - 1:
            history.epoch.append(epoch)
            history.total.append(loss.item())
            history.data.append(terms["data"].item())
            history.pde.append(terms["pde"].item())
            history.bc.append(terms["bc"].item())
            history.nu.append(float(torch.exp(log_nu)))
        if epoch > start_epoch and epoch % args.save_every == 0:
            save_state(epoch)
        if epoch % 2000 == 0:
            print(f"  [adam] {epoch:6d} total={loss.item():.4f} data={terms['data'].item():.2e} "
                  f"pde={terms['pde'].item():.2e} bc={terms['bc'].item():.2e} nu={float(torch.exp(log_nu)):.5f}",
                  flush=True)

    print(f"Reached epoch {end}/{adam_epochs}")

    if end >= adam_epochs and args.final:
        if not args.freeze_nu:
            log_nu.requires_grad_(False)
        lb = torch.optim.LBFGS(list(model.parameters()), max_iter=pinn_cfg["lbfgs_iters"],
                               line_search_fn="strong_wolfe")

        def closure():
            lb.zero_grad()
            _, l = losses()
            l.backward()
            return l

        lb.step(closure)
        terms, loss = losses()
        history.epoch.append(adam_epochs + pinn_cfg["lbfgs_iters"])
        history.total.append(loss.item())
        history.data.append(terms["data"].item())
        history.pde.append(terms["pde"].item())
        history.bc.append(terms["bc"].item())
        history.nu.append(float(torch.exp(log_nu)))
        print(f"Final: data={terms['data'].item():.3e} pde={terms['pde'].item():.3e} "
              f"bc={terms['bc'].item():.3e} nu={float(torch.exp(log_nu)):.5f}")
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state": model.state_dict(),
            "log_nu": torch.tensor(float(log_nu)),
            "hidden_layers": pinn_cfg["hidden_layers"],
            "hidden_units": pinn_cfg["hidden_units"],
            "history": vars(history),
            "nu_true": nu_true,
            "adam_epochs": adam_epochs,
            "frozen_nu": args.freeze_nu,
            "data_weight": weights["data"],
        }, out)
        print("Saved", out)
    else:
        save_state(end)
        print("Saved state to", args.state_out)


if __name__ == "__main__":
    main()
