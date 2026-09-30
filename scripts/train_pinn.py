"""Train the Phase 1 PINN on the generated sensor data."""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pinn_bloodflow.geometry import StenosisParams  # noqa: E402
from pinn_bloodflow.pinn.train import train_pinn  # noqa: E402
from pinn_bloodflow.sensors.sampling import SensorData  # noqa: E402


def main(config_path: str, sensors_path: str, out_path: str) -> None:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    geo, flow_cfg, pinn_cfg = cfg["geometry"], cfg["flow"], cfg["pinn"]

    params = StenosisParams(
        H=geo["H"], Lx=geo["Lx"], x0=geo["stenosis_x0"],
        pct_ds=geo["stenosis_pct_ds"], sigma=geo["stenosis_sigma"], eccentric=geo["eccentric"],
    )
    nu_true = flow_cfg["U0"] * params.H / flow_cfg["reynolds"]

    raw = np.load(sensors_path)
    sensor_data = SensorData(
        x=raw["x"], y=raw["y"], u_true=raw["u_true"], v_true=raw["v_true"],
        u_meas=raw["u_meas"], v_meas=raw["v_meas"], noise_level=float(raw["noise_level"]),
    )

    print(f"Training on {len(sensor_data.x)} sensor points, "
          f"{pinn_cfg['hidden_layers']}x{pinn_cfg['hidden_units']} network, "
          f"{pinn_cfg['adam_epochs']} Adam epochs + {pinn_cfg['lbfgs_iters']} L-BFGS iters ...")

    model, log_nu, history = train_pinn(sensor_data, params, U0=flow_cfg["U0"], nu_true_hint=nu_true, cfg=pinn_cfg)

    nu_final = torch.exp(log_nu).item()
    print(f"Final loss terms: data={history.data[-1]:.5f} pde={history.pde[-1]:.5f} bc={history.bc[-1]:.5f}")
    print(f"nu: true={nu_true:.5f}  recovered={nu_final:.5f}  rel_error={abs(nu_final-nu_true)/nu_true:.2%}")

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "log_nu": log_nu.detach(),
            "hidden_layers": pinn_cfg["hidden_layers"],
            "hidden_units": pinn_cfg["hidden_units"],
            "history": vars(history),
            "nu_true": nu_true,
        },
        out,
    )
    print("Saved checkpoint to", out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase1_steady.yaml")
    parser.add_argument("--sensors", default="data/sensors/phase1_sensors.npz")
    parser.add_argument("--out", default="data/checkpoints/phase1_pinn.pt")
    args = parser.parse_args()
    main(args.config, args.sensors, args.out)
