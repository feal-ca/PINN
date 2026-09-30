"""Train the Phase 2 (pulsatile) PINN on the generated sensor data (M12)."""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pinn_bloodflow.geometry import StenosisParams  # noqa: E402
from pinn_bloodflow.pinn.train_phase2 import train_pinn_phase2  # noqa: E402
from pinn_bloodflow.sensors.sampling import SensorDataPulsatile  # noqa: E402


def main(
    config_path: str, ground_truth_path: str, sensors_path: str, phase1_checkpoint_path: str, out_path: str,
) -> None:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    geo, flow_cfg, pulse_cfg, pinn_cfg = cfg["geometry"], cfg["flow"], cfg["pulsatile"], cfg["pinn"]

    params = StenosisParams(
        H=geo["H"], Lx=geo["Lx"], x0=geo["stenosis_x0"],
        pct_ds=geo["stenosis_pct_ds"], sigma=geo["stenosis_sigma"], eccentric=geo["eccentric"],
    )

    gt = np.load(ground_truth_path)
    raw = np.load(sensors_path)
    sensor_data = SensorDataPulsatile(
        x=raw["x"], y=raw["y"], t=raw["t"], u_true=raw["u_true"], v_true=raw["v_true"],
        u_meas=raw["u_meas"], v_meas=raw["v_meas"], noise_level=float(raw["noise_level"]),
    )

    # Reuse Phase 1's recovered viscosity rather than re-inferring it here
    # (see train_phase2.py's module docstring for why).
    phase1_ckpt = torch.load(phase1_checkpoint_path, weights_only=False)
    nu = torch.exp(phase1_ckpt["log_nu"]).item()
    record_start_t = float(gt["t"][0])

    print(
        f"Training on {len(sensor_data.x)} probes x {len(sensor_data.t)} snapshots, "
        f"{pinn_cfg['hidden_layers']}x{pinn_cfg['hidden_units']} network, "
        f"{pinn_cfg['adam_epochs']} Adam epochs + {pinn_cfg['lbfgs_iters']} L-BFGS iters, "
        f"nu={nu:.5f} (from Phase 1) ..."
    )

    model, history = train_pinn_phase2(
        sensor_data, params, U0=flow_cfg["U0"], nu=nu,
        period=pulse_cfg["period_T"],
        amplitude_fundamental=pulse_cfg["amplitude_fundamental"],
        amplitude_harmonic=pulse_cfg["amplitude_harmonic"],
        record_start_t=record_start_t,
        cfg=pinn_cfg,
        phase1_checkpoint_path=phase1_checkpoint_path if pinn_cfg.get("warm_start_from_phase1") else None,
    )

    print(
        f"Final loss terms: data={history.data[-1]:.5f} pde={history.pde[-1]:.5f} "
        f"bc={history.bc[-1]:.5f} periodicity={history.periodicity[-1]:.5f}"
    )

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "hidden_layers": pinn_cfg["hidden_layers"],
            "hidden_units": pinn_cfg["hidden_units"],
            "fourier_time_freqs": model.fourier_time_freqs,
            "history": vars(history),
            "nu": nu,
            "period": pulse_cfg["period_T"],
            "record_start_t": record_start_t,
        },
        out,
    )
    print("Saved checkpoint to", out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase2_pulsatile.yaml")
    parser.add_argument("--ground-truth", default="data/ground_truth/phase2_pulsatile.npz")
    parser.add_argument("--sensors", default="data/sensors/phase2_sensors.npz")
    parser.add_argument("--phase1-checkpoint", default="data/checkpoints/phase1_pinn.pt")
    parser.add_argument("--out", default="data/checkpoints/phase2_pinn.pt")
    args = parser.parse_args()
    main(args.config, args.ground_truth, args.sensors, args.phase1_checkpoint, args.out)
