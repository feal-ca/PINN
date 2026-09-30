"""Sample sparse, noisy sensor data from a generated ground-truth field."""

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pinn_bloodflow.geometry import StenosisParams  # noqa: E402
from pinn_bloodflow.sensors.sampling import sample_sensors  # noqa: E402


def main(config_path: str, ground_truth_path: str, out_path: str) -> None:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    geo, sensors_cfg = cfg["geometry"], cfg["sensors"]

    params = StenosisParams(
        H=geo["H"], Lx=geo["Lx"], x0=geo["stenosis_x0"],
        pct_ds=geo["stenosis_pct_ds"], sigma=geo["stenosis_sigma"], eccentric=geo["eccentric"],
    )

    ground_truth = np.load(ground_truth_path)
    data = sample_sensors(
        ground_truth, params,
        n_points=sensors_cfg["n_points"],
        wall_margin_frac=sensors_cfg["wall_margin_frac"],
        noise_level=sensors_cfg["noise_level"],
        seed=sensors_cfg["seed"],
    )

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        x=data.x, y=data.y,
        u_true=data.u_true, v_true=data.v_true,
        u_meas=data.u_meas, v_meas=data.v_meas,
        noise_level=data.noise_level,
    )
    print(f"Saved {len(data.x)} sensor points to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase1_steady.yaml")
    parser.add_argument("--ground-truth", default="data/ground_truth/phase1_steady.npz")
    parser.add_argument("--out", default="data/sensors/phase1_sensors.npz")
    args = parser.parse_args()
    main(args.config, args.ground_truth, args.out)
