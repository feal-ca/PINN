"""Train one PINN variant and score it against the CFD reference.

    python scripts/experiment.py --name wide --set hidden_units=64 --set adam_epochs=20000

Results land in data/experiments/<name>/: the checkpoint, metrics.json and
train.log. Config overrides go through --set key=value (pinn section) and
--sensors picks the probe layout: "random:20" or "zones:0-3:6,3-8:14".
"""

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pinn_bloodflow.evaluation import evaluate  # noqa: E402
from pinn_bloodflow.geometry import StenosisParams  # noqa: E402
from pinn_bloodflow.pinn.train import train_pinn  # noqa: E402
from pinn_bloodflow.sensors.sampling import sample_sensors, sample_sensors_zoned  # noqa: E402


def parse_value(text: str):
    try:
        return yaml.safe_load(text)
    except Exception:
        return text


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--config", default="configs/phase1_long.yaml")
    ap.add_argument("--gt", default="data/ground_truth/phase1_long.npz")
    ap.add_argument("--sensors", default="zones:0-2:6,2-4:14,4-6.5:14,6.5-9:6")
    ap.add_argument("--sensor-seed", type=int, default=None)
    ap.add_argument("--set", action="append", default=[], dest="overrides")
    ap.add_argument("--out-dir", default="data/experiments")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    pinn_cfg = copy.deepcopy(cfg["pinn"])
    for item in args.overrides:
        k, v = item.split("=", 1)
        pinn_cfg[k] = parse_value(v)

    out = Path(args.out_dir) / args.name
    out.mkdir(parents=True, exist_ok=True)
    pinn_cfg.setdefault("checkpoint_path", str(out / "partial.pt"))

    geo = cfg["geometry"]
    params = StenosisParams(H=geo["H"], Lx=geo["Lx"], x0=geo["stenosis_x0"], pct_ds=geo["stenosis_pct_ds"],
                            sigma=geo["stenosis_sigma"], eccentric=geo["eccentric"])
    gt = np.load(args.gt)
    nu_true = float(gt["nu"])
    scfg = cfg["sensors"]
    seed = scfg["seed"] if args.sensor_seed is None else args.sensor_seed
    kind, _, spec = args.sensors.partition(":")
    if kind == "random":
        sensors = sample_sensors(gt, params, int(spec), scfg["wall_margin_frac"], scfg["noise_level"], seed)
    else:
        zones = []
        for z in spec.split(","):
            rng_, n = z.rsplit(":", 1)
            lo, hi = rng_.split("-")
            zones.append((float(lo), float(hi), int(n)))
        sensors = sample_sensors_zoned(gt, params, zones, scfg["wall_margin_frac"], scfg["noise_level"], seed)

    print(f"[{args.name}] {len(sensors.x)} sensors, cfg={json.dumps(pinn_cfg)}", flush=True)
    t0 = time.time()
    model, log_nu, history = train_pinn(sensors, params, U0=cfg["flow"]["U0"], nu_true_hint=nu_true, cfg=pinn_cfg)
    minutes = (time.time() - t0) / 60
    model.eval()
    metrics, _ = evaluate(model, log_nu.item(), gt, params)
    result = {"name": args.name, "minutes": round(minutes, 1), "sensors": args.sensors, "sensor_seed": seed,
              "cfg": pinn_cfg, "metrics": vars(metrics)}
    (out / "metrics.json").write_text(json.dumps(result, indent=1, default=str))
    torch.save({"model_state": model.state_dict(), "log_nu": log_nu.detach(), "history": vars(history),
                "cfg": pinn_cfg, "nu_true": nu_true, "sensors": {"x": sensors.x, "y": sensors.y,
                "u_meas": sensors.u_meas, "v_meas": sensors.v_meas}}, out / "model.pt")
    m = metrics
    print(f"[{args.name}] {minutes:.1f} min | field {m.field_mag:.3f} vort {m.vorticity:.3f} wake-vort {m.vorticity_wake:.3f} "
          f"wss {m.wss:.3f} nu_err {m.nu_error:.3f} (nu={m.nu_recovered:.5f})", flush=True)


if __name__ == "__main__":
    main()
