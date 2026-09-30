"""Generate the Phase 2 (pulsatile) ground-truth dataset (M9/M10).

Runs the Chorin solver time-accurately through several cardiac cycles under
the asymmetric two-harmonic inlet waveform, discards the startup transient,
and saves dense snapshots from the final (converged, periodic) cycle --
interpolated onto cell centers, same layout convention as Phase 1's
`phase1_steady.npz` but with an extra leading time axis on every field.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pinn_bloodflow.geometry import StenosisParams  # noqa: E402
from pinn_bloodflow.solver.chorin import ChorinSolver  # noqa: E402
from pinn_bloodflow.solver.mac_grid import MACGrid  # noqa: E402
from pinn_bloodflow.solver.waveform import cardiac_pulse  # noqa: E402


def main(config_path: str, out_path: str) -> None:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    geo, grid_cfg, flow_cfg, pulse_cfg, solver_cfg = (
        cfg["geometry"], cfg["grid"], cfg["flow"], cfg["pulsatile"], cfg["solver"],
    )

    params = StenosisParams(
        H=geo["H"], Lx=geo["Lx"], x0=geo["stenosis_x0"],
        pct_ds=geo["stenosis_pct_ds"], sigma=geo["stenosis_sigma"], eccentric=geo["eccentric"],
    )
    grid = MACGrid(nx=grid_cfg["nx"], ny=grid_cfg["ny"], stenosis=params)
    nu = flow_cfg["U0"] * params.H / flow_cfg["reynolds"]

    period = pulse_cfg["period_T"]
    waveform = lambda t: cardiac_pulse(  # noqa: E731
        t, period=period,
        amplitude_fundamental=pulse_cfg["amplitude_fundamental"],
        amplitude_harmonic=pulse_cfg["amplitude_harmonic"],
    )
    solver = ChorinSolver(
        grid, nu=nu, U0=flow_cfg["U0"], cfl_safety=solver_cfg["cfl_safety"],
        inlet_amplitude_fn=waveform,
    )

    n_cycles = pulse_cfg["n_cycles"]
    record_from = pulse_cfg["record_from_cycle"]
    print(
        f"Running {grid.nx}x{grid.ny} grid, Re={flow_cfg['reynolds']}, {params.pct_ds}% DS, "
        f"{n_cycles} cycles (recording from cycle {record_from}) ..."
    )
    t0 = time.time()
    snapshots = solver.run_transient(
        period=period, n_cycles=n_cycles, snapshots_per_cycle=pulse_cfg["snapshots_per_cycle"],
        record_from_cycle=record_from,
    )
    elapsed = time.time() - t0
    print(f"Done: {len(snapshots)} snapshots in {elapsed:.1f}s")

    # Periodicity check: the first and last recorded snapshot are exactly
    # one period apart at the same phase -- if the flow has truly settled
    # into a repeating limit cycle, they should closely agree. Measured,
    # not assumed (dev-log 2026-08-28: a coarse/fast toy case showed this
    # can take several cycles at Re=200, likely nonlinear "steady
    # streaming" settling slowly).
    u_first, u_last = snapshots[0]["u"], snapshots[-1]["u"]
    scale = max(np.max(np.abs(u_last)), 1e-8)
    periodicity_residual = float(np.max(np.abs(u_last - u_first)) / scale)
    print(f"Periodicity residual (max|du| / max|u|, one period apart): {periodicity_residual:.2%}")

    # Interpolate the staggered u, v onto cell centers for every snapshot.
    t_arr = np.array([s["t"] for s in snapshots])
    u_center = np.stack([0.5 * (s["u"][:, :-1] + s["u"][:, 1:]) for s in snapshots])
    v_center = np.stack([0.5 * (s["v"][:-1, :] + s["v"][1:, :]) for s in snapshots])
    p_arr = np.stack([s["p"] for s in snapshots])

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        t=t_arr, xp=grid.xp, yp=grid.yp, mask_p=grid.mask_p,
        u=u_center, v=v_center, p=p_arr,
        nu=nu, U0=flow_cfg["U0"], H=params.H, Lx=params.Lx, x0=params.x0,
        sigma=params.sigma, pct_ds=params.pct_ds, reynolds=flow_cfg["reynolds"],
        period=period, periodicity_residual=periodicity_residual,
        amplitude_fundamental=pulse_cfg["amplitude_fundamental"],
        amplitude_harmonic=pulse_cfg["amplitude_harmonic"],
    )
    print("Saved to", out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase2_pulsatile.yaml")
    parser.add_argument("--out", default="data/ground_truth/phase2_pulsatile.npz")
    args = parser.parse_args()
    main(args.config, args.out)
