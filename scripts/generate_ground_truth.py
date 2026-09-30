"""Generate the Phase 1 (steady-state) ground-truth dataset.

Runs the validated Chorin solver to steady state and saves the resulting
field, interpolated onto the pressure grid (cell centers) for easy
downstream use by the sensor-sampling and evaluation code -- everything
becomes one consistent (x, y) -> (u, v, p) lookup instead of three
different staggered layouts.
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


def main(config_path: str, out_path: str) -> None:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    geo, grid_cfg, flow_cfg, solver_cfg = cfg["geometry"], cfg["grid"], cfg["flow"], cfg["solver"]

    params = StenosisParams(
        H=geo["H"], Lx=geo["Lx"], x0=geo["stenosis_x0"],
        pct_ds=geo["stenosis_pct_ds"], sigma=geo["stenosis_sigma"], eccentric=geo["eccentric"],
    )
    grid = MACGrid(nx=grid_cfg["nx"], ny=grid_cfg["ny"], stenosis=params)
    nu = flow_cfg["U0"] * params.H / flow_cfg["reynolds"]
    solver = ChorinSolver(grid, nu=nu, U0=flow_cfg["U0"], cfl_safety=solver_cfg["cfl_safety"])

    print(f"Running {grid.nx}x{grid.ny} grid, Re={flow_cfg['reynolds']}, {params.pct_ds}% DS ...")
    t0 = time.time()
    state, n_steps = solver.run_to_steady(
        max_steps=solver_cfg["max_steps"], tol=solver_cfg["convergence_tol"], check_every=50
    )
    elapsed = time.time() - t0
    status = "converged" if n_steps < solver_cfg["max_steps"] else "hit step budget (see dev-log)"
    print(f"{status} after {n_steps} steps ({elapsed:.1f}s)")

    # Interpolate the staggered u, v onto cell centers -- one consistent
    # (x, y) grid matching where p already lives.
    u_center = 0.5 * (state.u[:, :-1] + state.u[:, 1:])
    v_center = 0.5 * (state.v[:-1, :] + state.v[1:, :])

    div = solver._divergence(state.u, state.v)
    print(f"max|div u| = {np.max(np.abs(div[grid.mask_p])):.2e}")

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        xp=grid.xp, yp=grid.yp, mask_p=grid.mask_p,
        u=u_center, v=v_center, p=state.p,
        u_staggered=state.u, v_staggered=state.v,
        xu=grid.xu, yu=grid.yu, xv=grid.xv, yv=grid.yv,
        nu=nu, U0=flow_cfg["U0"], H=params.H, Lx=params.Lx, x0=params.x0,
        sigma=params.sigma, pct_ds=params.pct_ds, reynolds=flow_cfg["reynolds"],
        n_steps=n_steps,
    )
    print("Saved to", out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase1_steady.yaml")
    parser.add_argument("--out", default="data/ground_truth/phase1_steady.npz")
    args = parser.parse_args()
    main(args.config, args.out)
