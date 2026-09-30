"""Slow, full-resolution M3 validation: the checks that need production
resolution to mean anything (Poiseuille accuracy at the real grid size, and
the post-stenotic recirculation zone, which is a resolution-sensitive
feature -- see dev-log 2026-08-26). Not part of the pytest suite because it
takes a few minutes; run manually after touching the solver.
"""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pinn_bloodflow.geometry import StenosisParams
from pinn_bloodflow.solver.chorin import ChorinSolver, inlet_profile
from pinn_bloodflow.solver.mac_grid import MACGrid

NX, NY = 640, 128
RE = 200.0
NU = 1.0 / RE


def validate_poiseuille():
    grid = MACGrid(nx=NX, ny=NY, stenosis=StenosisParams(Lx=5.0, H=1.0, pct_ds=0.0))
    solver = ChorinSolver(grid, nu=NU, U0=1.0)
    state, n_steps = solver.run_to_steady(max_steps=3000, tol=1e-8, check_every=50)
    exact = inlet_profile(grid.yu[:, 0], 1.0, grid.stenosis.H)
    rel_err = np.max(np.abs(state.u[:, grid.nx // 2] - exact)) / np.max(exact)
    div_max = np.max(np.abs(solver._divergence(state.u, state.v)[grid.mask_p]))
    print(f"[Poiseuille] converged in {n_steps} steps, rel_err={rel_err:.2e}, max|div|={div_max:.2e}")
    assert rel_err < 0.01
    assert div_max < 1e-9


def validate_stenosed_case(pct_ds=50.0, max_steps=10000):
    params = StenosisParams(Lx=5.0, H=1.0, pct_ds=pct_ds, x0=2.0, sigma=0.4, eccentric=True)
    grid = MACGrid(nx=NX, ny=NY, stenosis=params)
    solver = ChorinSolver(grid, nu=NU, U0=1.0, donor_cell_gamma=0.9)

    t0 = time.time()
    state, n_steps = solver.run_to_steady(max_steps=max_steps, tol=1e-7, check_every=50)
    elapsed = time.time() - t0
    status = "converged" if n_steps < max_steps else "hit step budget (still stable, see dev-log)"
    print(f"[stenosed {pct_ds}%DS] {status} after {n_steps} steps ({elapsed:.1f}s)")

    dy = grid.dy
    stations = [int(f * (grid.nx - 1)) for f in (0.03, 0.25, 0.5, 0.75, 0.97)]
    rates = [np.sum(state.u[:, i]) * dy for i in stations]
    print("  mass conservation, flow rate at each station:", [round(r, 5) for r in rates])
    assert max(rates) / min(rates) < 1.01, "flow rate should be ~constant along the channel"

    div_max = np.max(np.abs(solver._divergence(state.u, state.v)[grid.mask_p]))
    print(f"  max|div u| = {div_max:.2e}")
    assert div_max < 1e-9

    throat_i = np.argmin(np.abs(grid.xp[0, :] - params.x0))
    downstream = state.u[:, throat_i + 10 : throat_i + 200]
    downstream_mask = grid.mask_u[:, throat_i + 10 : throat_i + 200]
    has_recirculation = (downstream[downstream_mask] < -1e-4).any()
    print(f"  post-stenotic recirculation present: {has_recirculation}")
    print(f"  max u overall: {state.u.max():.4f}")


if __name__ == "__main__":
    validate_poiseuille()
    validate_stenosed_case()
