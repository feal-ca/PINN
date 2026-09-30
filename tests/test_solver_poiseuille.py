"""Solver correctness tests (M3 validation).

These use small/coarse grids so the suite stays fast. Two things are
deliberately NOT covered here because they need a finer grid to show up
reliably (see `scripts/validate_solver.py` for the slower, full-resolution
version used to actually validate the Phase 1 dataset config):
  - the exact Poiseuille error target at production resolution
  - the post-stenotic recirculation zone, which only appears reliably once
    the stenosis is resolved by enough grid cells (see dev-log 2026-08-26)
"""

import numpy as np

from pinn_bloodflow.geometry import StenosisParams
from pinn_bloodflow.solver.chorin import ChorinSolver, inlet_profile
from pinn_bloodflow.solver.mac_grid import MACGrid

RE = 200.0
NU = 1.0 / RE


def test_nonstenosed_matches_analytical_poiseuille():
    grid = MACGrid(nx=160, ny=32, stenosis=StenosisParams(Lx=5.0, H=1.0, pct_ds=0.0))
    solver = ChorinSolver(grid, nu=NU, U0=1.0)
    state, n_steps = solver.run_to_steady(max_steps=2000, tol=1e-7, check_every=25)
    assert n_steps < 2000, "should reach steady state well within the step budget"

    exact = inlet_profile(grid.yu[:, 0], 1.0, grid.stenosis.H)
    mid = grid.nx // 2
    rel_err = np.max(np.abs(state.u[:, mid] - exact)) / np.max(np.abs(exact))
    assert rel_err < 0.01


def test_nonstenosed_is_divergence_free():
    grid = MACGrid(nx=160, ny=32, stenosis=StenosisParams(Lx=5.0, H=1.0, pct_ds=0.0))
    solver = ChorinSolver(grid, nu=NU, U0=1.0)
    state, _ = solver.run_to_steady(max_steps=2000, tol=1e-7, check_every=25)
    div = solver._divergence(state.u, state.v)
    assert np.max(np.abs(div[grid.mask_p])) < 1e-9


def _run_fixed_steps(grid, solver, n_steps):
    state = solver.initial_state()
    for _ in range(n_steps):
        dt = solver.stable_dt(state)
        state = solver.step(state, dt, t=0.0)
    return state


def test_stenosed_case_stays_divergence_free():
    # A mild stenosis at coarse resolution: the projection method should
    # keep every step divergence-free regardless of whether the flow has
    # fully reached steady state yet -- this checks the discretization's
    # structural correctness, not full convergence.
    params = StenosisParams(Lx=5.0, H=1.0, pct_ds=20.0, x0=2.0, sigma=0.4, eccentric=True)
    grid = MACGrid(nx=160, ny=32, stenosis=params)
    solver = ChorinSolver(grid, nu=NU, U0=1.0)
    state = _run_fixed_steps(grid, solver, n_steps=500)
    assert np.all(np.isfinite(state.u))
    div = solver._divergence(state.u, state.v)
    assert np.max(np.abs(div[grid.mask_p])) < 1e-9


def test_stenosed_case_conserves_mass():
    params = StenosisParams(Lx=5.0, H=1.0, pct_ds=20.0, x0=2.0, sigma=0.4, eccentric=True)
    grid = MACGrid(nx=160, ny=32, stenosis=params)
    solver = ChorinSolver(grid, nu=NU, U0=1.0)
    state = _run_fixed_steps(grid, solver, n_steps=500)

    dy = grid.dy
    flow_rates = [np.sum(state.u[:, i]) * dy for i in (5, 40, 80, 120, 155)]
    inlet_rate = flow_rates[0]
    for rate in flow_rates[1:]:
        assert abs(rate - inlet_rate) / inlet_rate < 0.01
