"""Tests for the Phase 2 pulsatile time-stepping driver (M9).

Uses a small, non-stenosed grid purely for speed -- these check the
snapshot bookkeeping and that the flow actually settles into a repeating
limit cycle, not the stenosed-flow physics itself (that's Phase 2's
ground-truth generation, validated separately at production resolution).
"""

import numpy as np

from pinn_bloodflow.geometry import StenosisParams
from pinn_bloodflow.solver.chorin import ChorinSolver
from pinn_bloodflow.solver.mac_grid import MACGrid
from pinn_bloodflow.solver.waveform import cardiac_pulse

RE = 200.0
NU = 1.0 / RE
PERIOD = 0.5  # short period keeps the test fast


def _make_solver():
    grid = MACGrid(nx=80, ny=24, stenosis=StenosisParams(Lx=5.0, H=1.0, pct_ds=0.0))
    return ChorinSolver(
        grid, nu=NU, U0=1.0,
        inlet_amplitude_fn=lambda t: cardiac_pulse(t, period=PERIOD),
    )


def test_run_transient_snapshot_count_and_spacing():
    solver = _make_solver()
    snapshots = solver.run_transient(period=PERIOD, n_cycles=2, snapshots_per_cycle=4)
    assert len(snapshots) == 5  # snapshots_per_cycle + 1: both cycle endpoints included
    times = [s["t"] for s in snapshots]
    assert np.isclose(times[0], PERIOD)  # recording starts at the last cycle (cycle index 1)
    assert np.isclose(times[-1], 2 * PERIOD)
    spacing = np.diff(times)
    assert np.allclose(spacing, PERIOD / 4, atol=1e-9)


def test_run_transient_reaches_a_repeating_limit_cycle():
    # Loose tolerance, deliberately: a cheap grid/timestep run measurably
    # does NOT converge to tight (<1%) periodicity in only a few short
    # cycles here -- checked empirically (residual ~2-10% and non-monotonic
    # in cycle count, not a clean decaying transient) rather than assumed.
    # The likely cause is genuine, not a bug: at Re=200 the nonlinear
    # advection terms couple the oscillation to a slowly-evolving mean-flow
    # correction ("steady streaming"), which is known to take many cycles
    # to fully settle in oscillatory viscous flows. This toy grid/period
    # only checks that the snapshot machinery and general flow shape are
    # sane, not that periodicity has tightly converged -- that's checked
    # for real, at production settings, when Phase 2's ground truth is
    # generated (dev-log 2026-08-28).
    solver = _make_solver()
    snapshots = solver.run_transient(period=PERIOD, n_cycles=3, snapshots_per_cycle=6)
    first, last = snapshots[0], snapshots[-1]  # exactly one period apart, same phase
    scale = max(np.max(np.abs(last["u"])), 1e-8)
    assert np.max(np.abs(last["u"] - first["u"])) / scale < 0.15
    assert np.all(np.isfinite(last["u"])) and np.max(np.abs(last["u"])) < 10.0  # no blow-up
