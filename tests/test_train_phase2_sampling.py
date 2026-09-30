"""Unit tests for Phase 2's near-wall collocation sampling (dev-log
2026-08-28's WSS fix) -- these are cheap, so they get a real test unlike
the full training loop, which is only exercised through the scripts.
"""

import numpy as np

from pinn_bloodflow.geometry import StenosisParams, is_fluid, stenosis_height
from pinn_bloodflow.pinn.train_phase2 import _sample_collocation_points_t, _sample_near_wall_xy

PARAMS = StenosisParams(H=1.0, Lx=5.0, x0=2.0, pct_ds=50, sigma=0.4, eccentric=True)


def test_near_wall_points_are_fluid_and_close_to_a_wall():
    rng = np.random.default_rng(0)
    margin = 0.05
    x, y = _sample_near_wall_xy(PARAMS, 300, rng, margin=margin)

    assert is_fluid(x, y, PARAMS).all()

    dist_bottom = y - stenosis_height(x, PARAMS)
    dist_top = PARAMS.H - y
    close_to_a_wall = (dist_bottom < margin + 1e-9) | (dist_top < margin + 1e-9)
    assert close_to_a_wall.all()


def test_near_wall_fraction_increases_wall_density():
    rng = np.random.default_rng(0)
    margin = 0.05

    def wall_fraction(near_wall_frac):
        x, y, _ = _sample_collocation_points_t(PARAMS, 1000, 0.0, 1.0, rng, near_wall_frac=near_wall_frac)
        dist_bottom = y - stenosis_height(x, PARAMS)
        dist_top = PARAMS.H - y
        return ((dist_bottom < margin) | (dist_top < margin)).mean()

    assert wall_fraction(0.3) > wall_fraction(0.0)
