"""Tests for the Phase 2 pulsatile sensor sampler (M10)."""

import numpy as np

from pinn_bloodflow.geometry import StenosisParams
from pinn_bloodflow.sensors.sampling import sample_sensors_pulsatile


def make_synthetic_pulsatile_ground_truth(params: StenosisParams, nx=100, ny=40, n_snap=20):
    """A known analytic field u = x + t, v = 2y (t-independent), so
    trilinear interpolation accuracy can be checked exactly.
    """
    x1d = np.linspace(0, params.Lx, nx)
    y1d = np.linspace(0, params.H, ny)
    t1d = np.linspace(0.0, 1.0, n_snap, endpoint=False)
    xp, yp = np.meshgrid(x1d, y1d)
    u = xp[None, :, :] + t1d[:, None, None]
    v = np.broadcast_to(2.0 * yp[None, :, :], u.shape).copy()
    return {"xp": xp, "yp": yp, "t": t1d, "u": u, "v": v, "U0": 1.0}


def test_pulsatile_sensor_shapes():
    params = StenosisParams(Lx=5.0, H=1.0, pct_ds=50.0, x0=2.0, sigma=0.4, eccentric=True)
    gt = make_synthetic_pulsatile_ground_truth(params)
    data = sample_sensors_pulsatile(
        gt, params, n_points=10, wall_margin_frac=0.08, noise_level=0.0, n_snapshots=6, seed=0,
    )
    assert data.x.shape == (10,)
    assert data.t.shape == (6,)
    assert data.u_true.shape == (6, 10)  # (n_snapshots, n_points)
    assert data.v_true.shape == (6, 10)


def test_pulsatile_noise_free_interpolation_matches_known_field():
    params = StenosisParams(pct_ds=30.0)
    gt = make_synthetic_pulsatile_ground_truth(params)
    data = sample_sensors_pulsatile(
        gt, params, n_points=15, wall_margin_frac=0.05, noise_level=0.0, n_snapshots=8, seed=1,
    )
    expected_u = data.x[None, :] + data.t[:, None]
    expected_v = 2.0 * data.y[None, :] * np.ones_like(data.t)[:, None]
    assert np.allclose(data.u_true, expected_u, atol=1e-8)
    assert np.allclose(data.v_true, expected_v, atol=1e-8)
    assert np.array_equal(data.u_meas, data.u_true)


def test_throat_frac_concentrates_probes_near_the_stenosis():
    params = StenosisParams(Lx=5.0, H=1.0, pct_ds=50.0, x0=2.0, sigma=0.4, eccentric=True)
    gt = make_synthetic_pulsatile_ground_truth(params)
    data = sample_sensors_pulsatile(
        gt, params, n_points=20, wall_margin_frac=0.08, noise_level=0.0, n_snapshots=6, seed=0,
        throat_frac=0.3, throat_half_width=0.6,
    )
    assert len(data.x) == 20
    near_throat = np.abs(data.x - params.x0) < 0.6
    # 6 of 20 (throat_frac=0.3) are drawn only from the throat band, so at
    # least that many must land there even though the remaining 14 could
    # also land there by chance.
    assert near_throat.sum() >= 6


def test_pulsatile_snapshot_times_stay_within_one_period_no_duplicate_endpoint():
    params = StenosisParams(pct_ds=0.0)
    gt = make_synthetic_pulsatile_ground_truth(params)
    data = sample_sensors_pulsatile(
        gt, params, n_points=5, wall_margin_frac=0.05, noise_level=0.0, n_snapshots=10, seed=2,
    )
    assert data.t.min() >= 0.0
    assert data.t.max() < 1.0  # endpoint excluded -- same phase as t=0, not new information
    assert len(np.unique(data.t)) == 10
