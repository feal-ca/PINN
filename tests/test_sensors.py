import numpy as np

from pinn_bloodflow.geometry import StenosisParams
from pinn_bloodflow.sensors.sampling import sample_sensors


def make_synthetic_ground_truth(params: StenosisParams, nx=200, ny=64):
    """A tiny fake 'ground truth' with a known analytic field (u=x, v=2y),
    so interpolation accuracy can be checked exactly, independent of
    whether the real solver's dataset has been generated yet.
    """
    x1d = np.linspace(0, params.Lx, nx)
    y1d = np.linspace(0, params.H, ny)
    xp, yp = np.meshgrid(x1d, y1d)
    return {"xp": xp, "yp": yp, "u": xp.copy(), "v": 2.0 * yp, "U0": 1.0}


def test_sampled_points_respect_wall_margin():
    params = StenosisParams(Lx=5.0, H=1.0, pct_ds=50.0, x0=2.0, sigma=0.4, eccentric=True)
    gt = make_synthetic_ground_truth(params)
    margin = 0.08
    data = sample_sensors(gt, params, n_points=50, wall_margin_frac=margin, noise_level=0.0, seed=0)

    from pinn_bloodflow.geometry import stenosis_height

    h = stenosis_height(data.x, params)
    assert np.all(data.y - h > margin - 1e-9)
    assert np.all(params.H - data.y > margin - 1e-9)


def test_no_pressure_sensors_and_correct_count():
    params = StenosisParams()
    gt = make_synthetic_ground_truth(params)
    data = sample_sensors(gt, params, n_points=25, wall_margin_frac=0.05, noise_level=0.0, seed=1)
    assert len(data.x) == 25
    assert not hasattr(data, "p_meas")  # pressure is never sensed, only inferred


def test_noise_free_interpolation_matches_known_field():
    params = StenosisParams(pct_ds=30.0)
    gt = make_synthetic_ground_truth(params)
    data = sample_sensors(gt, params, n_points=100, wall_margin_frac=0.05, noise_level=0.0, seed=2)
    # u=x, v=2y is linear, so bilinear interpolation should recover it almost exactly.
    assert np.allclose(data.u_true, data.x, atol=1e-8)
    assert np.allclose(data.v_true, 2.0 * data.y, atol=1e-8)
    assert np.array_equal(data.u_meas, data.u_true)  # zero noise -> measured == true


def test_noise_matches_requested_level():
    params = StenosisParams(pct_ds=0.0)  # plain channel, no margin complications
    gt = make_synthetic_ground_truth(params)
    noise_level = 0.1
    data = sample_sensors(gt, params, n_points=5000, wall_margin_frac=0.05, noise_level=noise_level, seed=3)
    residual_std = np.std(data.u_meas - data.u_true)
    assert abs(residual_std - noise_level * gt["U0"]) < 0.01
