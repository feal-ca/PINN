import numpy as np

from pinn_bloodflow.evaluation import relative_l2, vorticity, wss_gradient_reference
from pinn_bloodflow.geometry import StenosisParams


def _grid(nx=200, ny=100, lx=5.0, h=1.0):
    x = (np.arange(nx) + 0.5) * lx / nx
    y = (np.arange(ny) + 0.5) * h / ny
    return np.meshgrid(x, y)


def test_vorticity_of_solid_body_rotation_is_twice_the_rate():
    xp, yp = _grid()
    omega = 1.5
    u, v = -omega * yp, omega * xp
    w = vorticity(u, v, xp, yp, np.ones_like(xp, dtype=bool))
    assert np.allclose(w[5:-5, 5:-5], 2 * omega, atol=1e-6)


def test_wss_reference_recovers_slope_of_linear_shear_over_flat_wall():
    xp, yp = _grid()
    slope = 3.0
    gt = {"xp": xp, "yp": yp, "u": slope * yp, "v": np.zeros_like(yp)}
    flat = StenosisParams(pct_ds=0.0)
    _, grad = wss_gradient_reference(gt, flat)
    assert np.allclose(grad, slope, rtol=1e-3)


def test_relative_l2_ignores_masked_cells():
    a = np.ones((4, 4))
    b = np.ones((4, 4))
    b[0, 0] = 100.0
    mask = np.ones((4, 4), dtype=bool)
    mask[0, 0] = False
    assert relative_l2(b, a, mask) == 0.0
