import numpy as np

from pinn_bloodflow.geometry import StenosisParams
from pinn_bloodflow.solver.mac_grid import MACGrid


def make_grid(nx=320, ny=64):
    return MACGrid(nx=nx, ny=ny, stenosis=StenosisParams(Lx=5.0, H=1.0))


def test_shapes():
    grid = make_grid()
    assert grid.xp.shape == (64, 320)
    assert grid.xu.shape == (64, 321)
    assert grid.xv.shape == (65, 320)
    assert grid.mask_p.shape == grid.xp.shape
    assert grid.mask_u.shape == grid.xu.shape
    assert grid.mask_v.shape == grid.xv.shape


def test_square_cells_for_default_config():
    grid = make_grid()
    assert np.isclose(grid.dx, grid.dy)


def test_u_faces_span_full_domain_length():
    grid = make_grid()
    assert np.isclose(grid.xu[0, 0], 0.0)
    assert np.isclose(grid.xu[0, -1], grid.stenosis.Lx)


def test_v_faces_span_full_channel_height():
    grid = make_grid()
    assert np.isclose(grid.yv[0, 0], 0.0)
    assert np.isclose(grid.yv[-1, 0], grid.stenosis.H)
