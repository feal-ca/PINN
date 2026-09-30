import numpy as np
import pytest

from pinn_bloodflow.geometry import StenosisParams
from pinn_bloodflow.solver.mac_grid import MACGrid
from pinn_bloodflow.solver.poisson import PoissonSolver, assemble_poisson_matrix, poisson_row


def make_grid(nx=40, ny=16, pct_ds=50.0):
    return MACGrid(nx=nx, ny=ny, stenosis=StenosisParams(Lx=5.0, H=1.0, pct_ds=pct_ds))


def test_interior_row_sums_to_zero():
    # A Laplacian stencil must annihilate a constant field: L(const) = 0.
    # That's only true if every neighbor is either a real fluid cell
    # (whose coefficient cancels the matching diagonal piece) or a Neumann
    # boundary (which contributes nothing at all).
    grid = make_grid()
    # Cell far upstream, mid-channel: guaranteed all 4 neighbors are fluid.
    coeffs = poisson_row(grid, j=grid.ny // 2, i=2)
    assert np.isclose(sum(coeffs.values()), 0.0)


def test_wall_adjacent_row_still_sums_to_zero():
    # Bottom row, upstream of the stenosis: south neighbor is the Neumann
    # domain wall (out of grid), which should also contribute nothing.
    grid = make_grid()
    coeffs = poisson_row(grid, j=0, i=2)
    assert np.isclose(sum(coeffs.values()), 0.0)


def test_outlet_adjacent_row_does_not_sum_to_zero():
    # Last column: east neighbor is the Dirichlet outlet (p=0), which
    # breaks constant-preservation on purpose -- a genuine boundary effect,
    # not a bug.
    grid = make_grid()
    j = grid.ny // 2
    i = grid.nx - 1
    coeffs = poisson_row(grid, j=j, i=i)
    expected = -2.0 / grid.dx ** 2
    assert np.isclose(sum(coeffs.values()), expected)


def test_solid_cell_gets_identity_row():
    grid = make_grid(pct_ds=80.0)
    solid_cells = np.argwhere(~grid.mask_p)
    assert len(solid_cells) > 0
    j, i = solid_cells[0]
    matrix = assemble_poisson_matrix(grid).tocsr()
    idx = j * grid.nx + i
    row = matrix.getrow(idx).toarray().ravel()
    assert row[idx] == 1.0
    assert np.count_nonzero(row) == 1


def test_matrix_is_nonsingular():
    grid = make_grid()
    solver = PoissonSolver(grid)  # raises if splu fails on a singular matrix
    assert solver is not None


def test_zero_rhs_gives_zero_pressure():
    # With homogeneous Neumann everywhere except a Dirichlet p=0 outlet,
    # zero forcing has exactly one solution: p=0 everywhere in the fluid.
    grid = make_grid()
    solver = PoissonSolver(grid)
    p = solver.solve(grid.zeros_p())
    assert np.allclose(p[grid.mask_p], 0.0, atol=1e-10)
