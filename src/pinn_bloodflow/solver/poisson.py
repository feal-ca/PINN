"""Pressure Poisson solve for the Chorin projection method.

The pressure Poisson matrix is assembled once (the stenosis geometry is
static across the whole simulation, steady or pulsatile) and factorized
with a sparse LU decomposition, then reused every timestep -- this is the
single biggest performance lever available on a CPU-only machine.

Boundary conditions:
  - Inlet (x=0) and channel walls: homogeneous Neumann (dp/dn = 0).
  - Outlet (x=Lx): Dirichlet p=0 -- this is what removes the null space
    that an all-Neumann Poisson problem would otherwise have (pressure
    would only be defined up to an arbitrary constant).
  - Stenosis surface (immersed boundary): homogeneous Neumann, same as a
    domain wall -- no fluid crosses a solid face, so there's no pressure
    flux through it either.
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from pinn_bloodflow.solver.mac_grid import MACGrid


def poisson_row(grid: MACGrid, j: int, i: int) -> dict[tuple[int, int], float]:
    """5-point Laplacian stencil coefficients for pressure cell (j, i).

    For each of the four neighbors (j-1,i), (j+1,i), (j,i-1), (j,i+1), you
    must decide which of three cases applies and handle it:

      1. Neighbor is in-domain AND fluid (grid.mask_p[nj, ni] is True):
         standard interior coefficient, 1/dx**2 for x-neighbors or
         1/dy**2 for y-neighbors.

      2. Neighbor is out-of-domain past a channel wall or past the x=0
         inlet, OR is in-domain but solid (blocked by the stenosis, i.e.
         grid.mask_p[nj, ni] is False): this is a zero-flux (Neumann)
         wall -- no fluid crosses this face. Standard treatment is a
         mirror ghost cell, p_ghost = p_(j,i), which cancels that term's
         contribution to the flux (it still affects the diagonal).

      3. Neighbor is out-of-domain past the RIGHT edge (i == grid.nx,
         the outlet): Dirichlet p=0 there. The outlet boundary sits half
         a cell past the last cell center, so the standard ghost-cell
         treatment is p_ghost = -p_(j,i) (linear extrapolation to zero
         at the face).

    Return a dict {(nj, ni): coefficient} containing an entry for every
    neighbor that is a real in-domain fluid cell (case 1), PLUS an entry
    for (j, i) itself (the diagonal) that folds in whatever cases 2/3
    contribute. Do not add dict entries for out-of-domain or solid
    neighbors directly -- their effect is folded into the diagonal.

    Built face by face from the finite-volume view: the Laplacian at (j, i)
    is the sum of face fluxes (p_neighbor - p_self) / h, divided by h again.
    A Neumann (no-flux) face contributes a flux of exactly zero, so it adds
    nothing at all -- not even to the diagonal. A Dirichlet face's ghost
    value is p_ghost = -p_self (linear extrapolation to zero at the face,
    which sits half a cell beyond the last real cell), so its flux is
    -2*p_self/h, contributing -2/h**2 to the diagonal and no off-diagonal
    term, since it only ever refers back to p_self.
    """
    dx2, dy2 = grid.dx ** 2, grid.dy ** 2
    coeffs: dict[tuple[int, int], float] = {}
    diag = 0.0

    def add_fluid_neighbor(nj: int, ni: int, h2: float) -> None:
        nonlocal diag
        diag -= 1.0 / h2
        coeffs[(nj, ni)] = coeffs.get((nj, ni), 0.0) + 1.0 / h2

    # West (inlet side, x=0): Neumann if out of domain -> zero flux, skip.
    ni, nj = i - 1, j
    if ni >= 0 and grid.mask_p[nj, ni]:
        add_fluid_neighbor(nj, ni, dx2)
    # else: Neumann wall, contributes nothing.

    # East (outlet side, x=Lx): Dirichlet p=0 if out of domain.
    ni, nj = i + 1, j
    if ni >= grid.nx:
        diag += -2.0 / dx2
    elif grid.mask_p[nj, ni]:
        add_fluid_neighbor(nj, ni, dx2)
    # else: solid (stenosis), Neumann, contributes nothing.

    # South (bottom channel wall): Neumann if out of domain.
    ni, nj = i, j - 1
    if nj >= 0 and grid.mask_p[nj, ni]:
        add_fluid_neighbor(nj, ni, dy2)

    # North (top channel wall): Neumann if out of domain.
    ni, nj = i, j + 1
    if nj < grid.ny and grid.mask_p[nj, ni]:
        add_fluid_neighbor(nj, ni, dy2)

    coeffs[(j, i)] = coeffs.get((j, i), 0.0) + diag
    return coeffs


def assemble_poisson_matrix(grid: MACGrid) -> sp.csc_matrix:
    n = grid.nx * grid.ny

    def idx(j: int, i: int) -> int:
        return j * grid.nx + i

    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []

    for j in range(grid.ny):
        for i in range(grid.nx):
            if not grid.mask_p[j, i]:
                # Solid cell: decouple it with an identity row. Its pressure
                # value is unused/meaningless, but keeping the matrix full
                # size (rather than remapping to a reduced fluid-only index
                # set) keeps the indexing simple.
                rows.append(idx(j, i))
                cols.append(idx(j, i))
                vals.append(1.0)
                continue

            for (nj, ni), coeff in poisson_row(grid, j, i).items():
                rows.append(idx(j, i))
                cols.append(idx(nj, ni))
                vals.append(coeff)

    return sp.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsc()


class PoissonSolver:
    """Prefactorized pressure Poisson solver, reused every timestep."""

    def __init__(self, grid: MACGrid):
        self.grid = grid
        matrix = assemble_poisson_matrix(grid)
        self._lu = spla.splu(matrix)

    def solve(self, rhs: np.ndarray) -> np.ndarray:
        """rhs has shape (ny, nx); solid-cell entries are ignored (identity rows)."""
        b = rhs.reshape(-1).copy()
        b[~self.grid.mask_p.reshape(-1)] = 0.0
        p = self._lu.solve(b)
        return p.reshape(self.grid.ny, self.grid.nx)
