"""Stenosis geometry: the analytic wall shape, plus fluid/solid masks.

The same analytic curve is used two ways in this project: the CFD solver
rasterizes it into a staircase mask on a grid (see `solver/`), while the
PINN samples the exact continuous curve directly for its no-slip boundary
loss. Keeping the shape definition here in one place keeps those two
representations consistent.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class StenosisParams:
    H: float = 1.0          # channel height (reference length)
    Lx: float = 5.0         # domain length, in units of H
    x0: float = 2.0         # stenosis center, in units of H
    pct_ds: float = 50.0    # percent diameter stenosis (clinical severity)
    sigma: float = 0.4      # Gaussian bump width, in units of H
    eccentric: bool = True  # one-sided bump (from bottom wall) vs. symmetric


def stenosis_height(x: np.ndarray, params: StenosisParams) -> np.ndarray:
    """Gaussian bump intrusion height h(x), measured from the wall inward.

    Max height at the throat is A = (pct_ds / 100) * H, so pct_ds is exactly
    the clinical percent-diameter-stenosis metric.
    """
    A = (params.pct_ds / 100.0) * params.H
    return A * np.exp(-((x - params.x0) ** 2) / (2.0 * params.sigma ** 2))


def is_fluid(x: np.ndarray, y: np.ndarray, params: StenosisParams) -> np.ndarray:
    """Boolean fluid mask at arbitrary (x, y) points (broadcastable arrays).

    Eccentric: plaque protrudes from the bottom wall (y=0) only, so fluid is
    y > h(x). Symmetric: both walls advance by h(x)/2, keeping the total
    blockage height comparable to the eccentric case at the same pct_ds.
    """
    h = stenosis_height(x, params)
    if params.eccentric:
        return y > h
    return (y > h / 2.0) & (y < params.H - h / 2.0)


def build_fluid_mask(x_coords: np.ndarray, y_coords: np.ndarray, params: StenosisParams) -> np.ndarray:
    """Fluid mask on a grid, given 1D x and y coordinate arrays.

    Returns a boolean array of shape (len(y_coords), len(x_coords)).
    """
    xx, yy = np.meshgrid(x_coords, y_coords)
    return is_fluid(xx, yy, params)


def wall_curve_points(params: StenosisParams, n: int = 200) -> tuple[np.ndarray, np.ndarray]:
    """Sample points along the analytic stenosis wall curve (bottom wall).

    Used by the PINN's no-slip boundary loss, which samples the *exact*
    smooth curve rather than the solver's staircase approximation of it.
    For the symmetric case this is the lower of the two narrowed walls;
    the upper one is `params.H - h(x)` by symmetry.
    """
    x = np.linspace(0.0, params.Lx, n)
    h = stenosis_height(x, params)
    if params.eccentric:
        return x, h
    return x, h / 2.0
