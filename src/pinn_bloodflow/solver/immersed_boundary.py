"""Apply the stenosis's immersed boundary to the velocity fields.

After the explicit advection-diffusion substep, any face that sits inside
(or on the boundary of) the solid stenosis must be forced back to the
no-slip condition u=v=0 -- the substep has no knowledge of the mask.
"""

import numpy as np

from pinn_bloodflow.solver.mac_grid import MACGrid


def enforce_no_slip(u: np.ndarray, v: np.ndarray, grid: MACGrid) -> tuple[np.ndarray, np.ndarray]:
    """Zero out velocity on any solid (masked-out) face. Returns new arrays."""
    u_out = np.where(grid.mask_u, u, 0.0)
    v_out = np.where(grid.mask_v, v, 0.0)
    return u_out, v_out
