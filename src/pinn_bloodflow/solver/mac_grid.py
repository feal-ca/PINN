"""Staggered (Marker-and-Cell) grid for the Chorin projection-method solver.

Pressure lives at cell centers; u lives on vertical cell faces (left/right);
v lives on horizontal cell faces (top/bottom). This is the classic
arrangement that avoids checkerboard pressure oscillations in incompressible
solvers (Harlow & Welch, 1965).
"""

from dataclasses import dataclass, field

import numpy as np

from pinn_bloodflow.geometry import StenosisParams, build_fluid_mask


@dataclass
class MACGrid:
    nx: int
    ny: int
    stenosis: StenosisParams

    dx: float = field(init=False)
    dy: float = field(init=False)

    # Cell-center (pressure) locations: shape (ny, nx)
    xp: np.ndarray = field(init=False)
    yp: np.ndarray = field(init=False)
    mask_p: np.ndarray = field(init=False)

    # u locations, vertical faces: shape (ny, nx+1)
    xu: np.ndarray = field(init=False)
    yu: np.ndarray = field(init=False)
    mask_u: np.ndarray = field(init=False)

    # v locations, horizontal faces: shape (ny+1, nx)
    xv: np.ndarray = field(init=False)
    yv: np.ndarray = field(init=False)
    mask_v: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        Lx, H = self.stenosis.Lx, self.stenosis.H
        self.dx = Lx / self.nx
        self.dy = H / self.ny

        x_p_1d = (np.arange(self.nx) + 0.5) * self.dx
        y_p_1d = (np.arange(self.ny) + 0.5) * self.dy
        self.xp, self.yp = np.meshgrid(x_p_1d, y_p_1d)
        self.mask_p = build_fluid_mask(x_p_1d, y_p_1d, self.stenosis)

        x_u_1d = np.arange(self.nx + 1) * self.dx
        y_u_1d = y_p_1d
        self.xu, self.yu = np.meshgrid(x_u_1d, y_u_1d)
        self.mask_u = build_fluid_mask(x_u_1d, y_u_1d, self.stenosis)

        x_v_1d = x_p_1d
        y_v_1d = np.arange(self.ny + 1) * self.dy
        self.xv, self.yv = np.meshgrid(x_v_1d, y_v_1d)
        self.mask_v = build_fluid_mask(x_v_1d, y_v_1d, self.stenosis)

    def zeros_p(self) -> np.ndarray:
        return np.zeros((self.ny, self.nx))

    def zeros_u(self) -> np.ndarray:
        return np.zeros((self.ny, self.nx + 1))

    def zeros_v(self) -> np.ndarray:
        return np.zeros((self.ny + 1, self.nx))
