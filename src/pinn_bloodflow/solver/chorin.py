"""Chorin projection-method (fractional-step) incompressible Navier-Stokes solver.

Each timestep does three things:
  1. Advance velocity explicitly using convection + diffusion only (ignoring
     the pressure gradient for a moment) -> an intermediate, generally
     non-divergence-free "u*".
  2. Solve a pressure Poisson equation whose right-hand side is exactly the
     divergence of u* / dt -- the pressure that "absorbs" that divergence.
  3. Subtract the pressure gradient from u* to get a divergence-free u^(n+1).

Convection is discretized with a donor-cell (upwind-blended) scheme rather
than pure central differences: at Re=200 the local flow speed inside the
stenosis throat is high enough that pure central differencing would need a
finer grid than we're using to stay free of spurious oscillations. Blending
in a bit of upwinding trades a little accuracy for a lot of robustness --
the standard move in classical CFD codes (Patankar's "hybrid scheme").
"""

from dataclasses import dataclass

import numpy as np

from pinn_bloodflow.solver.immersed_boundary import enforce_no_slip
from pinn_bloodflow.solver.mac_grid import MACGrid
from pinn_bloodflow.solver.poisson import PoissonSolver


def inlet_profile(y: np.ndarray, U0: float, H: float) -> np.ndarray:
    """Fully-developed parabolic (Poiseuille) profile, mean velocity U0."""
    return 6.0 * U0 * (y / H) * (1.0 - y / H)


@dataclass
class SolverState:
    u: np.ndarray
    v: np.ndarray
    p: np.ndarray


class ChorinSolver:
    def __init__(
        self,
        grid: MACGrid,
        nu: float,
        U0: float,
        cfl_safety: float = 0.5,
        donor_cell_gamma: float = 0.9,
        inlet_amplitude_fn=None,
    ):
        self.grid = grid
        self.nu = nu
        self.U0 = U0
        self.cfl_safety = cfl_safety
        self.gamma = donor_cell_gamma
        # Optional f(t) multiplying the inlet profile, for Phase 2 pulsatile
        # flow. Defaults to a constant 1.0 (steady inflow).
        self.inlet_amplitude_fn = inlet_amplitude_fn or (lambda t: 1.0)
        self.poisson = PoissonSolver(grid)
        self._base_inlet_u = inlet_profile(grid.yu[:, 0], U0, grid.stenosis.H)

    def initial_state(self) -> SolverState:
        u = self.grid.zeros_u()
        u[:, :] = self._base_inlet_u[:, None]
        v = self.grid.zeros_v()
        u, v = enforce_no_slip(u, v, self.grid)
        return SolverState(u=u, v=v, p=self.grid.zeros_p())

    def stable_dt(self, state: SolverState) -> float:
        dx, dy = self.grid.dx, self.grid.dy
        u_max = max(np.max(np.abs(state.u)), 1e-8)
        v_max = max(np.max(np.abs(state.v)), 1e-8)
        cfl_limit = min(dx / u_max, dy / v_max)
        diffusion_limit = 0.25 * min(dx, dy) ** 2 / self.nu
        return self.cfl_safety * min(cfl_limit, diffusion_limit)

    # -- boundary conditions -------------------------------------------------

    def _apply_dirichlet_bcs(self, u: np.ndarray, v: np.ndarray, t: float) -> tuple[np.ndarray, np.ndarray]:
        """Prescribed-velocity boundaries: the inlet profile and the (exactly
        on-grid) top/bottom no-slip walls. Safe to reapply at any point in a
        step, unlike the outlet (see `_extrapolate_outlet` / the pressure
        correction in `_project`).
        """
        u = u.copy()
        v = v.copy()
        u[:, 0] = self._base_inlet_u * self.inlet_amplitude_fn(t)
        v[0, :] = 0.0  # bottom domain wall -- v lives exactly on this line
        v[-1, :] = 0.0  # top domain wall -- likewise
        return u, v

    @staticmethod
    def _extrapolate_outlet(u: np.ndarray) -> np.ndarray:
        """A first guess for the outlet face, used only to compute the
        divergence that feeds the Poisson solve. This is *not* the final
        outlet velocity -- see the note in `_project` about why re-copying
        this after projection would silently reintroduce divergence there.
        """
        u = u.copy()
        u[:, -1] = u[:, -2]
        return u

    # -- convection + diffusion (the explicit substep) ------------------------

    def _u_rhs(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        dx, dy, nx, ny, gamma = self.grid.dx, self.grid.dy, self.grid.nx, self.grid.ny, self.gamma

        # No-slip ghost rows for the top/bottom channel walls: u's y-locations
        # are cell-centered, so the wall doesn't sit exactly on a grid point --
        # a mirrored, sign-flipped ghost makes the linear interpolation to the
        # wall come out to exactly zero.
        u_pad = np.vstack([-u[0:1, :], u, -u[-1:, :]])  # shape (ny+2, nx+1)

        u_c = u[:, 1:nx]      # interior columns i = 1 .. nx-1
        u_minus = u[:, 0:nx - 1]
        u_plus = u[:, 2:nx + 1]

        # d(u^2)/dx, donor-cell blended.
        flux_e = 0.5 * (u_c + u_plus)
        flux_w = 0.5 * (u_minus + u_c)
        d_uu_dx = (flux_e ** 2 - flux_w ** 2) / dx
        d_uu_dx += gamma / dx * (
            np.abs(flux_e) * (u_c - u_plus) / 2 - np.abs(flux_w) * (u_minus - u_c) / 2
        )

        # d(uv)/dy: v (the transporting velocity) interpolated onto u's
        # location, u (the transported quantity) interpolated onto v's rows.
        v_south = 0.5 * (v[0:ny, 0:nx - 1] + v[0:ny, 1:nx])      # at row j, aligned with column i=1..nx-1
        v_north = 0.5 * (v[1:ny + 1, 0:nx - 1] + v[1:ny + 1, 1:nx])  # at row j+1

        u_row_below = u_pad[0:ny, 1:nx]   # real row j-1
        u_row_here = u_pad[1:ny + 1, 1:nx]  # real row j (== u_c)
        u_row_above = u_pad[2:ny + 2, 1:nx]  # real row j+1

        u_interp_south = 0.5 * (u_row_below + u_row_here)
        u_interp_north = 0.5 * (u_row_here + u_row_above)

        d_uv_dy = (v_north * u_interp_north - v_south * u_interp_south) / dy
        d_uv_dy += gamma / dy * (
            np.abs(v_north) * (u_c - u_row_above) / 2 - np.abs(v_south) * (u_row_below - u_c) / 2
        )

        diffusion = self.nu * (
            (u_plus - 2 * u_c + u_minus) / dx ** 2
            + (u_row_above - 2 * u_row_here + u_row_below) / dy ** 2
        )

        rhs = np.zeros_like(u)
        rhs[:, 1:nx] = -(d_uu_dx + d_uv_dy) + diffusion
        return rhs

    def _v_rhs(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        dx, dy, nx, ny, gamma = self.grid.dx, self.grid.dy, self.grid.nx, self.grid.ny, self.gamma

        # x-ghosts for v: sign-flip at the inlet (forces v=0 at x=0, matching
        # the prescribed inlet profile's v=0), plain copy at the outlet
        # (zero-gradient, matching the outflow BC used for u).
        v_pad = np.hstack([-v[:, 0:1], v, v[:, -1:]])  # shape (ny+1, nx+2)

        v_c = v[1:ny, :]       # interior rows j = 1 .. ny-1
        v_minus = v[0:ny - 1, :]
        v_plus = v[2:ny + 1, :]

        # d(v^2)/dy, donor-cell blended -- mirror image of the u^2 term.
        flux_n = 0.5 * (v_c + v_plus)
        flux_s = 0.5 * (v_minus + v_c)
        d_vv_dy = (flux_n ** 2 - flux_s ** 2) / dy
        d_vv_dy += gamma / dy * (
            np.abs(flux_n) * (v_c - v_plus) / 2 - np.abs(flux_s) * (v_minus - v_c) / 2
        )

        # d(uv)/dx: u (transporting) interpolated onto v's location, v
        # (transported) interpolated onto u's columns.
        u_row_lower = u[0:ny - 1, :]   # real row j-1
        u_row_upper = u[1:ny, :]       # real row j
        u_interp = 0.5 * (u_row_lower + u_row_upper)  # y-averaged, all nx+1 columns
        u_at_w = u_interp[:, 0:nx]
        u_at_e = u_interp[:, 1:nx + 1]

        v_pad_rows = v_pad[1:ny, :]  # interior rows, x-padded
        v_at_w = 0.5 * (v_pad_rows[:, 0:nx] + v_pad_rows[:, 1:nx + 1])
        v_at_e = 0.5 * (v_pad_rows[:, 1:nx + 1] + v_pad_rows[:, 2:nx + 2])

        d_uv_dx = (u_at_e * v_at_e - u_at_w * v_at_w) / dx
        d_uv_dx += gamma / dx * (
            np.abs(u_at_e) * (v_c - v_plus) / 2 - np.abs(u_at_w) * (v_minus - v_c) / 2
        )

        diffusion = self.nu * (
            (v_pad_rows[:, 2:nx + 2] - 2 * v_c + v_pad_rows[:, 0:nx]) / dx ** 2
            + (v_plus - 2 * v_c + v_minus) / dy ** 2
        )

        rhs = np.zeros_like(v)
        rhs[1:ny, :] = -(d_vv_dy + d_uv_dx) + diffusion
        return rhs

    # -- projection ------------------------------------------------------------

    def _divergence(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        dx, dy = self.grid.dx, self.grid.dy
        return (u[:, 1:] - u[:, :-1]) / dx + (v[1:, :] - v[:-1, :]) / dy

    def _project(self, u: np.ndarray, v: np.ndarray, p: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
        dx, dy, nx, ny = self.grid.dx, self.grid.dy, self.grid.nx, self.grid.ny
        u_new = u.copy()
        v_new = v.copy()

        # p at solid cells is an arbitrary dummy value (the Poisson matrix
        # decouples them with an identity row) -- it must never be read as a
        # real neighboring pressure. Where either side of a face is solid,
        # apply the same zero-flux convention used when assembling the
        # Poisson matrix itself: no pressure correction across that face.
        mask_w, mask_e = self.grid.mask_p[:, 0:nx - 1], self.grid.mask_p[:, 1:nx]
        grad_x = np.where(mask_w & mask_e, p[:, 1:nx] - p[:, 0:nx - 1], 0.0)
        u_new[:, 1:nx] = u[:, 1:nx] - dt * grad_x / dx

        mask_s, mask_n = self.grid.mask_p[0:ny - 1, :], self.grid.mask_p[1:ny, :]
        grad_y = np.where(mask_s & mask_n, p[1:ny, :] - p[0:ny - 1, :], 0.0)
        v_new[1:ny, :] = v[1:ny, :] - dt * grad_y / dy

        # The outlet face needs the SAME ghost-pressure convention used to
        # build the Poisson matrix's last column (p_ghost = -p[:, -1]), not
        # an independent zero-gradient copy -- otherwise the two boundary
        # treatments disagree about what "divergence-free" means right at
        # that face, and mass conservation quietly breaks there (this is
        # exactly the bug the M3 validation run caught: max|div u| growing
        # from 1e-8 to 1e-1 over a few thousand steps, concentrated entirely
        # in the last column). Flux through that face is
        # (p_ghost - p[:, -1]) / dx = -2 p[:, -1] / dx.
        u_new[:, -1] = u[:, -1] + 2.0 * dt * p[:, -1] / dx

        return u_new, v_new

    # -- public stepping API ----------------------------------------------------

    def step(self, state: SolverState, dt: float, t: float = 0.0) -> SolverState:
        u_star = state.u + dt * self._u_rhs(state.u, state.v)
        v_star = state.v + dt * self._v_rhs(state.u, state.v)
        u_star, v_star = self._apply_dirichlet_bcs(u_star, v_star, t)
        u_star = self._extrapolate_outlet(u_star)  # a guess, only for the divergence below
        u_star, v_star = enforce_no_slip(u_star, v_star, self.grid)

        rhs_poisson = self._divergence(u_star, v_star) / dt
        p_new = self.poisson.solve(rhs_poisson)

        # _project both corrects the interior faces AND replaces the outlet's
        # extrapolated guess with the pressure-consistent value -- do not
        # re-extrapolate the outlet after this, or the correction is lost.
        u_new, v_new = self._project(u_star, v_star, p_new, dt)
        u_new, v_new = self._apply_dirichlet_bcs(u_new, v_new, t)
        u_new, v_new = enforce_no_slip(u_new, v_new, self.grid)
        return SolverState(u=u_new, v=v_new, p=p_new)

    def run_to_steady(
        self, max_steps: int = 20000, tol: float = 1e-6, check_every: int = 50
    ) -> tuple[SolverState, int]:
        """Pseudo-time-step (t is irrelevant here, inlet amplitude stays 1) until
        the change in u between checks drops below tol. Returns (state, n_steps).
        """
        state = self.initial_state()
        for step_idx in range(1, max_steps + 1):
            dt = self.stable_dt(state)
            u_prev = state.u
            state = self.step(state, dt, t=0.0)
            if step_idx % check_every == 0:
                change = np.max(np.abs(state.u - u_prev)) / max(np.max(np.abs(state.u)), 1e-8)
                if change < tol:
                    return state, step_idx
        return state, max_steps

    def _advance_to(self, state: SolverState, t: float, t_target: float) -> tuple[SolverState, float]:
        """Real time-accurate stepping (unlike `run_to_steady`, `t` now
        matters -- it drives `inlet_amplitude_fn`) from `t` to `t_target`,
        using as many CFL-limited substeps as needed. The last substep is
        shortened to land exactly on `t_target` rather than overshoot it, so
        the caller can request a snapshot at a precise, predictable time.
        """
        while t < t_target - 1e-12:
            dt = min(self.stable_dt(state), t_target - t)
            state = self.step(state, dt, t=t)
            t += dt
        return state, t

    def run_transient(
        self,
        period: float,
        n_cycles: int,
        snapshots_per_cycle: int,
        record_from_cycle: int | None = None,
    ) -> list[dict]:
        """Time-accurate pulsatile run. Simulates `n_cycles` full periods but
        only *records* snapshots starting at `record_from_cycle` (default:
        the last cycle) -- the flow needs a cycle or two, starting from the
        steady-inflow initial condition, to forget that startup transient
        and settle into a repeating limit cycle (this is the transient/
        periodic distinction from the project plan). Returns a list of
        dicts with keys t, u, v, p, one per snapshot, evenly spaced over
        the recorded cycle(s) with `snapshots_per_cycle + 1` points so the
        first and last snapshot are exactly one period apart -- deliberately
        redundant (same phase, recomputed from scratch) so periodicity can
        be checked directly against the solver's own output rather than
        assumed.
        """
        if record_from_cycle is None:
            record_from_cycle = n_cycles - 1
        record_start_t = record_from_cycle * period
        total_time = n_cycles * period
        snapshot_dt = period / snapshots_per_cycle

        state = self.initial_state()
        t = 0.0
        state, t = self._advance_to(state, t, record_start_t)

        snapshot_times = record_start_t + snapshot_dt * np.arange(snapshots_per_cycle + 1)
        snapshot_times = snapshot_times[snapshot_times <= total_time + 1e-9]

        snapshots = []
        for t_target in snapshot_times:
            state, t = self._advance_to(state, t, float(t_target))
            snapshots.append({"t": t, "u": state.u.copy(), "v": state.v.copy(), "p": state.p.copy()})
        return snapshots
