"""Simulate sparse, noisy point velocity measurements.

This is a simplified stand-in for sparse Doppler-ultrasound-style velocity
probes: it samples the full (u, v) vector at a handful of random points,
each corrupted by isotropic Gaussian noise. It deliberately does NOT model
real duplex Doppler physics -- beam angle, speckle, a finite sample volume
-- that's a different, harder problem. This is the same level of
abstraction used in the inverse-PINN blood-flow literature this project is
built on (see article/draft.md's citation list).

Points are drawn only from the fluid interior, excluding a margin near any
wall or the stenosis surface. That exclusion is deliberate and honest: it's
exactly the near-wall region the PINN is later asked to reconstruct from
physics alone, since no sensor ever measures it directly.
"""

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from pinn_bloodflow.geometry import StenosisParams, stenosis_height


@dataclass
class SensorData:
    x: np.ndarray
    y: np.ndarray
    u_true: np.ndarray  # noise-free, held out for evaluation only
    v_true: np.ndarray
    u_meas: np.ndarray  # what the PINN actually trains on
    v_meas: np.ndarray
    noise_level: float


@dataclass
class SensorDataPulsatile:
    """Phase 2's sensor dataset: the same handful of *spatial* probe
    locations, but each one re-sampled at several times through the cycle
    -- mimicking a fixed Doppler probe recording continuously rather than
    a new random location at every instant. `x`, `y` are per-probe
    (n_points,); `t` is per-snapshot (n_snapshots,); the measurement
    arrays are (n_snapshots, n_points), one row per sampled time.
    """

    x: np.ndarray
    y: np.ndarray
    t: np.ndarray
    u_true: np.ndarray
    v_true: np.ndarray
    u_meas: np.ndarray
    v_meas: np.ndarray
    noise_level: float


def _sample_locations(
    params: StenosisParams,
    n_points: int,
    wall_margin_frac: float,
    rng: np.random.Generator,
    x_range: tuple[float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Rejection-sample points uniformly in the fluid interior, excluding a
    margin (as a fraction of H) near the top wall and the stenosis surface.
    `x_range`, when given, restricts candidate x to that band instead of the
    whole channel length -- used to guarantee some probes near the throat
    (see `sample_sensors_pulsatile`'s `throat_frac`).
    """
    margin = wall_margin_frac * params.H
    x_lo, x_hi = x_range if x_range is not None else (0.0, params.Lx)
    xs = np.empty(n_points)
    ys = np.empty(n_points)
    filled = 0
    while filled < n_points:
        batch = (n_points - filled) * 4  # oversample; rejection rate is modest
        x_candidates = rng.uniform(x_lo, x_hi, size=batch)
        y_candidates = rng.uniform(0, params.H, size=batch)
        h = stenosis_height(x_candidates, params)
        if params.eccentric:
            lower, upper = h, params.H
        else:
            lower, upper = h / 2.0, params.H - h / 2.0
        valid = (y_candidates - lower > margin) & (upper - y_candidates > margin)
        x_ok, y_ok = x_candidates[valid], y_candidates[valid]
        take = min(len(x_ok), n_points - filled)
        xs[filled:filled + take] = x_ok[:take]
        ys[filled:filled + take] = y_ok[:take]
        filled += take
    return xs, ys


def sample_sensors(
    ground_truth: dict,
    params: StenosisParams,
    n_points: int,
    wall_margin_frac: float,
    noise_level: float,
    seed: int,
) -> SensorData:
    """Draw sparse, noisy velocity measurements from a saved ground-truth field.

    `ground_truth` is a dict-like object (e.g. a loaded .npz) with cell-center
    arrays "xp", "yp", "u", "v" and a scalar "U0" -- exactly what
    `scripts/generate_ground_truth.py` produces.
    """
    rng = np.random.default_rng(seed)
    x, y = _sample_locations(params, n_points, wall_margin_frac, rng)

    x1d = ground_truth["xp"][0, :]
    y1d = ground_truth["yp"][:, 0]
    u_interp = RegularGridInterpolator((y1d, x1d), ground_truth["u"])
    v_interp = RegularGridInterpolator((y1d, x1d), ground_truth["v"])
    query_points = np.stack([y, x], axis=-1)
    u_true = u_interp(query_points)
    v_true = v_interp(query_points)

    sigma = noise_level * float(ground_truth["U0"])
    u_meas = u_true + rng.normal(0.0, sigma, size=u_true.shape)
    v_meas = v_true + rng.normal(0.0, sigma, size=v_true.shape)

    return SensorData(x=x, y=y, u_true=u_true, v_true=v_true, u_meas=u_meas, v_meas=v_meas, noise_level=noise_level)


def sample_sensors_pulsatile(
    ground_truth: dict,
    params: StenosisParams,
    n_points: int,
    wall_margin_frac: float,
    noise_level: float,
    n_snapshots: int,
    seed: int,
    throat_frac: float = 0.0,
    throat_half_width: float = 0.6,
) -> SensorDataPulsatile:
    """Draw sparse, noisy velocity measurements from a saved *pulsatile*
    ground-truth field (dense in time, produced by
    `scripts/generate_ground_truth_phase2.py`), at a sparser set of times
    than the ground truth itself was recorded at. This is meant to model a
    fixed probe re-read periodically through the cycle -- the same spatial
    locations every time, not a fresh random set of probes at each instant.

    `throat_frac`, when > 0, reserves that fraction of probes for the band
    `x0 +/- throat_half_width` around the stenosis instead of the whole
    channel: with fully uniform placement, 20 random probes across a 5-unit
    channel put essentially none of them near the throat by chance (dev-log
    2026-08-28 found only 1 of 20, and not even near the wall), starving the
    PINN of any direct data telling it what happens at the one location the
    headline wall-shear-stress result is evaluated. This is also the more
    clinically realistic choice: a real Doppler exam images the suspected
    narrowing deliberately, rather than scanning uniformly at random.
    """
    rng = np.random.default_rng(seed)
    n_throat = int(round(n_points * throat_frac))
    n_general = n_points - n_throat
    x_g, y_g = _sample_locations(params, n_general, wall_margin_frac, rng)
    if n_throat > 0:
        x_range = (max(0.0, params.x0 - throat_half_width), min(params.Lx, params.x0 + throat_half_width))
        x_t, y_t = _sample_locations(params, n_throat, wall_margin_frac, rng, x_range=x_range)
        x, y = np.concatenate([x_g, x_t]), np.concatenate([y_g, y_t])
    else:
        x, y = x_g, y_g

    t1d = ground_truth["t"]
    x1d = ground_truth["xp"][0, :]
    y1d = ground_truth["yp"][:, 0]
    u_interp = RegularGridInterpolator((t1d, y1d, x1d), ground_truth["u"])
    v_interp = RegularGridInterpolator((t1d, y1d, x1d), ground_truth["v"])

    # endpoint=False: t1d[0] and t1d[-1] are exactly one period apart, i.e.
    # the same phase in a periodic flow -- sampling both would just be a
    # duplicate measurement, not new information.
    t_snap = np.linspace(t1d[0], t1d[-1], n_snapshots, endpoint=False)
    tt, yy = np.meshgrid(t_snap, y, indexing="ij")  # (n_snapshots, n_points), broadcasting probes over times
    _, xx = np.meshgrid(t_snap, x, indexing="ij")
    query_points = np.stack([tt, yy, xx], axis=-1)
    u_true = u_interp(query_points)
    v_true = v_interp(query_points)

    sigma = noise_level * float(ground_truth["U0"])
    u_meas = u_true + rng.normal(0.0, sigma, size=u_true.shape)
    v_meas = v_true + rng.normal(0.0, sigma, size=v_true.shape)

    return SensorDataPulsatile(
        x=x, y=y, t=t_snap, u_true=u_true, v_true=v_true, u_meas=u_meas, v_meas=v_meas, noise_level=noise_level,
    )


def sample_sensors_zoned(
    ground_truth: dict,
    params: StenosisParams,
    zones: list[tuple[float, float, int]],
    wall_margin_frac: float,
    noise_level: float,
    seed: int,
) -> SensorData:
    """Steady-flow sensors placed by zone: each `(x_lo, x_hi, n)` puts `n`
    probes uniformly inside that x band. Lets the probes concentrate where
    the physics is hard (the throat and the wake) instead of spreading
    evenly over a channel that is mostly smooth."""
    rng = np.random.default_rng(seed)
    xs, ys = [], []
    for x_lo, x_hi, n in zones:
        x, y = _sample_locations(params, n, wall_margin_frac, rng, x_range=(x_lo, x_hi))
        xs.append(x)
        ys.append(y)
    x, y = np.concatenate(xs), np.concatenate(ys)

    x1d = ground_truth["xp"][0, :]
    y1d = ground_truth["yp"][:, 0]
    u_interp = RegularGridInterpolator((y1d, x1d), ground_truth["u"])
    v_interp = RegularGridInterpolator((y1d, x1d), ground_truth["v"])
    query_points = np.stack([y, x], axis=-1)
    u_true = u_interp(query_points)
    v_true = v_interp(query_points)

    sigma = noise_level * float(ground_truth["U0"])
    u_meas = u_true + rng.normal(0.0, sigma, size=u_true.shape)
    v_meas = v_true + rng.normal(0.0, sigma, size=v_true.shape)
    return SensorData(x=x, y=y, u_true=u_true, v_true=v_true, u_meas=u_meas, v_meas=v_meas, noise_level=noise_level)
