"""Metrics that compare a trained PINN with the CFD reference.

Everything here works on the cell-centre grid the ground truth is saved on,
so the reference and the network are always read the same way:

- velocity and vorticity errors are relative L2 over the fluid cells,
- wall shear stress (WSS) is estimated from the velocity a few cells away
  from the wall, along the wall normal, for the reference and the network
  alike. That avoids differencing across the solver's staircase boundary,
  which is what made the old reference so jagged.
"""

from dataclasses import dataclass

import numpy as np
import torch
from scipy.interpolate import RegularGridInterpolator

from pinn_bloodflow.geometry import StenosisParams, stenosis_height

# Distances from the wall (in units of H) at which the tangential velocity is
# sampled. The nearest one is about four cells from the wall, clear of the
# staircase.
WSS_OFFSETS = np.array([0.03, 0.045, 0.06, 0.075, 0.09])


def relative_l2(pred: np.ndarray, true: np.ndarray, mask: np.ndarray) -> float:
    diff = np.where(mask, pred - true, 0.0)
    ref = np.where(mask, true, 0.0)
    return float(np.linalg.norm(diff) / np.linalg.norm(ref))


def vorticity(u: np.ndarray, v: np.ndarray, xp: np.ndarray, yp: np.ndarray, mask: np.ndarray) -> np.ndarray:
    dx = xp[0, 1] - xp[0, 0]
    dy = yp[1, 0] - yp[0, 0]
    w = np.gradient(v, dx, axis=1) - np.gradient(u, dy, axis=0)
    return np.where(mask, w, 0.0)


def eroded(mask: np.ndarray, cells: int = 2) -> np.ndarray:
    """Fluid mask shrunk by a few cells, to keep the one-sided differences
    at the solid boundary out of vorticity comparisons."""
    out = mask.copy()
    for _ in range(cells):
        m = out.copy()
        m[1:, :] &= out[:-1, :]
        m[:-1, :] &= out[1:, :]
        m[:, 1:] &= out[:, :-1]
        m[:, :-1] &= out[:, 1:]
        out = m
    return out


def predict_grid(model, xp: np.ndarray, yp: np.ndarray):
    x = torch.tensor(xp.ravel(), dtype=torch.float32).reshape(-1, 1)
    y = torch.tensor(yp.ravel(), dtype=torch.float32).reshape(-1, 1)
    with torch.no_grad():
        u, v, p = model(x, y)
    shape = xp.shape
    return u.numpy().reshape(shape), v.numpy().reshape(shape), p.numpy().reshape(shape)


def wall_stations(params: StenosisParams, n: int = 400, margin: float = 0.2):
    """Points on the bottom wall with the unit tangent and inward normal."""
    x = np.linspace(margin, params.Lx - margin, n)
    h = stenosis_height(x, params)
    eps = 1e-4
    slope = (stenosis_height(x + eps, params) - stenosis_height(x - eps, params)) / (2 * eps)
    norm = np.sqrt(1 + slope**2)
    t = np.stack([1 / norm, slope / norm], axis=1)
    nrm = np.stack([-slope / norm, 1 / norm], axis=1)
    return x, h, t, nrm


def _fit_gradient(u_t: np.ndarray) -> np.ndarray:
    """Least-squares a in u_t(d) = a*d + b*d^2 (no-slip fixes the intercept).
    `u_t` has shape (n_stations, n_offsets). Returns a per station."""
    d = WSS_OFFSETS
    A = np.stack([d, d**2], axis=1)                       # (k, 2)
    coef, *_ = np.linalg.lstsq(A, u_t.T, rcond=None)      # (2, n)
    return coef[0]


def wss_gradient_reference(gt, params: StenosisParams):
    """d u_t / d n at the wall, from the CFD field."""
    x, h, t, nrm = wall_stations(params)
    xg, yg = gt["xp"][0], gt["yp"][:, 0]
    fu = RegularGridInterpolator((yg, xg), gt["u"])
    fv = RegularGridInterpolator((yg, xg), gt["v"])
    u_t = np.empty((len(x), len(WSS_OFFSETS)))
    for j, d in enumerate(WSS_OFFSETS):
        px = x + d * nrm[:, 0]
        py = h + d * nrm[:, 1]
        pts = np.stack([py, px], axis=1)
        u_t[:, j] = fu(pts) * t[:, 0] + fv(pts) * t[:, 1]
    return x, _fit_gradient(u_t)


def wss_gradient_pinn(model, params: StenosisParams):
    x, h, t, nrm = wall_stations(params)
    u_t = np.empty((len(x), len(WSS_OFFSETS)))
    for j, d in enumerate(WSS_OFFSETS):
        px = torch.tensor(x + d * nrm[:, 0], dtype=torch.float32).reshape(-1, 1)
        py = torch.tensor(h + d * nrm[:, 1], dtype=torch.float32).reshape(-1, 1)
        with torch.no_grad():
            u, v, _ = model(px, py)
        u_t[:, j] = u.numpy().ravel() * t[:, 0] + v.numpy().ravel() * t[:, 1]
    return x, _fit_gradient(u_t)


@dataclass
class Metrics:
    field_u: float
    field_v: float
    field_mag: float
    vorticity: float
    vorticity_wake: float
    pressure: float
    wss: float
    wss_true_nu: float
    wss_peak_ratio: float
    nu_true: float
    nu_recovered: float
    nu_error: float


def evaluate(model, log_nu: float, gt, params: StenosisParams, sensors=None) -> tuple[Metrics, dict]:
    mask = gt["mask_p"]
    xp, yp = gt["xp"], gt["yp"]
    u_p, v_p, p_p = predict_grid(model, xp, yp)
    u_t, v_t, p_t = gt["u"], gt["v"], gt["p"]

    mag_p, mag_t = np.hypot(u_p, v_p), np.hypot(u_t, v_t)
    w_t = vorticity(u_t, v_t, xp, yp, mask)
    w_p = vorticity(u_p, v_p, xp, yp, mask)
    core = eroded(mask, 3)
    wake = core & (xp > params.x0 + 0.4)

    pc = np.where(mask, p_p - p_p[mask].mean(), 0)
    tc = np.where(mask, p_t - p_t[mask].mean(), 0)

    nu_true = float(gt["nu"])
    nu_rec = float(np.exp(log_nu))
    xs, g_true = wss_gradient_reference(gt, params)
    _, g_pred = wss_gradient_pinn(model, params)
    wss_true = nu_true * g_true
    wss_pred = nu_rec * g_pred
    wss_pred_true_nu = nu_true * g_pred
    peak = np.argmax(wss_true)

    m = Metrics(
        field_u=relative_l2(u_p, u_t, mask),
        field_v=relative_l2(v_p, v_t, mask),
        field_mag=relative_l2(mag_p, mag_t, mask),
        vorticity=relative_l2(w_p, w_t, core),
        vorticity_wake=relative_l2(w_p, w_t, wake),
        pressure=relative_l2(pc, tc, mask),
        wss=float(np.linalg.norm(wss_pred - wss_true) / np.linalg.norm(wss_true)),
        wss_true_nu=float(np.linalg.norm(wss_pred_true_nu - wss_true) / np.linalg.norm(wss_true)),
        wss_peak_ratio=float(wss_pred[peak] / wss_true[peak]),
        nu_true=nu_true, nu_recovered=nu_rec, nu_error=abs(nu_rec - nu_true) / nu_true,
    )
    extras = dict(u_p=u_p, v_p=v_p, p_p=p_p, w_p=w_p, w_t=w_t, xs=xs, wss_true=wss_true, wss_pred=wss_pred)
    return m, extras
