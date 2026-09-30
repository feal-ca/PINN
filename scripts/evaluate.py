"""Evaluate the trained Phase 1 PINN against the ground-truth field.

Computes the headline numbers (field reconstruction error, viscosity
recovery, wall shear stress comparison) and produces the comparison
figures used in the article.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pinn_bloodflow.geometry import StenosisParams, stenosis_height  # noqa: E402
from pinn_bloodflow.pinn.network import PINN  # noqa: E402
from pinn_bloodflow.viz.plots import (  # noqa: E402
    plot_error_heatmap, plot_field_comparison, plot_training_curves, plot_wss_comparison,
)


def relative_l2_error(pred: np.ndarray, true: np.ndarray, mask: np.ndarray) -> float:
    diff = np.where(mask, pred - true, 0.0)
    ref = np.where(mask, true, 0.0)
    return float(np.linalg.norm(diff) / np.linalg.norm(ref))


def compute_wss_ground_truth(gt, params: StenosisParams) -> tuple[np.ndarray, np.ndarray]:
    """Near-wall velocity-gradient estimate from the first two fluid cells
    in each column, using the (fixed, well-behaved) grid spacing between
    them -- NOT the distance from a cell center to the continuous wall
    curve, which can coincidentally shrink to near-zero at some columns and
    blow up a naive one-sided finite difference (found via a 98% "error"
    that turned out to be a handful of columns reporting WSS ~14, ~50x
    every neighboring value -- see dev-log 2026-08-26). Using two interior
    points and linearly extrapolating the gradient sidesteps that
    coincidence entirely, at the cost of assuming the near-wall profile is
    locally linear, which is standard practice for this kind of
    staircase-boundary post-processing.
    """
    x1d = gt["xp"][0, :]
    y1d = gt["yp"][:, 0]
    wss = np.full_like(x1d, np.nan)
    for i in range(len(x1d)):
        fluid_js = np.where(gt["mask_p"][:, i])[0]
        if len(fluid_js) < 2:
            continue
        j0, j1 = fluid_js[0], fluid_js[1]
        slope = (gt["u"][j1, i] - gt["u"][j0, i]) / (y1d[j1] - y1d[j0])
        wss[i] = float(gt["nu"]) * slope
    return x1d, wss


def compute_wss_pinn(model: PINN, x1d: np.ndarray, params: StenosisParams, nu: float) -> np.ndarray:
    h = stenosis_height(x1d, params)
    x = torch.tensor(x1d, dtype=torch.float32).reshape(-1, 1)
    y = torch.tensor(h, dtype=torch.float32).reshape(-1, 1).requires_grad_(True)
    u, _, _ = model(x, y)
    u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u))[0]
    return nu * u_y.detach().numpy().ravel()


def main(config_path: str, ground_truth_path: str, sensors_path: str, checkpoint_path: str, figures_dir: str) -> None:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    geo = cfg["geometry"]
    params = StenosisParams(
        H=geo["H"], Lx=geo["Lx"], x0=geo["stenosis_x0"],
        pct_ds=geo["stenosis_pct_ds"], sigma=geo["stenosis_sigma"], eccentric=geo["eccentric"],
    )

    gt = np.load(ground_truth_path)
    sensors = np.load(sensors_path)
    ckpt = torch.load(checkpoint_path, weights_only=False)

    model = PINN(in_dim=2, hidden_layers=ckpt["hidden_layers"], hidden_units=ckpt["hidden_units"])
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    nu_recovered = torch.exp(ckpt["log_nu"]).item()
    nu_true = ckpt["nu_true"]

    # Evaluate the PINN on the same grid as the ground truth.
    x_flat = torch.tensor(gt["xp"].ravel(), dtype=torch.float32).reshape(-1, 1)
    y_flat = torch.tensor(gt["yp"].ravel(), dtype=torch.float32).reshape(-1, 1)
    with torch.no_grad():
        u_pred, v_pred, p_pred = model(x_flat, y_flat)
    shape = gt["xp"].shape
    u_pinn = u_pred.numpy().reshape(shape)
    v_pinn = v_pred.numpy().reshape(shape)

    mask = gt["mask_p"]
    err_u = relative_l2_error(u_pinn, gt["u"], mask)
    err_v = relative_l2_error(v_pinn, gt["v"], mask)
    mag_pinn = np.hypot(u_pinn, v_pinn)
    mag_gt = np.hypot(gt["u"], gt["v"])
    err_mag = relative_l2_error(mag_pinn, mag_gt, mask)

    print(f"Field relative L2 error: u={err_u:.3%}  v={err_v:.3%}  |velocity|={err_mag:.3%}")
    print(f"Viscosity: true={nu_true:.5f}  recovered={nu_recovered:.5f}  rel_error={abs(nu_recovered-nu_true)/nu_true:.2%}")

    # Pressure is only defined up to an additive constant, so compare the
    # fields after removing each one's mean over the fluid region. Without
    # this the "error" is dominated by an arbitrary offset.
    p_pinn = p_pred.numpy().reshape(shape)
    p_gt = gt["p"]
    p_pinn_c = p_pinn - p_pinn[mask].mean()
    p_gt_c = p_gt - p_gt[mask].mean()
    err_p = relative_l2_error(p_pinn_c, p_gt_c, mask)
    dp_gt = p_gt_c[mask].max() - p_gt_c[mask].min()
    dp_pinn = p_pinn_c[mask].max() - p_pinn_c[mask].min()
    print(
        f"Pressure (gauge-shifted) relative L2 error: {err_p:.3%}  "
        f"(range: gt={dp_gt:.3f} pinn={dp_pinn:.3f})"
    )

    x_wall, wss_gt = compute_wss_ground_truth(gt, params)
    wss_pinn = compute_wss_pinn(model, x_wall, params, nu_recovered)
    valid = ~np.isnan(wss_gt)  # columns with fewer than 2 fluid cells (deep in the stenosis)
    wss_err = np.linalg.norm(wss_pinn[valid] - wss_gt[valid]) / np.linalg.norm(wss_gt[valid])
    print(f"Wall shear stress relative L2 error: {wss_err:.3%}")

    # WSS is nu * du/dy, a product of an inferred parameter and an inferred
    # field, so agreement on the product says nothing about either factor.
    # An earlier version reported only the product and it looked good: the
    # 21.8%-high recovered viscosity was cancelling a 22.6%-low velocity
    # gradient. Report the gradient on its own so that can't happen again.
    grad_gt = wss_gt / float(gt["nu"])
    grad_pinn = wss_pinn / nu_recovered
    grad_err = np.linalg.norm(grad_pinn[valid] - grad_gt[valid]) / np.linalg.norm(grad_gt[valid])
    peak = np.nanargmax(grad_gt)
    print(
        f"Near-wall du/dy relative L2 error: {grad_err:.3%}  "
        f"(peak: gt={grad_gt[peak]:.3f} pinn={grad_pinn[peak]:.3f}, "
        f"ratio={grad_pinn[peak]/grad_gt[peak]:.3f})"
    )
    wss_at_true_nu = compute_wss_pinn(model, x_wall, params, nu_true)
    err_true_nu = (
        np.linalg.norm(wss_at_true_nu[valid] - wss_gt[valid]) / np.linalg.norm(wss_gt[valid])
    )
    print(f"  WSS error if the true viscosity were used instead: {err_true_nu:.3%}")

    # Where does the WSS error actually live? Quoted in the article, so it
    # gets computed here rather than in a one-off snippet.
    err = wss_pinn[valid] - wss_gt[valid]
    xv = x_wall[valid]
    near_peak = np.abs(xv - params.x0) < 0.4
    print(
        f"  error inside |x-x0|<0.4: {100 * (err[near_peak] ** 2).sum() / (err ** 2).sum():.0f}% "
        f"of squared error, vs {100 * (wss_gt[valid][near_peak] ** 2).sum() / (wss_gt[valid] ** 2).sum():.0f}% "
        f"of squared signal"
    )
    # How much of the error is just the reference being rough? Compare against
    # a moving average of the ground truth. Window matters, so report a few.
    for k in (5, 9, 21):
        sm = np.convolve(wss_gt[valid], np.ones(k) / k, mode="same")
        m = slice(k // 2, -(k // 2))
        rel = np.linalg.norm(wss_pinn[valid][m] - sm[m]) / np.linalg.norm(sm[m])
        print(f"  vs {k}-point smoothed reference: {rel:.3%}")

    figures = Path(figures_dir)
    figures.mkdir(parents=True, exist_ok=True)
    plot_field_comparison(
        gt["xp"], gt["yp"], mask, gt["u"], gt["v"], u_pinn, v_pinn,
        sensors["x"], sensors["y"], figures / "phase1_field_comparison.png",
    )
    plot_error_heatmap(gt["xp"], gt["yp"], mask, gt["u"], gt["v"], u_pinn, v_pinn, figures / "phase1_error_heatmap.png")
    plot_wss_comparison(x_wall, wss_gt, wss_pinn, figures / "phase1_wss_comparison.png")
    plot_training_curves(ckpt["history"], nu_true, figures / "phase1_training_curves.png")
    print("Saved figures to", figures)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase1_steady.yaml")
    parser.add_argument("--ground-truth", default="data/ground_truth/phase1_steady.npz")
    parser.add_argument("--sensors", default="data/sensors/phase1_sensors.npz")
    parser.add_argument("--checkpoint", default="data/checkpoints/phase1_pinn.pt")
    parser.add_argument("--figures-dir", default="article/figures")
    args = parser.parse_args()
    main(args.config, args.ground_truth, args.sensors, args.checkpoint, args.figures_dir)
