"""Evaluate the trained Phase 2 (pulsatile) PINN against the pulsatile
ground truth (M12). Produces the headline pulsatile results: field error
over the cycle, a throat WSS(t) trace, and a GT-vs-PINN animation.
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
    animate_field_comparison, plot_training_curves_phase2, plot_wss_time_trace,
)


def relative_l2_error(pred: np.ndarray, true: np.ndarray, mask: np.ndarray) -> float:
    diff = np.where(mask, pred - true, 0.0)
    ref = np.where(mask, true, 0.0)
    return float(np.linalg.norm(diff) / np.linalg.norm(ref))


def compute_wss_gt_time_trace(gt, throat_col: int, y1d: np.ndarray) -> np.ndarray:
    """Same two-interior-point gradient estimate as Phase 1's WSS fix
    (dev-log 2026-08-26), evaluated at every saved time snapshot for one
    fixed x-column (the throat).
    """
    fluid_js = np.where(gt["mask_p"][:, throat_col])[0]
    j0, j1 = fluid_js[0], fluid_js[1]
    slope = (gt["u"][:, j1, throat_col] - gt["u"][:, j0, throat_col]) / (y1d[j1] - y1d[j0])
    return float(gt["nu"]) * slope


def compute_wss_pinn_time_trace(model: PINN, t_arr: np.ndarray, x_throat: float, y_throat: float, nu: float) -> np.ndarray:
    n = len(t_arr)
    x = torch.full((n, 1), float(x_throat), dtype=torch.float32)
    y = torch.full((n, 1), float(y_throat), dtype=torch.float32).requires_grad_(True)
    t = torch.tensor(t_arr, dtype=torch.float32).reshape(-1, 1)
    u, _, _ = model(x, y, t)
    u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u))[0]
    return nu * u_y.detach().numpy().ravel()


def main(config_path: str, ground_truth_path: str, checkpoint_path: str, figures_dir: str) -> None:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    geo = cfg["geometry"]
    params = StenosisParams(
        H=geo["H"], Lx=geo["Lx"], x0=geo["stenosis_x0"],
        pct_ds=geo["stenosis_pct_ds"], sigma=geo["stenosis_sigma"], eccentric=geo["eccentric"],
    )

    gt = np.load(ground_truth_path)
    ckpt = torch.load(checkpoint_path, weights_only=False)

    model = PINN(
        in_dim=3, hidden_layers=ckpt["hidden_layers"], hidden_units=ckpt["hidden_units"],
        fourier_time_freqs=ckpt.get("fourier_time_freqs"),
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    nu = ckpt["nu"]

    t_arr = gt["t"]
    n_frames = len(t_arr)
    shape = gt["xp"].shape
    mask = gt["mask_p"]
    xp_flat = torch.tensor(gt["xp"].ravel(), dtype=torch.float32).reshape(-1, 1)
    yp_flat = torch.tensor(gt["yp"].ravel(), dtype=torch.float32).reshape(-1, 1)

    u_pinn = np.empty((n_frames,) + shape, dtype=np.float32)
    v_pinn = np.empty_like(u_pinn)
    errs = np.empty(n_frames)
    with torch.no_grad():
        for k in range(n_frames):
            t_flat = torch.full_like(xp_flat, float(t_arr[k]))
            u_k, v_k, _ = model(xp_flat, yp_flat, t_flat)
            u_pinn[k] = u_k.numpy().reshape(shape)
            v_pinn[k] = v_k.numpy().reshape(shape)
            mag_pinn = np.hypot(u_pinn[k], v_pinn[k])
            mag_gt = np.hypot(gt["u"][k], gt["v"][k])
            errs[k] = relative_l2_error(mag_pinn, mag_gt, mask)

    print(f"Field relative L2 error over the cycle: mean={errs.mean():.3%}  max={errs.max():.3%}")

    x1d = gt["xp"][0, :]
    y1d = gt["yp"][:, 0]
    throat_col = int(np.argmin(np.abs(x1d - params.x0)))
    x_throat = float(x1d[throat_col])
    y_throat = float(stenosis_height(np.array([x_throat]), params)[0])  # exact analytic wall, matches Phase 1

    wss_gt_t = compute_wss_gt_time_trace(gt, throat_col, y1d)
    wss_pinn_t = compute_wss_pinn_time_trace(model, t_arr, x_throat, y_throat, nu)
    wss_err = float(np.linalg.norm(wss_pinn_t - wss_gt_t) / np.linalg.norm(wss_gt_t))
    print(f"Throat wall shear stress relative L2 error (over the cycle): {wss_err:.3%}")

    figures = Path(figures_dir)
    figures.mkdir(parents=True, exist_ok=True)
    plot_wss_time_trace(t_arr, wss_gt_t, wss_pinn_t, figures / "phase2_wss_time_trace.png")
    animate_field_comparison(
        gt["xp"], gt["yp"], mask, t_arr, gt["u"], gt["v"], u_pinn, v_pinn,
        figures / "phase2_field_animation.gif",
    )
    plot_training_curves_phase2(ckpt["history"], figures / "phase2_training_curves.png")
    print("Saved figures to", figures)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase2_pulsatile.yaml")
    parser.add_argument("--ground-truth", default="data/ground_truth/phase2_pulsatile.npz")
    parser.add_argument("--checkpoint", default="data/checkpoints/phase2_pinn.pt")
    parser.add_argument("--figures-dir", default="article/figures")
    args = parser.parse_args()
    main(args.config, args.ground_truth, args.checkpoint, args.figures_dir)
