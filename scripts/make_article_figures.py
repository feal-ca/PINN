"""Hero image and the cancelling-errors figure for Part 1 of the article.

Both are drawn from the trained Phase 1 checkpoint and the same WSS functions
evaluate.py uses, so every number on them matches the script output quoted in
the text.
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import evaluate as ev  # noqa: E402
from pinn_bloodflow.geometry import StenosisParams, stenosis_height  # noqa: E402
from pinn_bloodflow.pinn.network import PINN  # noqa: E402

INK, PINN_C, TRUE_C = "#1d2733", "#d55e00", "#0072b2"


def load(config, ground_truth, sensors, checkpoint):
    cfg = yaml.safe_load(open(config))
    g = cfg["geometry"]
    params = StenosisParams(
        H=g["H"], Lx=g["Lx"], x0=g["stenosis_x0"], pct_ds=g["stenosis_pct_ds"],
        sigma=g["stenosis_sigma"], eccentric=g["eccentric"],
    )
    gt = np.load(ground_truth)
    sens = np.load(sensors)
    ck = torch.load(checkpoint, weights_only=False)
    model = PINN(in_dim=2, hidden_layers=ck["hidden_layers"], hidden_units=ck["hidden_units"])
    model.load_state_dict(ck["model_state"])
    model.eval()
    return params, gt, sens, ck, model


def hero(params, gt, sens, ck, model, out_path):
    nu_rec = torch.exp(ck["log_nu"]).item()
    xf = torch.tensor(gt["xp"].ravel(), dtype=torch.float32).reshape(-1, 1)
    yf = torch.tensor(gt["yp"].ravel(), dtype=torch.float32).reshape(-1, 1)
    with torch.no_grad():
        u, v, _ = model(xf, yf)
    shape = gt["xp"].shape
    mag = np.where(gt["mask_p"], np.hypot(u.numpy().reshape(shape), v.numpy().reshape(shape)), np.nan)
    xw, _ = ev.compute_wss_ground_truth(gt, params)
    wss = ev.compute_wss_pinn(model, xw, params, nu_rec)
    hw = stenosis_height(xw, params)

    bg = "#0b1220"
    cmap = LinearSegmentedColormap.from_list("hero", [bg, "#1b3a6b", "#2a9d8f", "#f4d35e", "#ffffff"])
    # No text on the image: it is cropped to the stenosis so the field, the wall
    # glow and the sensors fill a roughly 2.3:1 frame.
    x_lo, x_hi, y_lo, y_hi = 0.7, 3.5, -0.03, 1.03
    fig = plt.figure(figsize=(14, 6.1), dpi=100, facecolor=bg)
    ax = fig.add_axes([0, 0, 1, 1], facecolor=bg)
    ax.pcolormesh(gt["xp"], gt["yp"], mag, cmap=cmap, shading="auto", vmin=0,
                  vmax=np.nanmax(mag), rasterized=True)
    ax.fill_between(xw, 0, hw, color=bg, zorder=3)
    pts = np.column_stack([xw, hw]).reshape(-1, 1, 2)
    seg = np.concatenate([pts[:-1], pts[1:]], axis=1)
    w = np.clip(np.nan_to_num(wss[:-1]), 0, None)
    ax.add_collection(LineCollection(seg, cmap="magma", array=w, linewidths=3 + 16 * w / w.max(),
                                     zorder=4, capstyle="round"))
    inside = (sens["x"] > x_lo) & (sens["x"] < x_hi)
    ax.scatter(sens["x"][inside], sens["y"][inside], s=170, facecolors="none", edgecolors="#ff5a5f",
               linewidths=3, zorder=5)
    ax.scatter(sens["x"][inside], sens["y"][inside], s=18, c="#ff5a5f", zorder=5)
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    ax.axis("off")
    fig.savefig(out_path, facecolor=bg)
    plt.close(fig)


def cancelling_errors(params, gt, sens, ck, model, out_path):
    nu_rec = torch.exp(ck["log_nu"]).item()
    nu_true = float(ck["nu_true"])
    xw, wss_gt = ev.compute_wss_ground_truth(gt, params)
    wss_pinn = ev.compute_wss_pinn(model, xw, params, nu_rec)
    wss_true_nu = ev.compute_wss_pinn(model, xw, params, nu_true)
    ok = ~np.isnan(wss_gt)
    grad_gt = wss_gt / nu_true
    grad_pinn = wss_pinn / nu_rec
    i = np.nanargmax(grad_gt)  # ground truth's peak station, where the 0.766 ratio is taken

    with plt.rc_context({"font.size": 13, "axes.spines.top": False, "axes.spines.right": False}):
        fig = plt.figure(figsize=(8, 7.6))
        gs = fig.add_gridspec(2, 2, width_ratios=[2.4, 1], height_ratios=[1, 1.15], hspace=0.5, wspace=0.3)
        a0, a1, a2 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])
        xl = (1.2, 2.4)

        a0.plot(xw[ok], grad_gt[ok], color=INK, lw=1.5, label="truth (raw CFD)")
        a0.plot(xw[ok], grad_pinn[ok], color=PINN_C, lw=2.4, ls="--", label="PINN")
        a0.set(xlim=xl, ylim=(0, None), xlabel="x / H", ylabel=r"$\partial u/\partial y$")
        a0.set_title("1. Wall velocity gradient", loc="left", fontweight="bold")
        a0.legend(frameon=False, loc="upper left", fontsize=11)
        a0.annotate(f"PINN reads {grad_pinn[i] / grad_gt[i]:.0%}\nof truth at the peak",
                    xy=(xw[i], grad_pinn[i]), xytext=(1.22, 0.55 * grad_gt[i]), color=PINN_C,
                    fontsize=12, arrowprops=dict(arrowstyle="->", color=PINN_C))

        a1.bar([0, 1], [nu_true, nu_rec], color=[INK, PINN_C], width=0.6)
        a1.set_xticks([0, 1], ["truth", "PINN"])
        a1.set_ylim(0, nu_rec * 1.3)
        a1.set_title("2. Viscosity", loc="left", fontweight="bold")
        a1.text(1, nu_rec * 1.02, f"+{nu_rec / nu_true - 1:.0%}", ha="center", color=PINN_C, fontweight="bold")

        a2.plot(xw[ok], wss_gt[ok], color=INK, lw=1.5, label="truth")
        a2.plot(xw[ok], wss_pinn[ok], color=PINN_C, lw=2.4, ls="--", label="PINN, recovered viscosity")
        a2.plot(xw[ok], wss_true_nu[ok], color=TRUE_C, lw=2.4, ls=":", label="PINN, true viscosity")
        a2.set(xlim=xl, ylim=(0, None), xlabel="x / H", ylabel=r"$\nu\,\partial u/\partial y$")
        a2.set_title("3. Wall shear stress = viscosity x gradient", loc="left", fontweight="bold")
        a2.legend(frameon=False, loc="upper left", fontsize=11)
        a2.annotate(f"peak {np.nanmax(wss_pinn):.3f}: looks fine\n(truth {np.nanmax(wss_gt):.3f})",
                    xy=(xw[np.nanargmax(wss_pinn)], np.nanmax(wss_pinn)), xytext=(2.05, 0.28),
                    color=PINN_C, fontsize=12, arrowprops=dict(arrowstyle="->", color=PINN_C))
        a2.annotate(f"peak {np.nanmax(wss_true_nu):.3f}", xy=(xw[np.nanargmax(wss_true_nu)], np.nanmax(wss_true_nu)),
                    xytext=(2.05, 0.14), color=TRUE_C, fontsize=12,
                    arrowprops=dict(arrowstyle="->", color=TRUE_C))
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
    print(f"gradient ratio at truth peak {grad_pinn[i] / grad_gt[i]:.3f}, viscosity {nu_rec / nu_true - 1:+.1%}, "
          f"peaks: truth {np.nanmax(wss_gt):.4f} pinn {np.nanmax(wss_pinn):.4f} pinn@true nu {np.nanmax(wss_true_nu):.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase1_steady.yaml")
    parser.add_argument("--ground-truth", default="data/ground_truth/phase1_steady.npz")
    parser.add_argument("--sensors", default="data/sensors/phase1_sensors.npz")
    parser.add_argument("--checkpoint", default="data/checkpoints/phase1_pinn.pt")
    parser.add_argument("--figures-dir", default="article/figures")
    args = parser.parse_args()
    loaded = load(args.config, args.ground_truth, args.sensors, args.checkpoint)
    out = Path(args.figures_dir)
    hero(*loaded, out / "phase1_hero.png")
    cancelling_errors(*loaded, out / "phase1_cancelling_errors.png")
    print("Saved hero and cancelling-errors figures to", out)
