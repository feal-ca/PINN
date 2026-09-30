"""Figures for Part 1, drawn from one trained run (default: the final one).

    python scripts/make_figures_v2.py --run data/screen_long/R8_40sensors

Writes hero.png, field_comparison.png, vorticity.png, wss.png and
viscosity_paths.png into --figures-dir. Every number that appears on a
figure comes from the same evaluation code the article quotes.
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

from pinn_bloodflow.evaluation import evaluate  # noqa: E402
from pinn_bloodflow.geometry import StenosisParams, stenosis_height  # noqa: E402
from pinn_bloodflow.pinn.train import build_model  # noqa: E402

INK, PINN_C, TRUE_C = "#1d2733", "#d55e00", "#0072b2"


def load(run: Path, config: str, gt_path: str):
    cfg = yaml.safe_load(open(config))
    g = cfg["geometry"]
    params = StenosisParams(H=g["H"], Lx=g["Lx"], x0=g["stenosis_x0"], pct_ds=g["stenosis_pct_ds"],
                            sigma=g["stenosis_sigma"], eccentric=g["eccentric"])
    ck = torch.load(run / "model.pt", weights_only=False)
    model = build_model(ck["cfg"], params)
    model.load_state_dict(ck["model_state"])
    model.eval()
    gt = np.load(gt_path)
    metrics, ex = evaluate(model, ck["log_nu"].item(), gt, params)
    return params, gt, ck, model, metrics, ex


def hero(params, gt, ck, ex, out):
    """Text-free hero: the PINN's speed field with faint streamlines, the wall
    glowing where the inferred shear is highest, and the sensors in view."""
    bg = "#0b1220"
    mask = gt["mask_p"]
    xg, yg = gt["xp"][0], gt["yp"][:, 0]
    mag = np.where(mask, np.hypot(ex["u_p"], ex["v_p"]), np.nan)
    cmap = LinearSegmentedColormap.from_list("hero", [bg, "#16305c", "#1f7a8c", "#5fc9a3", "#f4d35e", "#fffdf2"])
    x_lo, x_hi, y_lo, y_hi = 0.9, 3.9, -0.02, 1.02
    fig = plt.figure(figsize=(14, 6.1), dpi=100, facecolor=bg)
    ax = fig.add_axes([0, 0, 1, 1], facecolor=bg)
    ax.imshow(mag, origin="lower", extent=[xg[0], xg[-1], yg[0], yg[-1]], cmap=cmap, vmin=0, vmax=2.5,
              aspect="auto", interpolation="bilinear", zorder=1)
    # faint streamlines from the PINN's own velocity field
    u = np.where(mask, ex["u_p"], 0.0)
    v = np.where(mask, ex["v_p"], 0.0)
    ax.streamplot(xg, yg, u, v, density=[2.0, 1.5], color=(1, 1, 1, 0.28), linewidth=0.8, arrowsize=0.01,
                  zorder=2, broken_streamlines=False)
    # the solid, and a glow on the wall that follows the inferred shear stress
    xw = np.linspace(x_lo - 0.2, x_hi + 0.2, 900)
    hw = stenosis_height(xw, params)
    ax.fill_between(xw, -0.1, hw, color=bg, zorder=4)
    from pinn_bloodflow.evaluation import wall_stations
    xs = ex["xs"]
    shear = np.clip(np.interp(xw, xs, ex["wss_pred"]), 0, None)
    shear = shear / shear.max()
    pts = np.column_stack([xw, hw]).reshape(-1, 1, 2)
    seg = np.concatenate([pts[:-1], pts[1:]], axis=1)
    for width, alpha in [(34, 0.10), (22, 0.16), (12, 0.35), (5, 1.0)]:
        lc = LineCollection(seg, cmap="magma", array=shear[:-1], linewidths=width * (0.25 + 0.75 * shear[:-1]),
                            zorder=5, capstyle="round", joinstyle="round", alpha=alpha)
        lc.set_clim(0, 1.0)
        ax.add_collection(lc)
    sx, sy = ck["sensors"]["x"], ck["sensors"]["y"]
    inside = (sx > x_lo + 0.05) & (sx < x_hi - 0.05)
    ax.scatter(sx[inside], sy[inside], s=120, facecolors="none", edgecolors="#ff5a5f", linewidths=2.2, zorder=6)
    ax.scatter(sx[inside], sy[inside], s=12, c="#ff5a5f", zorder=6)
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    ax.axis("off")
    fig.savefig(out, facecolor=bg, dpi=200)
    plt.close(fig)


def _panel(ax, field, gt, ext, vmin, vmax, cmap, title):
    im = ax.imshow(np.ma.array(field, mask=~gt["mask_p"]), origin="lower", extent=ext, vmin=vmin, vmax=vmax, cmap=cmap,
                   aspect="auto", interpolation="bilinear")
    ax.set_title(title, loc="left", fontsize=12)
    ax.set_ylabel("y / H")
    return im


def field_comparison(params, gt, ck, ex, out):
    ext = [0, params.Lx, 0, params.H]
    fig, ax = plt.subplots(2, 1, figsize=(10, 4.6), sharex=True, constrained_layout=True)
    mag_t = np.hypot(gt["u"], gt["v"])
    mag_p = np.hypot(ex["u_p"], ex["v_p"])
    im = _panel(ax[0], mag_t, gt, ext, 0, 2.5, "viridis", "CFD reference")
    _panel(ax[1], mag_p, gt, ext, 0, 2.5, "viridis", "PINN")
    ax[1].scatter(ck["sensors"]["x"], ck["sensors"]["y"], s=34, facecolors="none", edgecolors="white", linewidths=1.6)
    ax[1].scatter(ck["sensors"]["x"], ck["sensors"]["y"], s=5, c="white")
    ax[1].set_xlabel("x / H")
    fig.colorbar(im, ax=ax, label="velocity magnitude (U0)", shrink=0.9, pad=0.01)
    fig.savefig(out, dpi=170)
    plt.close(fig)


def vorticity_figure(params, gt, ck, ex, out):
    ext = [0, params.Lx, 0, params.H]
    lim = 20
    fig, ax = plt.subplots(2, 1, figsize=(10, 4.6), sharex=True, constrained_layout=True)
    im = _panel(ax[0], ex["w_t"], gt, ext, -lim, lim, "RdBu_r", "CFD reference")
    _panel(ax[1], ex["w_p"], gt, ext, -lim, lim, "RdBu_r", "PINN")
    ax[1].set_xlabel("x / H")
    fig.colorbar(im, ax=ax, label="vorticity (U0 / H)", shrink=0.9, pad=0.01)
    fig.savefig(out, dpi=170)
    plt.close(fig)


def wss_figure(params, ex, nu_true, out):
    with plt.rc_context({"font.size": 12, "axes.spines.top": False, "axes.spines.right": False}):
        fig, ax = plt.subplots(figsize=(8, 3.6))
        ax.plot(ex["xs"], ex["wss_true"], color=INK, lw=2.2, label="CFD reference")
        ax.plot(ex["xs"], ex["wss_pred"], color=PINN_C, lw=2.2, ls="--", label="PINN")
        ax.axhline(0, color="0.6", lw=0.8)
        ax.set(xlabel="x / H", ylabel="wall shear stress", xlim=(0.2, 8.0))
        ax.legend(frameon=False)
        fig.savefig(out, dpi=200, bbox_inches="tight")
        plt.close(fig)


def viscosity_paths(runs: dict[str, Path], nu_true: float, out):
    with plt.rc_context({"font.size": 12, "axes.spines.top": False, "axes.spines.right": False}):
        fig, ax = plt.subplots(figsize=(8, 3.6))
        colors = [PINN_C, TRUE_C, "#009e73", "#cc79a7"]
        for (label, run), c in zip(runs.items(), colors):
            h = torch.load(run / "model.pt", weights_only=False)["history"]
            ax.plot(h["epoch"], np.array(h["nu"]) / nu_true, color=c, lw=2, label=label)
        ax.axhline(1.0, color=INK, lw=1, ls=":")
        ax.set(xlabel="epoch", ylabel="viscosity / true viscosity", ylim=(0, 4.3))
        ax.legend(frameon=False, loc="upper right")
        fig.savefig(out, dpi=200, bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="data/screen_long/R8_40sensors")
    ap.add_argument("--config", default="configs/phase1_long.yaml")
    ap.add_argument("--ground-truth", default="data/ground_truth/phase1_long.npz")
    ap.add_argument("--figures-dir", default="article/figures")
    ap.add_argument("--paths", nargs="*", default=[], help="label=run_dir pairs for the viscosity trajectories")
    a = ap.parse_args()
    params, gt, ck, model, metrics, ex = load(Path(a.run), a.config, a.ground_truth)
    out = Path(a.figures_dir)
    out.mkdir(parents=True, exist_ok=True)
    hero(params, gt, ck, ex, out / "hero.png")
    field_comparison(params, gt, ck, ex, out / "field_comparison.png")
    vorticity_figure(params, gt, ck, ex, out / "vorticity.png")
    wss_figure(params, ex, float(gt["nu"]), out / "wss.png")
    if a.paths:
        viscosity_paths({p.split("=")[0]: Path(p.split("=")[1]) for p in a.paths}, float(gt["nu"]), out / "viscosity_paths.png")
    print(vars(metrics))
