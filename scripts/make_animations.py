"""Tracer-flow comparison GIF for the steady-flow PINN, plus the helpers the
explainer video (scripts/make_explainer.py) shares.

    python scripts/make_animations.py

The flow is steady, so what moves are tracer particles carried by the velocity
field: the CFD field on one side, the PINN's on the other.
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
from matplotlib.animation import FFMpegWriter, FuncAnimation
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, to_rgba
from scipy.interpolate import RegularGridInterpolator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pinn_bloodflow.evaluation import evaluate  # noqa: E402
from pinn_bloodflow.geometry import StenosisParams, stenosis_height  # noqa: E402
from pinn_bloodflow.pinn.train import build_model  # noqa: E402

BG, PANEL, SOLID = "#0b1220", "#101a2e", "#2c3a57"
TEXT, DIM = "#f2f5f9", "#9fb0c8"
CFD_C, PINN_C, SENSOR_C = "#8ab6ff", "#ff9a3d", "#ff5a5f"
CMAP = LinearSegmentedColormap.from_list("flow", ["#101a2e", "#1b3a6b", "#2a9d8f", "#f4d35e", "#ffffff"])


# ---------------------------------------------------------------- data ----
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
    return params, gt, ck, metrics, ex


class Tracers:
    """Particles advected by a steady velocity field, with short fading trails.

    Dead particles (left the channel, hit the wall, or grew old) reappear either
    at the inlet or at a random spot in the fluid, so the recirculation zone
    behind the bump stays populated instead of emptying out."""

    def __init__(self, u, v, gt, params, n=1100, trail=14, seed=0):
        xg, yg = gt["xp"][0], gt["yp"][:, 0]
        opts = dict(bounds_error=False, fill_value=0.0)
        self.fu = RegularGridInterpolator((yg, xg), u, **opts)
        self.fv = RegularGridInterpolator((yg, xg), v, **opts)
        self.fm = RegularGridInterpolator((yg, xg), gt["mask_p"].astype(float), method="nearest", **opts)
        self.params, self.n, self.trail = params, n, trail
        self.rng = np.random.default_rng(seed)
        self.pos = self._spawn(n, anywhere=True)
        self.age = self.rng.uniform(0, 4, n)
        self.life = self.rng.uniform(2.5, 6.0, n)
        self.hist = np.repeat(self.pos[:, None, :], trail, axis=1)

    def _spawn(self, k, anywhere):
        p = self.params
        out = np.empty((k, 2))
        filled = 0
        while filled < k:
            m = (k - filled) * 3
            x = self.rng.uniform(0, p.Lx, m) if anywhere else np.full(m, 0.02)
            y = self.rng.uniform(0.03, 0.97, m)
            ok = self.fm(np.stack([y, x], 1)) > 0.5
            take = min(ok.sum(), k - filled)
            out[filled:filled + take] = np.stack([x[ok], y[ok]], 1)[:take]
            filled += take
        return out

    def _vel(self, p):
        q = np.stack([p[:, 1], p[:, 0]], 1)
        return np.stack([self.fu(q), self.fv(q)], 1)

    def step(self, dt):
        k1 = self._vel(self.pos)
        k2 = self._vel(self.pos + 0.5 * dt * k1)
        self.pos = self.pos + dt * k2
        self.age += dt
        x, y = self.pos[:, 0], self.pos[:, 1]
        inside = self.fm(np.stack([y, x], 1)) > 0.5
        dead = (~inside) | (x > self.params.Lx - 0.02) | (x < 0.0) | (self.age > self.life)
        if dead.any():
            k = int(dead.sum())
            fresh = np.where(self.rng.random(k)[:, None] < 0.5, self._spawn(k, False), self._spawn(k, True))
            self.pos[dead] = fresh
            self.age[dead] = 0.0
            self.life[dead] = self.rng.uniform(2.5, 6.0, k)
            self.hist[dead] = fresh[:, None, :]
        self.hist = np.concatenate([self.hist[:, 1:], self.pos[:, None, :]], axis=1)

    def segments(self):
        h = self.hist
        segs = np.stack([h[:, :-1], h[:, 1:]], axis=2).reshape(-1, 2, 2)
        # trail fades toward its tail; a jump (respawn) is hidden by zero alpha
        fade = np.tile(np.linspace(0.0, 1.0, self.trail - 1), self.n)
        jump = np.linalg.norm(h[:, 1:] - h[:, :-1], axis=2).ravel() > 0.5
        return segs, np.where(jump, 0.0, fade)


# ------------------------------------------------------------- drawing ----
def channel_axes(fig, rect, params, title=None):
    ax = fig.add_axes(rect, facecolor=PANEL)
    ax.set_xlim(0, params.Lx)
    ax.set_ylim(0, params.H)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    xs = np.linspace(0, params.Lx, 600)
    ax.fill_between(xs, 0, stenosis_height(xs, params), color=SOLID, zorder=5)
    return ax


def speed_image(ax, gt, mag):
    data = np.where(gt["mask_p"], mag, np.nan)
    im = ax.imshow(data, origin="lower", extent=[0, float(gt["Lx"]), 0, float(gt["H"])], cmap=CMAP,
                   vmin=0, vmax=2.5, aspect="auto", interpolation="bilinear", zorder=1)
    return im


def add_tracers(ax, tr):
    lc = LineCollection([], linewidths=1.3, zorder=4, capstyle="round")
    ax.add_collection(lc)
    return lc


def update_tracers(lc, tr, alpha=1.0, color="#ffffff"):
    segs, fade = tr.segments()
    lc.set_segments(segs)
    rgba = np.tile(to_rgba(color), (len(fade), 1))
    rgba[:, 3] = 0.85 * fade * alpha
    lc.set_colors(rgba)


# --------------------------------------------------------------- GIF ------
def make_gif(params, gt, ck, ex, out: Path, seconds=6, fps=12, width_px=1150):
    """Two stacked panels, few tracers, rendered at width_px and palette-optimised
    with ffmpeg so the file stays small."""
    import subprocess
    import tempfile

    w_in = 13.0
    fig = plt.figure(figsize=(w_in, 4.9), dpi=100, facecolor=BG)
    axes, lcs = [], []
    trs = [Tracers(gt["u"], gt["v"], gt, params, n=170, trail=16, seed=1),
           Tracers(ex["u_p"], ex["v_p"], gt, params, n=170, trail=16, seed=1)]
    for rect, mag_field, label, color in [
        ([0.015, 0.545, 0.97, 0.36], np.hypot(gt["u"], gt["v"]), "CFD reference", CFD_C),
        ([0.015, 0.07, 0.97, 0.36], np.hypot(ex["u_p"], ex["v_p"]), "PINN", PINN_C),
    ]:
        ax = channel_axes(fig, rect, params)
        speed_image(ax, gt, mag_field)
        axes.append(ax)
        fig.text(rect[0] + 0.004, rect[1] + rect[3] + 0.022, label, color=color, fontsize=15, fontweight="bold")
    lcs = [add_tracers(a, t) for a, t in zip(axes, trs)]
    for lc in lcs:
        lc.set_linewidths(1.7)
    sx, sy = ck["sensors"]["x"], ck["sensors"]["y"]
    axes[1].scatter(sx, sy, s=34, facecolors="none", edgecolors=SENSOR_C, linewidths=1.5, zorder=6)
    for _ in range(150):
        for t in trs:
            t.step(0.05)

    def frame(i):
        for t, lc in zip(trs, lcs):
            t.step(0.07)
            update_tracers(lc, t)
        return lcs

    with tempfile.TemporaryDirectory() as tmp:
        mp4 = Path(tmp) / "loop.mp4"
        anim = FuncAnimation(fig, frame, frames=seconds * fps, blit=False)
        anim.save(mp4, writer=FFMpegWriter(fps=fps, codec="libx264", extra_args=["-pix_fmt", "yuv420p", "-crf", "12"]),
                  dpi=100)
        vf = (f"fps={fps},scale={width_px}:-2:flags=lanczos,split[a][b];"
              "[a]palettegen=max_colors=80:stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=4")
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(mp4), "-vf", vf, str(out)], check=True)
    plt.close(fig)
    print("saved", out, f"{out.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="data/experiments/final")
    ap.add_argument("--config", default="configs/phase1_long.yaml")
    ap.add_argument("--ground-truth", default="data/ground_truth/phase1_long.npz")
    ap.add_argument("--out-dir", default="article/figures")
    a = ap.parse_args()
    params, gt, ck, metrics, ex = load(Path(a.run), a.config, a.ground_truth)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    make_gif(params, gt, ck, ex, out / "flow_comparison.gif")
