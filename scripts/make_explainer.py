"""~33 s portrait explainer (1080x1350, no audio) for social media.

    python scripts/make_explainer.py                     # full video
    python scripts/make_explainer.py --still 15 --out x.png   # one frame at t = 15 s

Needs a run trained with `snapshot_epochs` (scripts/experiment.py), because the
middle scene replays the network's flow field and the loss curves as it learned.
Scenes: the question, the readings, the physics, training, comparison with the
CFD reference, wall shear stress, end card.
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import FFMpegWriter, FuncAnimation
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from make_animations import (  # noqa: E402
    BG, CFD_C, CMAP, DIM, PANEL, PINN_C, SENSOR_C, SOLID, TEXT, Tracers, add_tracers, channel_axes, load,
    speed_image, update_tracers,
)
from pinn_bloodflow.geometry import stenosis_height  # noqa: E402
from pinn_bloodflow.pinn.train import build_model  # noqa: E402

FPS = 30
SCENES = {"A": (0.0, 3.4), "B": (3.4, 7.2), "C": (7.2, 10.4), "D": (10.4, 19.6), "E": (19.6, 25.6),
          "F": (25.6, 30.4), "G": (30.4, 33.6)}
DURATION = SCENES["G"][1]
CAPTIONS = {
    "A": "How hard does blood drag\non this artery wall?",
    "B": "All we get is a few noisy\nvelocity readings",
    "C": "Plus a rule: the flow has\nto obey physics",
    "D": "A neural network learns\nfrom both",
    "E": "Now, against a CFD run\nit never saw",
    "F": "And the drag on the\nwall comes out too",
    "G": "A few readings plus physics\ngive the whole picture",
}
W_LOSS = {"data": 100.0, "pde": 10.0, "bc": 10.0}


def smooth(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


def ramp(t, a, b):
    return smooth((t - a) / (b - a))


def scene_at(t):
    for k, (a, b) in SCENES.items():
        if a <= t < b:
            return k
    return "G"


def snapshot_fields(ck, params, gt, step=2):
    """Speed magnitude of the network at every saved snapshot, on a coarse grid."""
    xp, yp = gt["xp"][::step, ::step], gt["yp"][::step, ::step]
    x = torch.tensor(xp.ravel(), dtype=torch.float32).reshape(-1, 1)
    y = torch.tensor(yp.ravel(), dtype=torch.float32).reshape(-1, 1)
    model = build_model(ck["cfg"], params)
    fields = []
    for epoch, state, nu in ck["history"]["snapshots"]:
        model.load_state_dict(state)
        with torch.no_grad():
            u, v, _ = model(x, y)
        fields.append((epoch, np.hypot(u.numpy(), v.numpy()).reshape(xp.shape), nu))
    return fields


def load_history(run: Path):
    path = run / "model.pt"
    if path.exists():
        return torch.load(path, weights_only=False)
    part = torch.load(run / "partial.pt", weights_only=False)      # a run still in progress
    h = part["history"]
    hist = {k: getattr(h, k) for k in ("epoch", "total", "data", "pde", "bc", "nu", "snapshots")}
    ck = {"history": hist, "cfg": part.get("cfg"), "log_nu": part["log_nu"], "model_state": part["model"]}
    return ck


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="data/experiments/final")
    ap.add_argument("--config", default="configs/phase1_long.yaml")
    ap.add_argument("--ground-truth", default="data/ground_truth/phase1_long.npz")
    ap.add_argument("--out", default="article/figures/pinn_explainer.mp4")
    ap.add_argument("--still", type=float, default=None)
    a = ap.parse_args()

    run = Path(a.run)
    if (run / "model.pt").exists():
        params, gt, ck, metrics, ex = load(run, a.config, a.ground_truth)
    else:                                   # test path: use the in-progress checkpoint, final field from last weights
        import yaml
        from pinn_bloodflow.evaluation import evaluate
        from pinn_bloodflow.geometry import StenosisParams
        g = yaml.safe_load(open(a.config))["geometry"]
        params = StenosisParams(H=g["H"], Lx=g["Lx"], x0=g["stenosis_x0"], pct_ds=g["stenosis_pct_ds"],
                                sigma=g["stenosis_sigma"], eccentric=g["eccentric"])
        gt = np.load(a.ground_truth)
        part = torch.load(run / "partial.pt", weights_only=False)
        h = part["history"]
        base = torch.load("data/screen_long/R8_40sensors/model.pt", weights_only=False)
        ck = dict(base)
        ck["history"] = {k: getattr(h, k) for k in ("epoch", "total", "data", "pde", "bc", "nu", "snapshots")}
        model = build_model(base["cfg"], params)
        model.load_state_dict(base["model_state"])
        model.eval()
        metrics, ex = evaluate(model, base["log_nu"].item(), gt, params)
    hist = ck["history"]
    nu_true = float(gt["nu"])
    snaps = snapshot_fields(ck, params, gt)
    n_snap = len(snaps)
    snap_epochs = np.array([s[0] for s in snaps])
    snap_nu = np.array([s[2] for s in snaps]) / nu_true
    h_ep = np.array(hist["epoch"], dtype=float)
    terms = {k: np.interp(snap_epochs, h_ep, np.array(hist[k])) * W_LOSS[k] for k in ("data", "pde", "bc")}
    Lx = params.Lx
    xg = gt["xp"][0]
    mag_c = np.hypot(gt["u"], gt["v"])
    mag_p = np.hypot(ex["u_p"], ex["v_p"])
    mask = gt["mask_p"]
    err = np.where(mask, np.abs(mag_p - mag_c), np.nan)
    sx, sy = ck["sensors"]["x"], ck["sensors"]["y"]
    su, sv = ck["sensors"]["u_meas"], ck["sensors"]["v_meas"]

    W, H = 10.8, 13.5
    fig = plt.figure(figsize=(W, H), dpi=100, facecolor=BG)
    pw = 0.92
    left = 0.04

    def strip_h(x_span):
        return pw * W / x_span / H

    # ---------------------------------------------------------- axes ---
    zoom_span = (1.1, 3.5)
    ax_zoom = channel_axes(fig, [left, 0.34, pw, strip_h(zoom_span[1] - zoom_span[0])], params)
    ax_zoom.set_xlim(*zoom_span)
    im_zoom = speed_image(ax_zoom, gt, mag_p)
    im_zoom.set_alpha(1.0)
    pinn_masked = np.where(mask, mag_p, np.nan)
    xw = np.linspace(0, Lx, 900)
    wall_glow, = ax_zoom.plot(xw, stenosis_height(xw, params) + 0.008, color="#ffd166", lw=6, zorder=6, alpha=0.0,
                              solid_capstyle="round")
    rings = ax_zoom.scatter(sx, sy, s=0, facecolors="none", edgecolors=SENSOR_C, linewidths=3, zorder=7)
    dots = ax_zoom.scatter(sx, sy, s=0, c=SENSOR_C, zorder=7)
    quiv = ax_zoom.quiver(sx, sy, su, sv, color=SENSOR_C, angles="xy", scale_units="xy", scale=5.0, width=0.0045,
                          zorder=7, alpha=0.0)

    h_strip = strip_h(Lx)
    ax_strip = channel_axes(fig, [left, 0.705, pw, h_strip], params)
    im_strip = speed_image(ax_strip, gt, mag_p)
    ax_strip.scatter(sx, sy, s=70, facecolors="none", edgecolors=SENSOR_C, linewidths=1.6, zorder=7)
    ax_loss = fig.add_axes([0.12, 0.35, 0.83, 0.28], facecolor=BG)
    ax_visc = fig.add_axes([0.12, 0.105, 0.83, 0.16], facecolor=BG)
    lab_strip = fig.text(left, 0.705 + h_strip + 0.010, "", color=PINN_C, fontsize=24, fontweight="bold")

    e_span = (0.6, 5.0)
    eh = strip_h(e_span[1] - e_span[0])
    e_axes, e_labels = [], []
    for y0, name, color in [(0.62, "CFD simulation", CFD_C), (0.395, "PINN", PINN_C), (0.17, "Difference", DIM)]:
        ax = channel_axes(fig, [left, y0, pw, eh], params)
        ax.set_xlim(*e_span)
        e_axes.append(ax)
        e_labels.append(fig.text(left, y0 + eh + 0.008, name, color=color, fontsize=24, fontweight="bold"))
    speed_image(e_axes[0], gt, mag_c)
    speed_image(e_axes[1], gt, mag_p)
    e_axes[2].imshow(err, origin="lower", extent=[0, Lx, 0, params.H], cmap="magma", vmin=0, vmax=0.5, aspect="auto",
                     interpolation="bilinear", zorder=1)
    tr_c = Tracers(gt["u"], gt["v"], gt, params, n=230, trail=12, seed=3)
    tr_p = Tracers(ex["u_p"], ex["v_p"], gt, params, n=230, trail=12, seed=3)
    lc_c, lc_p = add_tracers(e_axes[0], tr_c), add_tracers(e_axes[1], tr_p)
    for lc in (lc_c, lc_p):
        lc.set_linewidths(2.0)
    for _ in range(150):
        tr_c.step(0.05)
        tr_p.step(0.05)

    ax_w = fig.add_axes([0.13, 0.24, 0.80, 0.42], facecolor=BG)
    for s in ("top", "right"):
        ax_w.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax_w.spines[s].set_color(DIM)
    ax_w.tick_params(colors=DIM, labelsize=17)
    ax_w.set_xlim(0.2, 8.0)
    ax_w.set_ylim(-0.06, 0.34)
    ax_w.set_xlabel("distance along the artery", color=DIM, fontsize=20)
    ax_w.set_ylabel("wall shear stress", color=DIM, fontsize=20)
    ax_w.axhline(0, color=DIM, lw=0.8, alpha=0.5)
    ln_c, = ax_w.plot([], [], color=CFD_C, lw=6, label="CFD")
    ln_p, = ax_w.plot([], [], color=PINN_C, lw=6, ls="--", label="PINN")
    leg = ax_w.legend(frameon=False, fontsize=24, loc="upper right", labelcolor=TEXT)

    hero_img = plt.imread(ROOT / "article/figures/hero.png")
    ax_hero = fig.add_axes([left, 0.30, pw, pw * W / (hero_img.shape[1] / hero_img.shape[0]) / H])
    ax_hero.imshow(hero_img)
    ax_hero.axis("off")

    cap = fig.text(0.5, 0.905, "", color=TEXT, fontsize=38, ha="center", va="center", fontweight="bold",
                   linespacing=1.25)
    tail = fig.text(0.5, 0.20, "Full write-up on Towards Data Science", color=DIM, fontsize=26, ha="center")
    veil = Rectangle((0, 0), 1, 1, transform=fig.transFigure, facecolor=BG, edgecolor="none", zorder=100, alpha=0.0)
    fig.add_artist(veil)

    boundaries = [b for _, b in list(SCENES.values())[:-1]]
    groups = {"A": [ax_zoom], "B": [ax_zoom], "C": [ax_zoom], "D": [ax_strip, ax_loss, ax_visc],
              "E": e_axes, "F": [ax_w], "G": [ax_hero]}
    all_axes = {id(x): x for v in groups.values() for x in v}
    tr_step = {"n": 0}

    def draw_loss(i):
        ax_loss.clear()
        ax_loss.set_facecolor(BG)
        ep = snap_epochs[: i + 1]
        stack = [terms["data"][: i + 1], terms["pde"][: i + 1], terms["bc"][: i + 1]]
        if i == 0:
            ep = np.array([0.0, 1.0])
            stack = [np.array([v[0], v[0]]) for v in stack]
        ax_loss.stackplot(ep, *stack, colors=["#ff5a5f", "#4cc9f0", "#f4d35e"], alpha=0.9,
                          labels=["fits the readings", "obeys the flow equations", "respects the walls"])
        total = np.sum(stack, axis=0)
        ax_loss.plot(ep, total, color="white", lw=2.6)
        ax_loss.set_yscale("log")
        ax_loss.set_xscale("symlog", linthresh=250)
        ax_loss.set_xlim(0, 10400)
        ax_loss.set_ylim(0.4, 500)
        ax_loss.set_xticks([0, 100, 1000, 10000])
        ax_loss.set_xticklabels([])
        ax_loss.set_ylabel("loss (total in white)", color=DIM, fontsize=18)
        ax_loss.tick_params(colors=DIM, labelsize=15)
        for s in ("top", "right"):
            ax_loss.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax_loss.spines[s].set_color(DIM)
        leg_l = ax_loss.legend(loc="upper right", frameon=False, fontsize=17, labelcolor=TEXT)
        return leg_l

    def draw_visc(i):
        ax_visc.clear()
        ax_visc.set_facecolor(BG)
        ax_visc.axhline(1.0, color="white", lw=1.6, ls=":")
        ax_visc.plot(snap_epochs[: i + 1], snap_nu[: i + 1], color=PINN_C, lw=3.6)
        ax_visc.set_xscale("symlog", linthresh=250)
        ax_visc.set_xlim(0, 10400)
        ax_visc.set_ylim(0, 4.3)
        ax_visc.set_xticks([0, 100, 1000, 10000])
        ax_visc.set_xticklabels(["0", "100", "1,000", "10,000"])
        ax_visc.set_xlabel("training epochs", color=DIM, fontsize=18)
        ax_visc.set_ylabel("viscosity estimate", color=DIM, fontsize=18)
        ax_visc.text(9000, 1.18, "true value", color="white", fontsize=15, ha="right")
        ax_visc.tick_params(colors=DIM, labelsize=15)
        for s in ("top", "right"):
            ax_visc.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax_visc.spines[s].set_color(DIM)

    def frame(fi):
        t = fi / FPS
        sc = scene_at(t)
        vis = set(id(x) for x in groups[sc])
        for k, ax in all_axes.items():
            ax.set_visible(k in vis)
        lab_strip.set_visible(sc == "D")
        for lab in e_labels:
            lab.set_visible(sc == "E")
        tail.set_visible(sc == "G")

        # A to C: the zoomed channel
        if sc in "ABC":
            wall_glow.set_alpha(0.95 * ramp(t, 0.5, 1.2) * (1 - ramp(t, 3.0, 3.4)) * (0.75 + 0.25 * np.sin(2 * np.pi * t)))
            t0 = 3.9 + np.arange(len(sx)) * 0.05
            grow = np.clip((t - t0) / 0.35, 0, 1)
            pulse = 1 + 0.5 * np.sin(grow * np.pi)
            rings.set_sizes(520 * smooth(grow) * pulse)
            dots.set_sizes(60 * smooth(grow))
            quiv.set_alpha(0.95 * ramp(t, 5.6, 6.6))
            front = zoom_span[0] + (zoom_span[1] - zoom_span[0]) * ramp(t, 7.8, 9.8)
            im_zoom.set_data(np.where(xg[None, :] <= front, pinn_masked, np.nan) if t >= 7.2 else np.full_like(pinn_masked, np.nan))
        # D: training replay
        if sc == "D":
            p = ramp(t, SCENES["D"][0] + 0.4, SCENES["D"][1] - 1.0)
            i = int(round(p * (n_snap - 1)))
            im_strip.set_data(np.where(mask[::2, ::2], snaps[i][1], np.nan))
            im_strip.set_extent([0, Lx, 0, params.H])
            lab_strip.set_text(f"PINN after {int(snap_epochs[i]):,} epochs")
            draw_loss(i)
            draw_visc(i)
        # E: comparison with tracers
        if sc == "E":
            tr_c.step(0.06)
            tr_p.step(0.06)
            update_tracers(lc_c, tr_c)
            update_tracers(lc_p, tr_p)
            for ax in e_axes:
                ax.set_xlim(*e_span)
        # F: wall shear stress
        if sc == "F":
            prog = ramp(t, SCENES["F"][0] + 0.5, SCENES["F"][1] - 1.2)
            n = max(2, int(prog * len(ex["xs"])))
            ln_c.set_data(ex["xs"][:n], ex["wss_true"][:n])
            ln_p.set_data(ex["xs"][:n], ex["wss_pred"][:n])
        cap.set_text(CAPTIONS[sc])
        # fade through the background at each scene change
        d = min(abs(t - b) for b in boundaries)
        veil.set_alpha(float(1 - smooth(d / 0.28)) if d < 0.28 else 0.0)
        if sc == "G":
            tail.set_alpha(ramp(t, 31.4, 32.2))
        return []

    if a.still is not None:
        frame(int(a.still * FPS))
        fig.savefig(a.out, facecolor=BG)
        print("saved", a.out)
        return
    anim = FuncAnimation(fig, frame, frames=int(DURATION * FPS), blit=False)
    writer = FFMpegWriter(fps=FPS, codec="libx264",
                          extra_args=["-pix_fmt", "yuv420p", "-crf", "17", "-preset", "medium", "-movflags", "+faststart"])
    anim.save(a.out, writer=writer, dpi=100)
    print("saved", a.out)


if __name__ == "__main__":
    main()
