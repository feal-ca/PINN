"""Static comparison/diagnostic plots for the Phase 1 PINN, plus the
Phase 2 animation and time-trace plots.
"""

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np


def plot_field_comparison(xp, yp, mask, u_gt, v_gt, u_pinn, v_pinn, sensor_x, sensor_y, out_path):
    mag_gt = np.where(mask, np.hypot(u_gt, v_gt), np.nan)
    mag_pinn = np.where(mask, np.hypot(u_pinn, v_pinn), np.nan)
    vmax = np.nanmax(mag_gt)

    with plt.rc_context({"font.size": 13}):
        fig, axes = plt.subplots(2, 1, figsize=(8, 5.2), sharex=True, sharey=True)
        for ax, mag, title in [(axes[0], mag_gt, "Ground truth (CFD)"), (axes[1], mag_pinn, "PINN reconstruction")]:
            im = ax.pcolormesh(xp, yp, mag, shading="auto", cmap="viridis", vmin=0, vmax=vmax)
            ax.set_title(title, fontsize=14)
            ax.set_ylabel("y / H")
            ax.set_aspect("equal")
            # Sensors go on both panels so the truth panel does not read as "unmeasured".
            ax.scatter(sensor_x, sensor_y, s=42, facecolors="none", edgecolors="white",
                       linewidths=1.6)
            ax.scatter(sensor_x, sensor_y, s=8, c="black", zorder=3)
            # Zero contour of signed u marks the edge of the recirculation zone.
            u_show = np.where(mask & (yp < 0.6), u_gt if ax is axes[0] else u_pinn, np.nan)
            ax.contour(xp, yp, u_show, levels=[0.0], colors="red", linewidths=1.6)
        axes[1].set_xlabel("x / H (flow left to right)")
        fig.colorbar(im, ax=axes, label="|velocity| / inlet velocity", shrink=0.8)
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)


def plot_error_heatmap(xp, yp, mask, u_gt, v_gt, u_pinn, v_pinn, out_path):
    err = np.where(mask, np.hypot(u_pinn - u_gt, v_pinn - v_gt), np.nan)
    mag_gt = np.hypot(u_gt, v_gt)
    # Relative to the local speed: absolute error flatters slow near-wall fluid.
    rel = np.where(mask, err / np.maximum(mag_gt, 0.05), np.nan)
    with plt.rc_context({"font.size": 13}):
        fig, axes = plt.subplots(2, 1, figsize=(8, 5.2), sharex=True, sharey=True)
        for ax, field, cmap, title, label, vmax in [
            (axes[0], err, "magma", "Absolute error", "|velocity error|", None),
            (axes[1], rel, "magma", "Error relative to local speed (speed floored at 0.05)", "relative error", 1.0),
        ]:
            im = ax.pcolormesh(xp, yp, field, shading="auto", cmap=cmap, vmin=0, vmax=vmax)
            ax.set_aspect("equal")
            ax.set_title(title, fontsize=14)
            ax.set_ylabel("y / H")
            fig.colorbar(im, ax=ax, label=label, shrink=0.9)
        axes[1].set_xlabel("x / H")
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)


def plot_wss_comparison(x, wss_gt, wss_pinn, out_path):
    valid = ~np.isnan(wss_gt)
    with plt.rc_context({"font.size": 13}):
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(x, wss_gt, label="ground truth (raw CFD)", linewidth=2, color="#1f2a37")
        ax.plot(x, wss_pinn, label="PINN", linewidth=2.4, linestyle="--", color="#d55e00")
        ax.axhline(0, color="grey", linewidth=0.8)
        i = np.nanargmax(wss_pinn)
        ax.annotate(f"PINN peak {wss_pinn[i]:.3f}\nCFD peak {np.nanmax(wss_gt):.3f}",
                    xy=(x[i], wss_pinn[i]), xytext=(x[i] - 1.7, wss_pinn[i] * 0.8),
                    fontsize=12, arrowprops=dict(arrowstyle="->"))
        ax.set_xlabel("x / H")
        ax.set_ylabel(r"$\nu\,\partial u/\partial y$ (nondimensional)")
        ax.set_title("Wall shear stress along the stenosed wall", fontsize=14)
        ax.legend(loc="upper right", fontsize=11)
        ax.spines[["top", "right"]].set_visible(False)
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)


def plot_wss_time_trace(t, wss_gt, wss_pinn, out_path):
    """WSS(t) at a single fixed location (the stenosis throat) -- the
    Phase 2 analogue of Phase 1's WSS(x) plot, and the headline pulsatile
    result: it shows the PINN reconstructing how peak wall shear stress
    tracks the cardiac cycle from sparse, noisy point-in-time sensors.
    """
    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.plot(t, wss_gt, label="ground truth", linewidth=2, marker="o", markersize=3)
    ax.plot(t, wss_pinn, label="PINN", linewidth=2, linestyle="--")
    ax.set_xlabel("t (cardiac cycle)")
    ax.set_ylabel(r"wall shear stress $\tau_w$ at throat")
    ax.set_title("Wall shear stress through the cardiac cycle")
    ax.legend()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def animate_field_comparison(xp, yp, mask, t, u_gt, v_gt, u_pinn, v_pinn, out_path, fps=10):
    """GT-vs-PINN |velocity| field, animated over one cardiac cycle.

    u_gt/v_gt/u_pinn/v_pinn have shape (n_frames, ny, nx). Saved via
    Pillow (no system ffmpeg dependency, matching the environment
    constraints noted in the project plan).
    """
    mag_gt = np.where(mask[None, :, :], np.hypot(u_gt, v_gt), np.nan)
    mag_pinn = np.where(mask[None, :, :], np.hypot(u_pinn, v_pinn), np.nan)
    vmax = np.nanmax(mag_gt)

    fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True, sharey=True)
    im0 = axes[0].pcolormesh(xp, yp, mag_gt[0], shading="auto", cmap="viridis", vmin=0, vmax=vmax)
    im1 = axes[1].pcolormesh(xp, yp, mag_pinn[0], shading="auto", cmap="viridis", vmin=0, vmax=vmax)
    axes[0].set_title("Ground truth (CFD)")
    axes[1].set_title("PINN reconstruction")
    for ax in axes:
        ax.set_ylabel("y")
        ax.set_aspect("equal")
    axes[1].set_xlabel("x")
    fig.colorbar(im0, ax=axes, label="|velocity|", shrink=0.8)
    time_text = axes[0].text(0.02, 0.85, "", transform=axes[0].transAxes, color="white", fontsize=9)

    def update(frame):
        im0.set_array(mag_gt[frame].ravel())
        im1.set_array(mag_pinn[frame].ravel())
        time_text.set_text(f"t = {t[frame]:.3f}")
        return im0, im1, time_text

    anim = animation.FuncAnimation(fig, update, frames=len(t), blit=False)
    anim.save(out_path, writer=animation.PillowWriter(fps=fps))
    plt.close(fig)


def plot_training_curves(history, nu_true, out_path):
    epoch = np.asarray(history["epoch"])
    nu = np.asarray(history["nu"])
    with plt.rc_context({"font.size": 13}):
        fig, axes = plt.subplots(2, 1, figsize=(7.5, 6.5))
        axes[0].semilogy(epoch, history["data"], label="data")
        axes[0].semilogy(epoch, history["pde"], label="pde")
        axes[0].semilogy(epoch, history["bc"], label="bc")
        axes[0].set_ylabel("unweighted loss")
        axes[0].set_title("Per-term training loss", fontsize=14)
        axes[0].legend(ncol=3, fontsize=11)

        axes[1].plot(epoch, nu / nu_true, color="#d55e00", linewidth=2)
        axes[1].axhline(1.0, color="black", linestyle="--", label="true viscosity")
        best = np.argmin(np.abs(nu - nu_true))
        axes[1].scatter(epoch[best], nu[best] / nu_true, color="black", zorder=3)
        axes[1].annotate(f"closest: epoch {epoch[best]:,}\n{abs(nu[best] - nu_true) / nu_true:.1%} off",
                         xy=(epoch[best], nu[best] / nu_true),
                         xytext=(epoch[best] * 0.35, nu[best] / nu_true + 0.45),
                         fontsize=11, arrowprops=dict(arrowstyle="->"))
        axes[1].annotate(f"final: {abs(nu[-1] - nu_true) / nu_true:.1%} off",
                         xy=(epoch[-1], nu[-1] / nu_true), xytext=(epoch[-1] * 0.72, nu[-1] / nu_true + 0.7),
                         fontsize=11, arrowprops=dict(arrowstyle="->"))
        axes[1].set_xlabel("epoch")
        axes[1].set_ylabel(r"recovered $\nu$ / true $\nu$")
        axes[1].set_title("Viscosity trajectory", fontsize=14)
        axes[1].legend(fontsize=11)
        for ax in axes:
            ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)


def plot_training_curves_phase2(history, out_path):
    # No viscosity subplot here -- Phase 2 reuses Phase 1's recovered nu
    # rather than re-inferring it (see train_phase2.py) -- so the second
    # panel tracks the periodicity loss instead, the one genuinely new
    # loss term this phase adds.
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
    axes[0].semilogy(history["epoch"], history["data"], label="data")
    axes[0].semilogy(history["epoch"], history["pde"], label="pde")
    axes[0].semilogy(history["epoch"], history["bc"], label="bc")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss (log scale)")
    axes[0].set_title("Per-term training loss")
    axes[0].legend()

    axes[1].semilogy(history["epoch"], history["periodicity"], color="tab:purple")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("periodicity loss (log scale)")
    axes[1].set_title(r"$u(x,y,t_0) \approx u(x,y,t_0+T)$ residual")

    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
