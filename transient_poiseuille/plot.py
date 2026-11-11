"""Plot the velocity field of a solver run.

Left panel: axial-radial map of the axial velocity u(x, r) over the full diameter
(the solution is symmetric about the axis, so the half-profile is mirrored).

Right panel: radial profiles u(r) at the saved times and at the final time, with
the analytical Poiseuille and start-up profiles as dashed lines when they apply
(uniform viscosity, constant radius, pressure-drop drive).

Uses matplotlib's Figure/Agg API directly, so it works without a display.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from . import analytic
from .solver import Params, Result, is_uniform_mu, mu_reference

_MM_PER_M = 1.0e3


def _figure_classes():
    try:
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise RuntimeError("plotting needs matplotlib: pip install matplotlib") from exc
    return Figure, FigureCanvasAgg


def plot_velocity(res: Result, p: Params, out: str | Path, title: str | None = None) -> Path:
    """Write a two-panel PNG of the velocity field to ``out`` and return its path."""
    Figure, FigureCanvasAgg = _figure_classes()
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)

    mu_ref = mu_reference(p)
    uniform = is_uniform_mu(p)
    tau = analytic.viscous_time(p.R, mu_ref, p.rho)

    fig = Figure(figsize=(12, 5), layout="constrained")
    FigureCanvasAgg(fig)
    ax_map, ax_prof = fig.subplots(1, 2)

    R_mm = p.R * _MM_PER_M
    r_mm = res.eta * R_mm
    # mirror the half-profile to the full diameter, r from -R to +R
    u_half = res.u_final * _MM_PER_M
    u_full = np.hstack([u_half[:, ::-1], u_half])
    x_lo, x_hi = res.x[0], res.x[-1]
    if len(res.x) == 1:
        x_lo, x_hi = 0.0, p.L
    im = ax_map.imshow(
        u_full.T,
        origin="lower",
        aspect="auto",
        extent=(x_lo, x_hi, -R_mm, R_mm),
        cmap="viridis",
    )
    fig.colorbar(im, ax=ax_map, label="u [mm/s]")
    ax_map.set_xlabel("x [m]")
    ax_map.set_ylabel("r [mm]")
    ax_map.set_title("Velocitat axial u(x, r)")

    # radial profiles at the first axial station
    colours = plt_colours(len(res.snapshots) + 1)
    times = sorted(res.snapshots)
    ref_station = 0
    for k, t in enumerate(times):
        ax_prof.plot(r_mm, res.snapshots[t][ref_station] * _MM_PER_M,
                     color=colours[k], label=f"t = {t / tau:.2f} τ")
    ax_prof.plot(r_mm, res.u_final[ref_station] * _MM_PER_M, color=colours[-1],
                 label=f"t = {res.t[-1] / tau:.2f} τ (final)")

    if p.mode == "pressure_drop" and uniform:
        G = -p.dp / p.L
        eta = res.eta
        ax_prof.plot(r_mm, analytic.steady_profile(eta, G, p.R, mu_ref) * _MM_PER_M,
                     "k--", lw=1, label="Poiseuille (estacionari)")
        for k, t in enumerate(times):
            u_ex = analytic.transient_profile(eta, t, G, p.R, mu_ref, p.rho)
            ax_prof.plot(r_mm, u_ex * _MM_PER_M, ":", color=colours[k], lw=1)

    ax_prof.set_xlabel("r [mm]")
    ax_prof.set_ylabel("u [mm/s]")
    ax_prof.set_title("Perfil radial (estació x = primera cel·la)")
    ax_prof.grid(alpha=0.3)
    ax_prof.legend(fontsize=8)

    if title:
        fig.suptitle(title)
    fig.savefig(out, dpi=150)
    return out


def plt_colours(n: int):
    """n distinct colours from a perceptually uniform colormap (no pyplot needed)."""
    from matplotlib import colormaps

    cmap = colormaps["plasma"]
    return [cmap(v) for v in np.linspace(0.05, 0.9, max(n, 1))]


def plot_convection_effect(res_off: Result, res_on: Result, p: Params,
                           out: str | Path, title: str | None = None) -> Path:
    """Time-resolved view of the convective term: run with it on minus run with it off.

    Shows the centre-line velocity u(x, r=0, t) without convection, the relative
    difference (on - off) as an x-t map, a few centre-line profiles with and
    without convection, and the time series of the maximum difference and of the
    axial spread of the velocity.
    """
    Figure, FigureCanvasAgg = _figure_classes()
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)

    times = sorted(set(res_off.snapshots) & set(res_on.snapshots))
    if not times:
        raise ValueError("the two runs have no common snapshot times; set save_times")
    tau = analytic.viscous_time(p.R, mu_reference(p), p.rho)
    tt = np.array(times) / tau
    x = res_off.x

    U_off = np.array([res_off.snapshots[t][:, 0] for t in times])   # (nt, Nx), r = 0
    U_on = np.array([res_on.snapshots[t][:, 0] for t in times])
    u_scale = np.max(np.abs(U_off))
    if u_scale == 0.0:
        raise ValueError("velocity is zero everywhere; nothing to compare")
    D = (U_on - U_off) / u_scale                                     # relative difference
    spread = (U_off.max(axis=1) - U_off.min(axis=1)) / u_scale       # axial spread, no conv.

    fig = Figure(figsize=(13, 9), layout="constrained")
    FigureCanvasAgg(fig)
    ax = fig.subplots(2, 2)
    extent = (tt[0], tt[-1], x[0], x[-1]) if len(times) > 1 else (tt[0] - 0.5, tt[0] + 0.5, x[0], x[-1])

    im = ax[0, 0].imshow(U_off.T * _MM_PER_M, origin="lower", aspect="auto",
                         extent=extent, cmap="viridis")
    fig.colorbar(im, ax=ax[0, 0], label="u(x, r=0) [mm/s]")
    ax[0, 0].set_xlabel("t / τ")
    ax[0, 0].set_ylabel("x [m]")
    ax[0, 0].set_title("Velocitat central sense convectiu")

    lim = 100.0 * np.max(np.abs(D))
    lim = lim if lim > 0 else 1.0
    im2 = ax[0, 1].imshow(100.0 * D.T, origin="lower", aspect="auto", extent=extent,
                          cmap="RdBu_r", vmin=-lim, vmax=lim)
    fig.colorbar(im2, ax=ax[0, 1], label="(amb − sense) / max|u| [%]")
    ax[0, 1].set_xlabel("t / τ")
    ax[0, 1].set_ylabel("x [m]")
    ax[0, 1].set_title("Efecte del convectiu")

    # show the times where the effect develops, not evenly spaced ones
    idx = sorted({int(np.argmin(np.abs(tt - s))) for s in (0.02, 0.06, 0.15, 0.3)})
    colours = plt_colours(len(idx))
    for k, i in enumerate(idx):
        ax[1, 0].plot(x, U_on[i] * _MM_PER_M, color=colours[k],
                      label=f"t = {tt[i]:.2f} τ")
        ax[1, 0].plot(x, U_off[i] * _MM_PER_M, "--", color=colours[k], lw=1)
    ax[1, 0].set_xlabel("x [m]")
    ax[1, 0].set_ylabel("u(x, r=0) [mm/s]")
    ax[1, 0].set_title("Contínua: amb convectiu. Discontínua: sense")
    ax[1, 0].grid(alpha=0.3)
    ax[1, 0].legend(fontsize=8)

    ax[1, 1].plot(tt, 100.0 * np.max(np.abs(D), axis=1), label="max en x |diferència| [%]")
    ax[1, 1].plot(tt, 100.0 * spread, label="dispersió en x de u sense convectiu [%]")
    ax[1, 1].set_xlabel("t / τ")
    ax[1, 1].set_ylabel("%")
    ax[1, 1].set_title("Magnitud de l'efecte en el temps")
    ax[1, 1].grid(alpha=0.3)
    ax[1, 1].legend(fontsize=8)

    if title:
        fig.suptitle(title)
    fig.savefig(out, dpi=150)
    return out
