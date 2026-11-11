"""Plot the velocity field of a solver run.

Left panel: axial-radial map of the axial velocity u(x, r) over the full diameter
(the solution is symmetric about the axis, so the half-profile is mirrored).

Right panel: radial profiles u(r) at the saved times and at the final time, with
the analytical Poiseuille and start-up profiles as dashed lines when they apply
(constant viscosity, constant radius, pressure-drop drive).

Uses matplotlib's Figure/Agg API directly, so it works without a display.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from . import analytic
from .solver import Params, Result

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
                     color=colours[k], label=f"t = {t / analytic.viscous_time(p.R, p.mu, p.rho):.2f} τ")
    ax_prof.plot(r_mm, res.u_final[ref_station] * _MM_PER_M, color=colours[-1],
                 label=f"t = {res.t[-1] / analytic.viscous_time(p.R, p.mu, p.rho):.2f} τ (final)")

    if p.mode == "pressure_drop":
        G = -p.dp / p.L
        eta = res.eta
        ax_prof.plot(r_mm, analytic.steady_profile(eta, G, p.R, p.mu) * _MM_PER_M,
                     "k--", lw=1, label="Poiseuille (estacionari)")
        for k, t in enumerate(times):
            u_ex = analytic.transient_profile(eta, t, G, p.R, p.mu, p.rho)
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
