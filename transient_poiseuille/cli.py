"""Command-line interface: run the solver from a JSON configuration file.

Usage::

    python -m transient_poiseuille configs/copper_pressure_drop.json
    python -m transient_poiseuille configs/copper_flow.json -o results/flow.npz --plot results/flow.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from . import analytic
from .config import load_config
from .solver import Result, is_uniform_mu, mu_reference, solve


def save_result(res: Result, path: Path) -> None:
    """Write the solution to a compressed .npz file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    times = np.array(sorted(res.snapshots), dtype=float)
    snap_u = np.stack([res.snapshots[t] for t in times]) if len(times) else np.empty((0,))
    np.savez_compressed(
        path,
        t=res.t,
        x=res.x,
        eta=res.eta,
        u_final=res.u_final,
        Q_history=res.Q_history,
        G_final=res.G_final,
        p_final=res.p_final,
        p_out=res.p_out,
        snapshot_times=times,
        snapshot_u=snap_u,
    )


def report(cfg, res: Result) -> None:
    """Print a short summary; compare with Poiseuille when it applies."""
    p = cfg.params
    mu_ref = mu_reference(p)
    tau = analytic.viscous_time(p.R, mu_ref, p.rho)
    uniform = is_uniform_mu(p)

    print(f"config            : {cfg.source}")
    print(f"mode              : {p.mode}")
    if uniform:
        print(f"viscosity         : {mu_ref:.4g} Pa s (uniform)")
    else:
        mu = np.asarray(p.mu, dtype=float)
        print(f"viscosity         : {mu.min():.4g} .. {mu.max():.4g} Pa s (varies along x)")
    print(f"tau = rho R^2/mu  : {tau:.4g} s (mean mu)")
    print(f"final time        : {res.t[-1]:.6g} s ({res.t[-1] / tau:.3g} tau)")
    print(f"final Q           : {res.Q_history[-1]:.6e} m^3/s")
    print(f"outlet pressure   : {res.p_out:.6e} Pa")

    if p.mode == "pressure_drop" and uniform:
        G = -p.dp / p.L
        u_ex = analytic.steady_profile(res.eta, G, p.R, mu_ref)
        u_scale = np.max(np.abs(u_ex))
        err_u = np.max(np.abs(res.u_final[0] - u_ex)) / u_scale
        Q_ex = analytic.steady_flow_rate(G, p.R, mu_ref)
        err_Q = abs(res.Q_history[-1] - Q_ex) / abs(Q_ex)
        print(f"vs Poiseuille     : profile error {err_u:.2e}, flow error {err_Q:.2e}")
        print("                    (meaningful once t_end is several tau)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="transient_poiseuille",
                                     description=__doc__.splitlines()[0])
    parser.add_argument("config", help="path to a JSON configuration file")
    parser.add_argument("-o", "--output", help="output .npz file (overrides the config)")
    parser.add_argument("--plot", metavar="PNG",
                        help="also write a PNG of the velocity field (needs matplotlib)")
    args = parser.parse_args(argv)

    try:
        cfg = load_config(args.config)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    try:
        res = solve(cfg.params)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    report(cfg, res)

    out = Path(args.output) if args.output else cfg.output_file
    if out is not None:
        save_result(res, out)
        print(f"saved             : {out}")

    if args.plot:
        from .plot import plot_velocity

        try:
            png = plot_velocity(res, cfg.params, args.plot, title=cfg.source.name)
        except RuntimeError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        print(f"plot              : {png}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
