"""Command-line interface: run the solver from a JSON configuration file.

Usage::

    python -m transient_poiseuille configs/copper_pressure_drop.json
    python -m transient_poiseuille configs/copper_flow.json -o results/flow.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from . import analytic
from .config import load_config
from .solver import Result, solve


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
    """Print a short summary; compare with Poiseuille when the drive is a pressure drop."""
    p = cfg.params
    print(f"config            : {cfg.source}")
    print(f"mode              : {p.mode}")
    print(f"tau = rho R^2/mu  : {analytic.viscous_time(p.R, p.mu, p.rho):.4g} s")
    print(f"final time        : {res.t[-1]:.6g} s ({res.t[-1] / analytic.viscous_time(p.R, p.mu, p.rho):.3g} tau)")
    print(f"final Q           : {res.Q_history[-1]:.6e} m^3/s")
    print(f"outlet pressure   : {res.p_out:.6e} Pa")

    if p.mode == "pressure_drop":
        G = -p.dp / p.L
        u_ex = analytic.steady_profile(res.eta, G, p.R, p.mu)
        u_scale = np.max(np.abs(u_ex))
        err_u = np.max(np.abs(res.u_final[0] - u_ex)) / u_scale
        Q_ex = analytic.steady_flow_rate(G, p.R, p.mu)
        err_Q = abs(res.Q_history[-1] - Q_ex) / abs(Q_ex)
        print(f"vs Poiseuille     : profile error {err_u:.2e}, flow error {err_Q:.2e}")
        print("                    (meaningful once t_end is several tau)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="transient_poiseuille",
                                     description=__doc__.splitlines()[0])
    parser.add_argument("config", help="path to a JSON configuration file")
    parser.add_argument("-o", "--output", help="output .npz file (overrides the config)")
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
    return 0


if __name__ == "__main__":
    sys.exit(main())
