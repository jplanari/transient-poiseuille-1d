"""Run a configuration with and without the convective term and plot the difference.

Usage (from the repository root):
    python examples/convection_effect.py configs/copper_mu_gradient_convective.json
    python examples/convection_effect.py <config.json> --out results/effect.png --every 50
"""

import argparse
import dataclasses
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from transient_poiseuille.config import load_config  # noqa: E402
from transient_poiseuille.plot import plot_convection_effect  # noqa: E402
from transient_poiseuille.solver import solve  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("config")
    parser.add_argument("--out", default="results/convection_effect.png")
    parser.add_argument("--every", type=float, default=25.0,
                        help="snapshot interval [s] (rounded to a multiple of dt)")
    args = parser.parse_args()

    try:
        p = load_config(args.config).params
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if p.Nx < 2:
        print("error: the convective term needs numerics.Nx >= 2", file=sys.stderr)
        return 2

    step = max(1, int(round(args.every / p.dt)))
    n_steps = int(round(p.t_end / p.dt))
    saves = tuple(k * p.dt for k in range(step, n_steps + 1, step))

    try:
        off = solve(dataclasses.replace(p, convection=False, save_times=saves))
        on = solve(dataclasses.replace(p, convection=True, save_times=saves))
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    png = plot_convection_effect(off, on, p, args.out, title=pathlib.Path(args.config).name)
    scale = np.max(np.abs(off.snapshots[max(off.snapshots)][:, 0]))
    final = on.u_final[:, 0] - off.u_final[:, 0]
    print(f"plot              : {png}")
    print(f"snapshots         : {len(off.snapshots)}")
    print(f"max |diff| (final): {100 * np.max(np.abs(final)) / scale:.3f} % of centre-line max")
    return 0


if __name__ == "__main__":
    sys.exit(main())
