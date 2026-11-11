"""Print a verification report: numerical solver vs analytical Poiseuille solution.

Usage (from the repository root):
    python examples/verify_poiseuille.py
"""

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from transient_poiseuille import analytic  # noqa: E402
from transient_poiseuille.solver import Params, solve

R = 0.05          # m
L = 1.0           # m
MU = 4.0e-3       # Pa s
RHO = 8000.0      # kg/m^3
TAU = RHO * R**2 / MU
DP = 1.0e-3 * L   # Pa, imposed pressure drop over the bar
G = -DP / L


def main() -> None:
    save = tuple(f * TAU for f in (0.02, 0.1, 0.5, 1.0, 3.0))
    prm = Params(R=R, L=L, mu=MU, rho=RHO, mode="pressure_drop", dp=DP,
                 Nx=1, Nr=100, dt=TAU / 400, t_end=3 * TAU, theta=0.5,
                 save_times=save)
    res = solve(prm)
    u_scale = R**2 * abs(G) / (4 * MU)

    print(f"tau = rho R^2 / mu = {TAU:.1f} s")
    print(f"centre-line steady velocity = {u_scale:.4e} m/s")
    print()
    print(f"{'t/tau':>8} {'max|u_num-u_ex|/u_max':>24}")
    for ts in sorted(res.snapshots):
        u_ex = analytic.transient_profile(res.eta, ts, G, R, MU, RHO)
        err = np.max(np.abs(res.snapshots[ts][0] - u_ex)) / u_scale
        print(f"{ts / TAU:8.3f} {err:24.3e}")

    u_ex_ss = analytic.steady_profile(res.eta, G, R, MU)
    err_ss = np.max(np.abs(res.u_final[0] - u_ex_ss)) / u_scale
    Q_ex = analytic.steady_flow_rate(G, R, MU)
    print()
    print(f"steady profile error (relative to u_max) : {err_ss:.3e}")
    print(f"flow rate  numerical / analytical        : "
          f"{res.Q_history[-1]:.6e} / {Q_ex:.6e} m^3/s")


if __name__ == "__main__":
    main()
