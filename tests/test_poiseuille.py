"""Verification of the solver against the analytical Poiseuille solution.

Run from the repository root with:  python -m pytest -q
"""

import numpy as np
import pytest

from transient_poiseuille import analytic
from transient_poiseuille.solver import Params, radial_operator, solve

# Copper-like properties, SI units
R = 0.05          # m
L = 1.0           # m
MU = 4.0e-3       # Pa s
RHO = 8000.0      # kg/m^3
TAU = RHO * R**2 / MU   # viscous diffusion time, 5000 s


def test_steady_pressure_drop_recovers_poiseuille():
    """Constant viscosity and imposed pressure drop -> parabolic profile and Q."""
    dp = 1.0e-3 * L                 # Pa
    G = -dp / L
    prm = Params(R=R, L=L, mu=MU, rho=RHO, mode="pressure_drop", dp=dp,
                 Nx=1, Nr=80, dt=TAU / 50, t_end=6 * TAU, theta=0.5)
    res = solve(prm)

    u_ex = analytic.steady_profile(res.eta, G, R, MU)
    u_num = res.u_final[0]
    assert np.max(np.abs(u_num - u_ex)) / np.max(np.abs(u_ex)) < 1e-3

    Q_ex = analytic.steady_flow_rate(G, R, MU)
    assert abs(res.Q_history[-1] - Q_ex) / abs(Q_ex) < 1e-3


def test_transient_start_up_matches_bessel_series():
    """Start-up from rest: numerical profiles follow the Fourier-Bessel series."""
    dp = 1.0e-3 * L
    G = -dp / L
    u_scale = R**2 * abs(G) / (4 * MU)     # centre-line steady velocity
    save = tuple(f * TAU for f in (0.05, 0.2, 0.5, 1.0, 2.0))
    prm = Params(R=R, L=L, mu=MU, rho=RHO, mode="pressure_drop", dp=dp,
                 Nx=1, Nr=100, dt=TAU / 400, t_end=2 * TAU, theta=0.5,
                 save_times=save)
    res = solve(prm)

    assert len(res.snapshots) == len(save)
    for ts, field in res.snapshots.items():
        u_ex = analytic.transient_profile(res.eta, ts, G, R, MU, RHO)
        err = np.max(np.abs(field[0] - u_ex)) / u_scale
        assert err < 5e-3, f"t = {ts / TAU:.2f} tau: relative error {err:.2e}"


def test_analytic_series_starts_from_rest():
    """The series must vanish at t = 0 (fluid initially at rest)."""
    eta = np.linspace(0.0, 1.0, 201)
    u0 = analytic.transient_profile(eta, 0.0, -1.0e-3, R, MU, RHO, n_modes=200)
    assert np.max(np.abs(u0)) < 1e-3 * abs(1.0e-3 * R**2 / (4 * MU))


def test_radial_refinement_converges():
    """Steady error decreases when the radial mesh is refined."""
    G = -1.0e-3
    errs = []
    for Nr in (20, 40, 80):
        prm = Params(R=R, L=L, mu=MU, rho=RHO, mode="pressure_drop",
                     dp=-G * L, Nx=1, Nr=Nr, dt=TAU / 20, t_end=8 * TAU)
        res = solve(prm)
        u_ex = analytic.steady_profile(res.eta, G, R, MU)
        errs.append(np.max(np.abs(res.u_final[0] - u_ex)))
    assert errs[0] > errs[1] > errs[2]
    # roughly second order: halving the mesh divides the error by about 4
    assert errs[1] / errs[2] > 3.0


def test_flow_mode_gives_linear_pressure_and_analytic_outlet_pressure():
    """Prescribed V_c and p_in: G = -8 mu V_c / R^2, p_out = p_in - 8 mu V_c L / R^2."""
    V = 1.0e-3
    prm = Params(R=R, L=L, mu=MU, rho=RHO, mode="flow", p_in=0.0, V_c=V,
                 Nx=10, Nr=60, dt=TAU / 50, t_end=8 * TAU, theta=0.5)
    res = solve(prm)

    G_ex = -8.0 * MU * V / R**2
    assert np.allclose(res.G_final, G_ex, rtol=1e-3)
    p_out_ex = -8.0 * MU * V * L / R**2
    assert abs(res.p_out - p_out_ex) / abs(p_out_ex) < 1e-3


def test_radial_operator_spectrum_is_real_and_negative():
    """The diffusion operator with Dirichlet wall must be stable: real, negative eigenvalues."""
    Lm = radial_operator(40, R, MU)
    eig = np.linalg.eigvals(Lm)
    assert np.max(eig.real) < 0.0
    assert np.max(np.abs(eig.imag)) < 1e-8 * np.max(np.abs(eig))


@pytest.mark.parametrize("theta", [0.5, 1.0])
def test_both_time_schemes_reach_steady_state(theta):
    G = -1.0e-3
    prm = Params(R=R, L=L, mu=MU, rho=RHO, mode="pressure_drop", dp=-G * L,
                 Nx=1, Nr=60, dt=TAU / 50, t_end=8 * TAU, theta=theta)
    res = solve(prm)
    u_ex = analytic.steady_profile(res.eta, G, R, MU)
    assert np.max(np.abs(res.u_final[0] - u_ex)) / np.max(np.abs(u_ex)) < 5e-3
