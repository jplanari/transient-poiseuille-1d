"""Viscosity varying along the axis, mu(x).

Each axial cell is a Poiseuille section with its own viscosity, and the axial
coupling is the common flow Q with G_i = (a_i - Q) / b_i.
"""

import json

import numpy as np
import pytest

from transient_poiseuille.cli import main
from transient_poiseuille.config import load_config
from transient_poiseuille.solver import Params, mu_per_station, solve

R = 0.05
L = 1.0
RHO = 8000.0
NX = 10
MU_X = np.linspace(0.004, 0.008, NX)   # increasing with x


def _steady(mode, mu, **extra):
    """Run to a steady state, long compared with the slowest relaxation time.

    Backward Euler (theta = 1) is used: Crank-Nicolson does not damp the stiff radial
    modes excited by the sudden start, which leaves a ~1e-5 spread in G/mu at dt = 250 s.
    """
    mu_min = float(np.min(mu))
    t_end = 10.0 * RHO * R**2 / mu_min
    prm = Params(R=R, L=L, mu=mu, rho=RHO, mode=mode, Nx=NX, Nr=60,
                 dt=250.0, t_end=t_end, theta=1.0, convection=False, **extra)
    return solve(prm)


def test_constant_array_matches_scalar_viscosity():
    """mu given as a constant array must reproduce the scalar case exactly."""
    common = dict(R=R, L=L, rho=RHO, mode="pressure_drop", dp=1e-3, Nx=NX,
                  Nr=40, dt=250.0, t_end=5000.0, theta=0.5)
    a = solve(Params(mu=0.004, **common))
    b = solve(Params(mu=np.full(NX, 0.004), **common))
    assert np.allclose(a.u_final, b.u_final, rtol=0, atol=1e-14)
    assert np.allclose(a.Q_history, b.Q_history, rtol=0, atol=1e-18)


def test_mu_array_must_match_nx():
    with pytest.raises(ValueError, match="length Nx"):
        mu_per_station(Params(R=R, L=L, mu=np.ones(3), rho=RHO, Nx=NX))


def test_flow_mode_gives_local_poiseuille_gradient():
    """Prescribed outlet velocity: G_i = -8 mu_i V_c / R^2 in each cell."""
    V = 1e-3
    res = _steady("flow", MU_X, V_c=V, p_in=0.0)
    G_ex = -8.0 * MU_X * V / R**2
    assert np.allclose(res.G_final, G_ex, rtol=2e-3)
    assert np.allclose(res.Q_history[-1], np.pi * R**2 * V, rtol=1e-3)


def test_pressure_drop_mode_gives_series_resistance_flow():
    """Prescribed total drop: Q = pi R^4 dp / (8 * integral of mu dx)."""
    dp = 1e-3
    res = _steady("pressure_drop", MU_X, dp=dp)
    Q_ex = np.pi * R**4 * dp / (8.0 * np.sum(MU_X) * (L / NX))
    assert res.Q_history[-1] == pytest.approx(Q_ex, rel=2e-3)
    # the pressure gradient is distributed in proportion to the local viscosity
    ratio = res.G_final / MU_X
    assert np.allclose(ratio, ratio[0], rtol=1e-10)
    assert res.p_out == pytest.approx(-dp, rel=1e-10)


def test_config_accepts_list_and_rejects_bad_lists(tmp_path):
    cfg = {
        "geometry": {"R": R, "L": L},
        "fluid": {"mu": MU_X.tolist(), "rho": RHO},
        "drive": {"mode": "pressure_drop", "dp": 1e-3},
        "numerics": {"Nx": NX, "Nr": 30, "dt": 250.0, "t_end": 1000.0},
    }
    path = tmp_path / "mu.json"
    path.write_text(json.dumps(cfg), encoding="utf-8")
    assert load_config(path).params.mu.shape == (NX,)

    cfg["fluid"]["mu"] = [0.004, 0.005]          # wrong length
    path.write_text(json.dumps(cfg), encoding="utf-8")
    with pytest.raises(ValueError, match="length"):
        load_config(path)

    cfg["fluid"]["mu"] = [-0.004] * NX           # not positive
    path.write_text(json.dumps(cfg), encoding="utf-8")
    with pytest.raises(ValueError, match="positive"):
        load_config(path)


def test_cli_runs_variable_viscosity_config_with_plot(tmp_path):
    pytest.importorskip("matplotlib")
    cfg = {
        "geometry": {"R": R, "L": L},
        "fluid": {"mu": MU_X.tolist(), "rho": RHO},
        "drive": {"mode": "flow", "p_in": 0.0, "V_c": 1e-3},
        "numerics": {"Nx": NX, "Nr": 30, "dt": 250.0, "t_end": 5000.0},
    }
    path = tmp_path / "mu.json"
    path.write_text(json.dumps(cfg), encoding="utf-8")
    png = tmp_path / "mu.png"
    assert main([str(path), "--plot", str(png)]) == 0
    assert png.exists() and png.stat().st_size > 1000


def test_shipped_mu_gradient_config_loads():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    rc = load_config(root / "configs" / "copper_mu_gradient.json")
    assert rc.params.mu.shape == (rc.params.Nx,)
