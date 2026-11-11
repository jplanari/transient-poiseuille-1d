"""Time-dependent casting speed V_c(t): on/off square wave in flow mode."""

import json
from pathlib import Path

import numpy as np
import pytest

from transient_poiseuille.config import load_config
from transient_poiseuille.drive import drive_from_config, square_wave
from transient_poiseuille.solver import Params, solve

R = 0.05
L = 1.0
MU = 0.004
RHO = 8000.0
TAU = RHO * R**2 / MU   # 5000 s


def test_square_wave_values():
    V = square_wave(V_on=0.01, V_off=0.0, period=100.0, duty=0.5, t_start=10.0)
    assert V(0.0) == 0.0            # before t_start
    assert V(10.0) == 0.01          # on at the start of the cycle
    assert V(59.9) == 0.01
    assert V(60.0) == 0.0           # off in the second half
    assert V(110.0) == 0.01         # next cycle
    assert V(170.0) == 0.0


@pytest.mark.parametrize("bad", [
    {"type": "triangle", "V_on": 1.0, "period": 1.0, "duty": 0.5},
    {"type": "square", "period": 1.0, "duty": 0.5},                    # no V_on
    {"type": "square", "V_on": 1.0, "period": 0.0, "duty": 0.5},       # period <= 0
    {"type": "square", "V_on": 1.0, "period": 1.0, "duty": 1.5},       # duty > 1
    {"type": "square", "V_on": -1.0, "period": 1.0, "duty": 0.5},      # negative
    {"type": "square", "V_on": 1.0, "period": 1.0, "duty": 0.5, "x": 1},  # unknown key
    {"type": "square", "V_on": 1.0, "period": 1.0, "duty": 0.5, "ramp": 0.9},  # ramp too long
])
def test_invalid_drive_is_rejected(bad):
    with pytest.raises(ValueError):
        drive_from_config(bad)


def test_ramped_square_wave_is_continuous_and_bounded():
    V = square_wave(V_on=0.01, V_off=0.0, period=100.0, duty=0.5, ramp=10.0)
    assert V(0.0) == 0.0
    assert V(5.0) == pytest.approx(0.005)       # halfway up the rising switch
    assert V(10.0) == 0.01
    assert V(45.0) == 0.01                      # still on before the falling switch (at 50 s)
    assert V(55.0) == pytest.approx(0.005)      # halfway down the falling switch
    assert V(65.0) == 0.0                       # off after the falling switch (50..60 s)
    assert V(70.0) == 0.0
    t = np.linspace(0.0, 300.0, 30001)
    vals = np.array([V(x) for x in t])
    # slope 0.01 / 10 s, sampled every 0.01 s: consecutive values differ by <= 1e-5
    assert np.max(np.abs(np.diff(vals))) <= 1e-5 * 1.01


def test_ramp_must_fit_in_the_cycle():
    with pytest.raises(ValueError):
        square_wave(V_on=1.0, V_off=0.0, period=100.0, duty=0.5, ramp=60.0)


def test_constant_number_still_works():
    assert drive_from_config(0.01) == 0.01


def _run(V, t_end, theta=1.0):
    prm = Params(R=R, L=L, mu=MU, rho=RHO, mode="flow", p_in=0.0, V_c=V,
                 Nx=2, Nr=40, dt=50.0, t_end=t_end, theta=theta, convection=False)
    return solve(prm)


def test_always_on_reaches_the_constant_flow_rate():
    V = square_wave(V_on=0.01, V_off=0.0, period=20000.0, duty=1.0)
    res = _run(V, t_end=10 * TAU)
    assert res.Q_history[-1] == pytest.approx(np.pi * R**2 * 0.01, rel=1e-6)


def test_always_off_gives_no_flow():
    V = square_wave(V_on=0.01, V_off=0.0, period=20000.0, duty=0.0)
    res = _run(V, t_end=5 * TAU)
    assert abs(res.Q_history[-1]) < 1e-12


def test_flow_follows_the_on_off_cycle():
    """Two cycles of 2 tau on and 2 tau off: the flow is at its target at the end of
    each on phase and has decayed to (almost) zero at the end of each off phase."""
    period = 4 * TAU        # on for 2 tau, off for 2 tau
    V = square_wave(V_on=0.01, V_off=0.0, period=period, duty=0.5)
    res = _run(V, t_end=2 * period)
    dt = res.t[1] - res.t[0]
    Q_on = np.pi * R**2 * 0.01
    end_on = int(round(0.5 * period / dt))
    end_off = int(round(period / dt))
    assert res.Q_history[end_on] == pytest.approx(Q_on, rel=1e-3)
    assert abs(res.Q_history[end_off]) < 1e-3 * Q_on
    assert res.Q_history[int(round(1.5 * period / dt))] == pytest.approx(Q_on, rel=1e-3)


def test_shipped_square_casting_config_loads_and_runs(tmp_path):
    root = Path(__file__).resolve().parents[1]
    rc = load_config(root / "configs" / "copper_square_casting.json")
    assert callable(rc.params.V_c)
    cfg = json.loads((root / "configs" / "copper_square_casting.json").read_text())
    cfg["numerics"]["t_end"] = 2000.0      # keep the test short
    cfg["output"] = {}
    path = tmp_path / "short.json"
    path.write_text(json.dumps(cfg), encoding="utf-8")
    from transient_poiseuille.cli import main

    assert main([str(path)]) == 0
