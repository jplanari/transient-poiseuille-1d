"""Tests for the JSON configuration loader and the command-line interface."""

import json
from pathlib import Path

import numpy as np
import pytest

from transient_poiseuille import analytic
from transient_poiseuille.cli import main
from transient_poiseuille.config import load_config

ROOT = Path(__file__).resolve().parents[1]


def _write(tmp_path: Path, data: dict, name: str = "run.json") -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def _base(**overrides) -> dict:
    cfg = {
        "geometry": {"R": 0.05, "L": 1.0},
        "fluid": {"mu": 0.004, "rho": 8000.0},
        "drive": {"mode": "pressure_drop", "dp": 0.001},
        "numerics": {"Nx": 1, "Nr": 30, "dt": 250.0, "t_end": 1000.0},
    }
    cfg.update(overrides)
    return cfg


def test_load_minimal_config_uses_defaults(tmp_path):
    rc = load_config(_write(tmp_path, _base()))
    p = rc.params
    assert p.R == 0.05 and p.L == 1.0 and p.mu == 0.004 and p.rho == 8000.0
    assert p.mode == "pressure_drop" and p.dp == 0.001
    assert p.theta == 0.5 and p.convection is False
    assert rc.output_file is None


def test_output_path_is_relative_to_config_file(tmp_path):
    cfg = _base(output={"file": "out/run.npz"})
    rc = load_config(_write(tmp_path, cfg))
    assert rc.output_file == (tmp_path / "out" / "run.npz").resolve()


@pytest.mark.parametrize(
    "bad",
    [
        {"geometry": {"R": 0.05}},                                  # missing L
        {"drive": {"mode": "pressure_drop"}},                       # missing dp
        {"drive": {"mode": "flow"}},                                # missing V_c
        {"drive": {"mode": "gravity", "dp": 1.0}},                  # unknown mode
        {"numerics": {"Nx": 1, "Nr": 30, "dt": 250.0, "t_end": 1000.0, "thetaa": 1.0}},  # typo
        {"fluid": {"mu": -1.0, "rho": 8000.0}},                     # non-positive
        {"numerics": {"Nx": 1, "Nr": 1, "dt": 250.0, "t_end": 1000.0}},  # Nr too small
        {"numerics": {"Nx": 1, "Nr": 30, "dt": 250.0, "t_end": 1000.0, "theta": 2.0}},
        {"extra": {}},                                              # unknown section
    ],
)
def test_invalid_config_is_rejected(tmp_path, bad):
    cfg = _base()
    for key, value in bad.items():
        cfg[key] = value
    with pytest.raises(ValueError):
        load_config(_write(tmp_path, cfg))


def test_invalid_json_is_rejected(tmp_path):
    path = tmp_path / "broken.json"
    path.write_text("{ not json", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid JSON"):
        load_config(path)


def test_shipped_configs_load():
    for name in ("copper_pressure_drop.json", "copper_flow.json"):
        rc = load_config(ROOT / "configs" / name)
        assert rc.params.R > 0


def test_cli_runs_pressure_drop_config_and_saves_npz(tmp_path, capsys):
    cfg = _base(output={"file": "res.npz", "save_times": [500.0, 1000.0]})
    path = _write(tmp_path, cfg)
    code = main([str(path)])
    assert code == 0

    out = tmp_path / "res.npz"
    assert out.exists()
    data = np.load(out)
    assert data["u_final"].shape == (1, 30)
    assert list(data["snapshot_times"]) == [500.0, 1000.0]
    assert data["snapshot_u"].shape == (2, 1, 30)

    # the printed summary contains the Poiseuille comparison
    assert "vs Poiseuille" in capsys.readouterr().out


def test_cli_returns_error_code_for_bad_config(tmp_path, capsys):
    path = _write(tmp_path, {"geometry": {}})
    assert main([str(path)]) == 2
    assert "error" in capsys.readouterr().err


def test_cli_flow_mode_from_shipped_style_config(tmp_path):
    cfg = _base(drive={"mode": "flow", "p_in": 0.0, "V_c": 1e-3},
                numerics={"Nx": 4, "Nr": 60, "dt": 250.0, "t_end": 25000.0})
    rc = load_config(_write(tmp_path, cfg))
    from transient_poiseuille.solver import solve

    res = solve(rc.params)
    G_ex = -8.0 * rc.params.mu * 1e-3 / rc.params.R**2
    assert np.allclose(res.G_final, G_ex, rtol=1e-3)
    # analytic check of the Poiseuille relation used in the CLI report
    assert analytic.steady_flow_rate(G_ex, rc.params.R, rc.params.mu) == pytest.approx(
        np.pi * rc.params.R**2 * 1e-3, rel=1e-6)
