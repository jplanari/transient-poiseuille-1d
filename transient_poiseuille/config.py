"""Load a JSON configuration file into solver parameters.

Example file::

    {
      "geometry": {"R": 0.05, "L": 1.0},
      "fluid":    {"mu": 0.004, "rho": 8000.0},
      "drive":    {"mode": "pressure_drop", "dp": 0.001},
      "numerics": {"Nx": 1, "Nr": 80, "dt": 250.0, "t_end": 25000.0,
                   "theta": 0.5, "convection": false},
      "output":   {"file": "results/run.npz", "save_times": [500.0, 2500.0]}
    }

Only "geometry", "fluid", "drive" (mode-dependent) and "numerics" with dt and t_end
are required. Unknown keys are rejected so that typos do not pass silently.
Relative output paths are resolved against the directory of the configuration file.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .solver import Params

_ALLOWED = {
    "geometry": {"R", "L"},
    "fluid": {"mu", "rho"},
    "drive": {"mode", "dp", "p_in", "V_c"},
    "numerics": {"Nx", "Nr", "dt", "t_end", "theta", "convection"},
    "output": {"file", "save_times"},
}
_REQUIRED = {
    "geometry": ("R", "L"),
    "fluid": ("mu", "rho"),
    "numerics": ("dt", "t_end"),
}


@dataclass
class RunConfig:
    """Solver parameters plus the optional output location."""

    params: Params
    output_file: Path | None
    source: Path


def _section(raw: dict, name: str) -> dict:
    data = raw.get(name, {})
    if not isinstance(data, dict):
        raise ValueError(f"section '{name}' must be a JSON object")
    unknown = set(data) - _ALLOWED[name]
    if unknown:
        raise ValueError(f"unknown key(s) in '{name}': {sorted(unknown)}")
    for key in _REQUIRED.get(name, ()):
        if key not in data:
            raise ValueError(f"missing required key '{name}.{key}'")
    return data


def _positive(value: float, name: str) -> float:
    if value <= 0:
        raise ValueError(f"'{name}' must be positive, got {value}")
    return value


def load_config(path: str | Path) -> RunConfig:
    """Read and validate a configuration file. Raises ValueError on bad input."""
    path = Path(path)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: invalid JSON ({exc})") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: top level must be a JSON object")
    unknown = set(raw) - set(_ALLOWED)
    if unknown:
        raise ValueError(f"unknown top-level section(s): {sorted(unknown)}")

    geom = _section(raw, "geometry")
    fluid = _section(raw, "fluid")
    drive = _section(raw, "drive")
    num = _section(raw, "numerics")
    out = _section(raw, "output")

    mode = str(drive.get("mode", "pressure_drop"))
    if mode == "pressure_drop" and "dp" not in drive:
        raise ValueError("mode 'pressure_drop' requires 'drive.dp'")
    if mode == "flow" and "V_c" not in drive:
        raise ValueError("mode 'flow' requires 'drive.V_c'")

    Nx = int(num.get("Nx", 1))
    Nr = int(num.get("Nr", 40))
    if Nx < 1 or Nr < 2:
        raise ValueError("'numerics.Nx' must be >= 1 and 'numerics.Nr' >= 2")
    convection = bool(num.get("convection", False))
    if convection and Nx < 2:
        raise ValueError("'numerics.convection' requires 'numerics.Nx' >= 2 "
                         "(there is no axial gradient with a single cell)")

    params = Params(
        R=_positive(float(geom["R"]), "geometry.R"),
        L=_positive(float(geom["L"]), "geometry.L"),
        mu=_positive(float(fluid["mu"]), "fluid.mu"),
        rho=_positive(float(fluid["rho"]), "fluid.rho"),
        mode=mode,
        dp=float(drive.get("dp", 0.0)),
        p_in=float(drive.get("p_in", 0.0)),
        V_c=float(drive.get("V_c", 0.0)),
        Nx=Nx,
        Nr=Nr,
        dt=_positive(float(num["dt"]), "numerics.dt"),
        t_end=_positive(float(num["t_end"]), "numerics.t_end"),
        theta=float(num.get("theta", 0.5)),
        convection=convection,
        save_times=tuple(float(t) for t in out.get("save_times", [])),
    )
    if params.mode not in ("pressure_drop", "flow"):
        raise ValueError(f"unknown drive.mode {mode!r}")
    if not 0.0 <= params.theta <= 1.0:
        raise ValueError("'numerics.theta' must lie in [0, 1]")

    output_file = None
    if "file" in out:
        output_file = Path(out["file"])
        if not output_file.is_absolute():
            output_file = (path.parent / output_file).resolve()

    return RunConfig(params=params, output_file=output_file, source=path)
