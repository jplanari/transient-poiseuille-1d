"""Time-dependent outlet velocity V_c(t) for the flow drive.

A casting speed that switches between an on state and an off state is a square
wave: V_on while the cycle is in its on part, V_off otherwise. The switch times are
sharp, so a time-stepping scheme that damps stiff modes (theta = 1, backward Euler)
is recommended; Crank-Nicolson rings after each jump.

A sharp switch is also unphysical for the pressure: the inertial term rho L dV/dt
diverges as the switch becomes sharper (the outlet pressure spike grows like 1/dt).
A finite ``ramp`` (seconds) makes the switch smooth.
"""

from __future__ import annotations

from typing import Callable


def square_wave(V_on: float, V_off: float, period: float, duty: float,
                t_start: float = 0.0, ramp: float = 0.0) -> Callable[[float], float]:
    """V(t): V_on during the first ``duty * period`` of each cycle, V_off otherwise.

    With ``ramp > 0`` each switch is a linear transition of that duration, which
    keeps dV/dt, and so the inertial pressure ``rho L dV/dt``, finite. Before
    ``t_start`` the velocity is V_off.
    """
    if period <= 0.0:
        raise ValueError("square wave period must be positive")
    if not 0.0 <= duty <= 1.0:
        raise ValueError("square wave duty must lie in [0, 1]")
    on_len = duty * period
    off_len = period - on_len
    if ramp < 0.0:
        raise ValueError("square wave ramp must not be negative")
    if ramp > 0.0 and ramp > min(on_len, off_len):
        raise ValueError("ramp must fit in both the on and the off part of the cycle")

    def V(t: float) -> float:
        if t < t_start:
            return V_off
        s = (t - t_start) % period
        if ramp == 0.0:
            return V_on if s < on_len else V_off
        if s < ramp:                                   # rising switch
            return V_off + (V_on - V_off) * s / ramp
        if s < on_len:
            return V_on
        if s < on_len + ramp:                          # falling switch
            return V_on + (V_off - V_on) * (s - on_len) / ramp
        return V_off

    return V


def drive_from_config(raw) -> float | Callable[[float], float]:
    """Parse ``drive.V_c``: a number (constant) or a square-wave description."""
    if isinstance(raw, dict):
        kind = raw.get("type")
        if kind != "square":
            raise ValueError(f"unknown drive.V_c type {kind!r} (expected 'square')")
        allowed = {"type", "V_on", "V_off", "period", "duty", "t_start", "ramp"}
        unknown = set(raw) - allowed
        if unknown:
            raise ValueError(f"unknown key(s) in drive.V_c: {sorted(unknown)}")
        if "V_on" not in raw or "period" not in raw or "duty" not in raw:
            raise ValueError("drive.V_c square wave requires V_on, period and duty")
        V_on = float(raw["V_on"])
        V_off = float(raw.get("V_off", 0.0))
        if V_on < 0.0 or V_off < 0.0:
            raise ValueError("drive.V_c velocities must not be negative")
        return square_wave(V_on, V_off, float(raw["period"]), float(raw["duty"]),
                           float(raw.get("t_start", 0.0)), float(raw.get("ramp", 0.0)))
    return float(raw)
