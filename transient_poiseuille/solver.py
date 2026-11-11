"""Transient axisymmetric flow in a bar of constant radius.

For each axial cell (length dx) the radial velocity u(eta, t), with eta = r/R, obeys

    rho * (du/dt + u du/dx) = -G + (mu / R^2) * (1/eta) d/deta (eta du/deta)

with  du/deta = 0 at eta = 0 (symmetry),  u = 0 at eta = 1 (wall),
and G = dp/dx uniform over the section.

Discretisation
--------------
* Radial: cell-centred finite volumes in eta, wall handled with a half-cell ghost.
* Time: theta-scheme (theta = 0.5 Crank-Nicolson, theta = 1 backward Euler).
* Convection (optional): treated explicitly with the previous step's velocity.

Drive modes
-----------
"pressure_drop": G = -dp/L is prescribed. Used to check against Poiseuille.
"flow":          outlet mean velocity V_c and inlet pressure p_in are prescribed.
                 With constant R and no source terms, continuity forces
                 Q_i = pi R^2 V_c in every cell, so G_i follows from the radial solve
                 and the pressure is integrated from p_in.

Units are SI throughout.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Params:
    """Input data for :func:`solve`."""

    R: float                      # radius [m]
    L: float                      # bar length [m]
    mu: float                     # dynamic viscosity [Pa s]
    rho: float                    # density [kg/m^3]
    mode: str = "pressure_drop"   # "pressure_drop" or "flow"
    dp: float = 0.0               # pressure drop p_in - p_out [Pa] (pressure_drop mode)
    p_in: float = 0.0             # inlet pressure [Pa] (flow mode; reference otherwise)
    V_c: float = 0.0              # outlet mean velocity [m/s] (flow mode)
    Nx: int = 1                   # axial cells
    Nr: int = 40                  # radial cells
    dt: float = 1.0               # time step [s]
    t_end: float = 1.0            # final time [s]
    theta: float = 0.5            # 0.5 Crank-Nicolson, 1.0 backward Euler
    convection: bool = False      # include u du/dx (needs Nx > 1)
    save_times: tuple = ()        # times at which to store the full (Nx, Nr) field


@dataclass
class Result:
    """Output of :func:`solve`."""

    t: np.ndarray                 # time levels [s]
    x: np.ndarray                 # axial cell centres [m]
    eta: np.ndarray               # radial cell centres [-]
    u_final: np.ndarray           # velocity at t_end, shape (Nx, Nr) [m/s]
    Q_history: np.ndarray         # station-averaged flow rate per time level [m^3/s]
    G_final: np.ndarray           # dp/dx per axial cell at t_end [Pa/m]
    p_final: np.ndarray           # pressure at axial cell centres at t_end [Pa]
    p_out: float                  # pressure at the outlet face at t_end [Pa]
    snapshots: dict = field(default_factory=dict)  # time -> (Nx, Nr) field


def radial_operator(Nr: int, R: float, mu: float) -> np.ndarray:
    """Matrix Lm such that Lm @ u approximates (mu/R^2)(1/eta) d/deta(eta du/deta).

    Cell-centred finite volumes on eta in (0, 1). The face at eta = 0 carries no
    flux, and the wall at eta = 1 is a Dirichlet condition u = 0 imposed through
    a half-cell ghost distance.
    """
    deta = 1.0 / Nr
    eta_c = (np.arange(Nr) + 0.5) * deta
    eta_f = (np.arange(Nr) + 1.0) * deta  # eta of the face between cell j and j+1

    Lm = np.zeros((Nr, Nr))
    for j in range(Nr):
        if j < Nr - 1:
            c_out = eta_f[j] / (eta_c[j] * deta**2)
            Lm[j, j] -= c_out
            Lm[j, j + 1] += c_out
        else:
            # wall: u = 0 at eta = 1, located half a cell away from the centre
            Lm[j, j] -= 2.0 * eta_f[j] / (eta_c[j] * deta**2)
        if j > 0:
            c_in = eta_f[j - 1] / (eta_c[j] * deta**2)
            Lm[j, j] -= c_in
            Lm[j, j - 1] += c_in
    return (mu / R**2) * Lm


def solve(p: Params, u0: np.ndarray | None = None) -> Result:
    """Advance the transient problem from u0 (default: fluid at rest) to p.t_end."""
    if p.mode not in ("pressure_drop", "flow"):
        raise ValueError(f"unknown mode {p.mode!r}")
    if not 0.0 <= p.theta <= 1.0:
        raise ValueError("theta must lie in [0, 1]")
    if p.convection and p.Nx < 2:
        raise ValueError("convection requires Nx >= 2")

    dx = p.L / p.Nx
    x = (np.arange(p.Nx) + 0.5) * dx
    deta = 1.0 / p.Nr
    eta = (np.arange(p.Nr) + 0.5) * deta
    w = 2.0 * np.pi * p.R**2 * eta * deta  # Q = w @ u

    Lm = radial_operator(p.Nr, p.R, p.mu)
    M = (p.rho / p.dt) * np.eye(p.Nr) - p.theta * Lm
    Minv = np.linalg.inv(M)
    phi = Minv @ np.ones(p.Nr)   # response of u to a unit pressure gradient
    b = w @ phi                  # Q response to a unit pressure gradient (negative)

    Q_target = np.pi * p.R**2 * p.V_c
    G0 = -p.dp / p.L

    nsteps = int(round(p.t_end / p.dt))
    t = np.arange(nsteps + 1) * p.dt
    u = np.zeros((p.Nx, p.Nr)) if u0 is None else np.array(u0, dtype=float, copy=True)
    if u.shape != (p.Nx, p.Nr):
        raise ValueError(f"u0 must have shape {(p.Nx, p.Nr)}")

    save_steps = {int(round(ts / p.dt)) for ts in p.save_times}
    snapshots: dict = {}
    Q_hist = np.zeros(nsteps + 1)
    Q_hist[0] = np.mean(u @ w)
    G = np.full(p.Nx, G0)

    for n in range(nsteps):
        if p.convection:
            dudx = np.gradient(u, dx, axis=0, edge_order=2)
        else:
            dudx = np.zeros_like(u)

        rhs = (p.rho / p.dt) * u + (1.0 - p.theta) * (u @ Lm.T) - p.rho * u * dudx
        v = rhs @ Minv.T                      # solution with G = 0

        if p.mode == "pressure_drop":
            G = np.full(p.Nx, G0)
        else:
            G = ((v @ w) - Q_target) / b      # enforce Q_i = pi R^2 V_c

        u = v - G[:, None] * phi[None, :]
        Q_hist[n + 1] = np.mean(u @ w)
        if (n + 1) in save_steps:
            snapshots[float(t[n + 1])] = u.copy()

    cum = np.cumsum(G) * dx
    p_centre = p.p_in + (cum - 0.5 * G * dx)
    p_out = p.p_in + cum[-1]

    return Result(
        t=t,
        x=x,
        eta=eta,
        u_final=u,
        Q_history=Q_hist,
        G_final=G,
        p_final=p_centre,
        p_out=p_out,
        snapshots=snapshots,
    )
