"""Transient axisymmetric flow in a bar of constant radius, with viscosity mu(x).

For each axial cell i (length dx) the radial velocity u_i(eta, t), with eta = r/R,
obeys

    rho * (du/dt + u du/dx) = -G_i + (mu_i / R^2) * (1/eta) d/deta (eta du/deta)

with  du/deta = 0 at eta = 0 (symmetry),  u = 0 at eta = 1 (wall),
and G_i = dp/dx uniform over the section of cell i. Viscosity mu_i is constant
over each section, so it may vary along the axis but not across it.

Discretisation
--------------
* Radial: cell-centred finite volumes in eta, wall handled with a half-cell ghost.
* Time: theta-scheme (theta = 0.5 Crank-Nicolson, theta = 1 backward Euler).
* Convection (optional): explicit, first-order upwind in x, lagged. Requires the
  CFL condition max|u| dt / dx <= 1, checked at every step.

Coupling along the axis
-----------------------
Each cell gives Q_i = a_i - b_i G_i, with a_i the flow from the previous state and
b_i = pi R^4 / (8 mu_i) in the steady limit. Continuity (constant R, no source)
forces one common Q, so G_i = (a_i - Q)/b_i.

Drive modes
-----------
"flow":          the outlet mean velocity V_c(t) is prescribed, so Q = pi R^2 V_c(t).
                 V_c may be a number or a callable of time (e.g. a square wave, see
                 drive.py). The pressure is integrated from p_in.
"pressure_drop": the total drop dp = p_in - p_out is prescribed. Q is found from
                 sum_i G_i dx = -dp. With constant mu this gives a uniform G.

Units are SI throughout.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Params:
    """Input data for :func:`solve`.

    ``mu`` is either a scalar (uniform viscosity) or an array of length ``Nx``
    giving the viscosity of each axial cell.
    """

    R: float                      # radius [m]
    L: float                      # bar length [m]
    mu: object                    # viscosity [Pa s]: scalar or array of length Nx
    rho: float                    # density [kg/m^3]
    mode: str = "pressure_drop"   # "pressure_drop" or "flow"
    dp: float = 0.0               # total pressure drop p_in - p_out [Pa] (pressure_drop mode)
    p_in: float = 0.0             # inlet pressure [Pa]
    V_c: object = 0.0             # outlet mean velocity [m/s]: number or V(t) callable (flow mode)
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


def mu_per_station(p: Params) -> np.ndarray:
    """Viscosity of each axial cell, shape (Nx,). Validates the input."""
    mu = np.asarray(p.mu, dtype=float)
    if mu.ndim == 0:
        mu = np.full(p.Nx, float(mu))
    elif mu.shape != (p.Nx,):
        raise ValueError(f"mu must be a scalar or have length Nx = {p.Nx}, got {mu.shape[0]}")
    if np.any(~np.isfinite(mu)) or np.any(mu <= 0.0):
        raise ValueError("mu must be finite and positive")
    return mu


def mu_reference(p: Params) -> float:
    """Mean viscosity, used for time scales and summaries."""
    return float(np.mean(mu_per_station(p)))


def is_uniform_mu(p: Params) -> bool:
    """True when every axial cell has the same viscosity."""
    mu = mu_per_station(p)
    return bool(np.all(mu == mu[0]))


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


def _upwind_dudx(u: np.ndarray, dx: float) -> np.ndarray:
    """First-order upwind approximation of du/dx along the axis (axis 0).

    The stencil follows the sign of u: backward difference where u >= 0, forward
    where u < 0. No upstream value exists at the ends, so the derivative is set to
    zero there. Centred differences are avoided because the explicit centred
    advection scheme is unconditionally unstable.
    """
    back = np.zeros_like(u)
    fwd = np.zeros_like(u)
    diff = (u[1:] - u[:-1]) / dx
    back[1:] = diff
    fwd[:-1] = diff
    return np.where(u >= 0.0, back, fwd)


def solve(p: Params, u0: np.ndarray | None = None) -> Result:
    """Advance the transient problem from u0 (default: fluid at rest) to p.t_end."""
    if p.mode not in ("pressure_drop", "flow"):
        raise ValueError(f"unknown mode {p.mode!r}")
    if not 0.0 <= p.theta <= 1.0:
        raise ValueError("theta must lie in [0, 1]")
    if p.convection and p.Nx < 2:
        raise ValueError("convection requires Nx >= 2")
    mu = mu_per_station(p)

    dx = p.L / p.Nx
    x = (np.arange(p.Nx) + 0.5) * dx
    deta = 1.0 / p.Nr
    eta = (np.arange(p.Nr) + 0.5) * deta
    w = 2.0 * np.pi * p.R**2 * eta * deta  # Q = w @ u

    # radial operator per unit viscosity; the station operator is mu_i * Lunit
    Lunit = radial_operator(p.Nr, p.R, 1.0)
    I = np.eye(p.Nr)
    M = (p.rho / p.dt) * I[None, :, :] - p.theta * mu[:, None, None] * Lunit[None, :, :]
    Minv = np.linalg.inv(M)                          # (Nx, Nr, Nr)
    phi = np.einsum("ijk,k->ij", Minv, np.ones(p.Nr))  # (Nx, Nr): response to unit G
    b = phi @ w                                      # (Nx,): Q response to unit G (> 0)

    if callable(p.V_c):
        V_of_t = p.V_c
    else:
        V_const = float(p.V_c)
        V_of_t = lambda _t: V_const  # noqa: E731

    nsteps = int(round(p.t_end / p.dt))
    t = np.arange(nsteps + 1) * p.dt
    u = np.zeros((p.Nx, p.Nr)) if u0 is None else np.array(u0, dtype=float, copy=True)
    if u.shape != (p.Nx, p.Nr):
        raise ValueError(f"u0 must have shape {(p.Nx, p.Nr)}")

    save_steps = {int(round(ts / p.dt)) for ts in p.save_times}
    snapshots: dict = {}
    Q_hist = np.zeros(nsteps + 1)
    Q_hist[0] = np.mean(u @ w)
    G = np.zeros(p.Nx)

    for n in range(nsteps):
        if p.convection:
            umax = float(np.max(np.abs(u)))
            cfl = umax * p.dt / dx
            if cfl > 1.0:
                raise ValueError(
                    f"convective CFL = {cfl:.2f} > 1 at t = {t[n]:.4g} s: the explicit "
                    f"upwind convection would be unstable; reduce numerics.dt to "
                    f"<= {dx / umax:.3g} s")
            dudx = _upwind_dudx(u, dx)
        else:
            dudx = np.zeros_like(u)

        Lu = mu[:, None] * (u @ Lunit.T)             # (mu_i / R^2) Lm u_i
        rhs = (p.rho / p.dt) * u + (1.0 - p.theta) * Lu - p.rho * u * dudx
        v = np.einsum("ijk,ik->ij", Minv, rhs)       # solution with G = 0
        a = v @ w                                    # (Nx,)

        if p.mode == "flow":
            Q_target = np.pi * p.R**2 * V_of_t(float(t[n + 1]))
            G = (a - Q_target) / b
        else:
            # sum_i G_i dx = -dp  with  G_i = (a_i - Q) / b_i  ->  solve for Q
            Q = (np.sum(a / b) * dx + p.dp) / (np.sum(1.0 / b) * dx)
            G = (a - Q) / b

        u = v - G[:, None] * phi
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
