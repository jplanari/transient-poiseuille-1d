"""Analytical reference solutions for the constant-radius, constant-viscosity case.

Steady state (Poiseuille), with G = dp/dx < 0 for flow in +x:

    u_s(eta) = -(G R^2 / 4 mu) (1 - eta^2),     Q = -pi R^4 G / (8 mu)

Transient start-up from rest (separation of variables, Fourier-Bessel modes):

    u(eta, t) = u_s(eta) + sum_n a_n J0(alpha_n eta) exp(-alpha_n^2 t / tau)

with J0(alpha_n) = 0, tau = rho R^2 / mu, c = -G R^2 / (4 mu), and

    a_n = -c I_n / (J1(alpha_n)^2 / 2),   I_n = int_0^1 eta (1 - eta^2) J0(alpha_n eta) d eta.

Using J0(alpha_n) = 0 the integral has the closed form I_n = 4 J1(alpha_n) / alpha_n^3, so

    a_n = -8 c / (alpha_n^3 J1(alpha_n)).
"""

from __future__ import annotations

import numpy as np
from scipy.special import j1, jn_zeros, j0


def steady_profile(eta, G: float, R: float, mu: float) -> np.ndarray:
    """Poiseuille velocity profile u_s(eta)."""
    eta = np.asarray(eta, dtype=float)
    return -(G * R**2 / (4.0 * mu)) * (1.0 - eta**2)


def steady_flow_rate(G: float, R: float, mu: float) -> float:
    """Poiseuille volumetric flow rate Q = -pi R^4 G / (8 mu)."""
    return -np.pi * R**4 * G / (8.0 * mu)


def viscous_time(R: float, mu: float, rho: float) -> float:
    """Radial viscous diffusion time tau = rho R^2 / mu."""
    return rho * R**2 / mu


def transient_profile(eta, t: float, G: float, R: float, mu: float, rho: float,
                      n_modes: int = 60) -> np.ndarray:
    """Start-up profile u(eta, t) from rest under a constant pressure gradient G."""
    eta = np.asarray(eta, dtype=float)
    c = -G * R**2 / (4.0 * mu)
    tau = viscous_time(R, mu, rho)

    alphas = jn_zeros(0, n_modes)
    coeffs = -8.0 * c / (alphas**3 * j1(alphas))

    u = c * (1.0 - eta**2)
    for a_n, alpha in zip(coeffs, alphas):
        u = u + a_n * j0(alpha * eta) * np.exp(-alpha**2 * t / tau)
    return u
