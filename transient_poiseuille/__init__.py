"""Transient 1D (axial) / radial velocity solver for a bar of constant radius."""

from .solver import Params, Result, radial_operator, solve
from . import analytic

__all__ = ["Params", "Result", "radial_operator", "solve", "analytic"]
__version__ = "0.1.0"
