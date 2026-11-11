"""Transient 1D (axial) / radial velocity solver for a bar of constant radius."""

from . import analytic
from .config import RunConfig, load_config
from .solver import Params, Result, radial_operator, solve

__all__ = ["Params", "Result", "RunConfig", "radial_operator", "solve",
           "load_config", "analytic"]
__version__ = "0.2.0"
