"""Numerical diagnostics that analyze solver operators.

Analysis routines are kept outside :mod:`torchalg.utils`: they are optional,
potentially expensive diagnostics rather than dependency-floor primitives.
"""

from .spectral import condition_number, preconditioned_condition_number

__all__ = ["condition_number", "preconditioned_condition_number"]
