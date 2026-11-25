"""Identity preconditioner (no preconditioning).

The class is defined in ``torchalg.preconditioners.base`` so solvers can default to it
without depending on a concrete implementation; this module keeps the
``implementations`` import path.
"""

from __future__ import annotations

from ..base import Identity

__all__ = ["Identity"]
