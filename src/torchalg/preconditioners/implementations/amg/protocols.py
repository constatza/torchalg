"""Re-export shim: AMG protocols now live in ``torchalg.multigrid.protocols``.

Promoted (see ``docs/plan.md``) because these ``Protocol`` definitions never
committed to dense or sparse storage. Kept here so every existing
``from .protocols import ...``/``from ...amg.protocols import ...`` call
site keeps working unchanged - mirrors ``transfer.py``'s
``DenseTransferOperator`` re-export convention.
"""

from __future__ import annotations

from torchalg.multigrid.protocols import (
    CoarseningStrategy,
    MultigridCycle,
    MultigridSmoother,
    TransferOperator,
)

__all__ = ["CoarseningStrategy", "MultigridCycle", "MultigridSmoother", "TransferOperator"]
