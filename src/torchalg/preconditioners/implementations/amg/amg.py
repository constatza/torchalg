"""Re-export shim: ``AMGPreconditioner`` now lives in ``torchalg.multigrid.amg``.

Promoted (see ``docs/plan.md``) because it only ever treats its matrix as an
opaque buffer, calling injected ``CoarseningStrategy``/``MultigridCycle``
strategies - never committing to dense or sparse storage. Kept here so every
existing ``from .amg import AMGPreconditioner``/``from ...amg.amg import
AMGPreconditioner`` call site keeps working unchanged - mirrors
``transfer.py``'s ``DenseTransferOperator`` re-export convention.
"""

from __future__ import annotations

from torchalg.multigrid.amg import AMGPreconditioner

__all__ = ["AMGPreconditioner"]
