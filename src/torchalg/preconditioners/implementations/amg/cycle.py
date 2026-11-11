"""Re-export shim: ``VCycle``/``WCycle`` now live in ``torchalg.multigrid.cycle``.

Promoted (see ``docs/plan.md``) because both cycles only ever do
``matrix @ x`` or delegate to injected strategies - never committing to
dense or sparse storage. Kept here so every existing ``from .cycle import
...``/``from ...amg.cycle import ...`` call site keeps working unchanged -
mirrors ``transfer.py``'s ``DenseTransferOperator`` re-export convention.
"""

from __future__ import annotations

from torchalg.multigrid.cycle import VCycle, WCycle

__all__ = ["VCycle", "WCycle"]
