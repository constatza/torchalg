"""Typing-only Protocol-conformance pins for ``torchalg.sparse``'s sibling classes.

See ``docs/plan.md``'s "Next steps" (Protocol-conformance pins) / "SOLID
review summary" (Interface Segregation): ``CoarseningStrategy``/
``TransferOperator`` conformance for every sparse sibling class was asserted
only in docstring prose - nothing imported ``torchalg.multigrid.protocols``
from ``torchalg.sparse`` or its tests, so a signature drift in a sparse
sibling would pass ``pytest``/``ruff``/``tach`` without ever surfacing.

Everything below lives under ``if TYPE_CHECKING:`` so it has zero runtime
cost or side effect (no object is ever actually constructed) - the entire
point is static verification by ``uv run ty check src/``, the project's
standard command: ``ty`` still type-checks code inside a ``TYPE_CHECKING``
block (that's the documented purpose of the flag), so each annotated
assignment below is rejected at check time if its right-hand-side instance
doesn't structurally satisfy the left-hand-side Protocol annotation. This
module is never imported by anything; ``ty check src/`` discovers and checks
every file under ``src/``, not just reachable ones, so a standalone pin file
is sufficient.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

    from torchalg.multigrid.protocols import CoarseningStrategy, TransferOperator
    from torchalg.sparse.preconditioners.amg.coarsening import (
        AggregationCoarsening,
        TargetDimensionCoarsening,
    )
    from torchalg.sparse.preconditioners.amg.transfer import SparseTransferOperator
    from torchalg.sparse.preconditioners.pod.coarsening import PODCoarseningStrategy

    _aggregation_coarsening: CoarseningStrategy = AggregationCoarsening()
    _target_dimension_coarsening: CoarseningStrategy = TargetDimensionCoarsening(
        target_coarse_dim=4, theta_min=0.1, theta_max=0.9, step=0.1
    )
    _pod_coarsening_strategy: CoarseningStrategy = PODCoarseningStrategy(rank=4)
    _sparse_transfer_operator: TransferOperator = SparseTransferOperator(torch.empty(0))
