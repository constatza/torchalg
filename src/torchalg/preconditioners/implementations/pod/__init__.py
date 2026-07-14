"""POD-2G (Proper Orthogonal Decomposition, two-grid) preconditioner package.

Reuses the generic multigrid engine from ``implementations.amg``
(``AMGPreconditioner``, ``VCycle``, ``JacobiSmoother``) with a POD-basis
coarsening strategy in place of algebraic aggregation, following
Nikolopoulos et al. (2022), §3.3-3.4.

Unlike the reference (``neuralls.domain.solver``), this package does **not**
define its own ``DenseTransferOperator``. The reference's POD
``DenseTransferOperator`` (prolongate = ``basis @ coarse``, restrict =
``basis.T @ fine``) and this repo's ``amg.DenseTransferOperator``
(prolongate = ``P @ coarse``, restrict = ``P.T @ fine``) are the exact same
operator - the POD basis Phi_r simply *is* the dense prolongation matrix
``P``. Defining a second, behaviorally-identical class here would violate
DRY for no benefit, so ``PODCoarseningStrategy.build_transfer`` constructs
``amg.DenseTransferOperator(self._basis)`` directly; see
``preconditioners.implementations.amg.transfer``, whose module docstring
already anticipated this reuse.

Public API:
    Preset (recommended entry point):
        - POD2GPreconditioner: POD-2G bundled with VCycle + JacobiSmoother.

    Core (for custom wiring, e.g. via AMGPreconditioner directly):
        - PODCoarseningStrategy: Builds a POD-reduced coarse level from a
          snapshot ensemble; satisfies the ``CoarseningStrategy`` protocol.
        - compute_pod_basis: Pure function computing the truncated POD basis
          (snapshot method) from a snapshot ensemble.

    Reused from ``.amg`` (not redefined here):
        - DenseTransferOperator: P/R backed by a dense tensor; satisfies the
          ``TransferOperator`` protocol. Import it from
          ``torchalg.preconditioners.implementations.amg``.

References:
    - Nikolopoulos, S., Kalogeris, I., Stavroulakis, G., & Papadopoulos, V.
      (2022). AI-enhanced iterative solvers for accelerating the solution of
      large-scale parametrized systems. arXiv:2207.02543.
"""

from .basis import compute_pod_basis
from .coarsening import PODCoarseningStrategy
from .variants import POD2GPreconditioner

__all__ = [
    # Preset
    "POD2GPreconditioner",
    # Core
    "PODCoarseningStrategy",
    "compute_pod_basis",
]
