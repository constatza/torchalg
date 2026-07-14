"""AMG (Algebraic Multigrid) preconditioner package.

Ported from ``neuralls.domain.solver.preconditioners.implementations.amg``
(see ``docs/plan.md``'s Stage 5 entry), dense-only throughout - no
``scipy.sparse``, no ``torch.sparse``. Two renames from the reference (both
noted in the Stage 5 port report): ``SparseAggregationCoarsening`` ->
``AggregationCoarsening`` and ``SparseTransferOperator`` ->
``DenseTransferOperator``, since both are now dense-``torch.Tensor``-backed,
not scipy-sparse-backed.

Public API:
    Presets (recommended entry points):
        - VCycleAMG: SA-AMG with V-cycle; standard default.
        - WCycleAMG: SA-AMG with W-cycle; more robust, higher cost per cycle.

    Core (for custom wiring):
        - AMGPreconditioner: Top-level preconditioner; implements Preconditioner + BindableInputs.

    Protocols (OCP extension points):
        - TransferOperator: P/R interface (dense, neural, ...).
        - MultigridSmoother: Smoothing interface (Jacobi, Gauss-Seidel, ...).
        - CoarseningStrategy: Coarsening interface (aggregation, RS, neural, ...).
        - MultigridCycle: Cycle interface (V, W, F, ...).

    Smoother hierarchy:
        - SmootherBase: ABC for fixed-step error dampers.
        - JacobiSmoother: Weighted Jacobi smoother (SmootherBase).

    Cycles:
        - VCycle: Recursive V-cycle (gamma = 1).
        - WCycle: W-cycle with two coarse-grid corrections (gamma = 2).

    Coarsening:
        - AggregationCoarsening: Smoothed aggregation (SA-AMG).
        - NeuralCoarseningStrategy: Neural coarsening strategy (stub).

    Transfer operators:
        - DenseTransferOperator: P/R backed by a dense torch.Tensor.
        - NeuralTransferOperator: P/R backed by neural predictors (stub).

    Data:
        - MultigridHierarchy, MultigridLevel: Frozen dataclasses for the grid hierarchy.
"""

from .amg import AMGPreconditioner
from .coarsening import AggregationCoarsening, NeuralCoarseningStrategy
from .cycle import VCycle, WCycle
from .hierarchy import MultigridHierarchy, MultigridLevel
from .protocols import CoarseningStrategy, MultigridCycle, MultigridSmoother, TransferOperator
from .smoothers import JacobiSmoother, SmootherBase
from .transfer import DenseTransferOperator, NeuralTransferOperator
from .variants import VCycleAMG, WCycleAMG

__all__ = [
    # Presets
    "VCycleAMG",
    "WCycleAMG",
    # Core
    "AMGPreconditioner",
    # Protocols
    "TransferOperator",
    "MultigridSmoother",
    "CoarseningStrategy",
    "MultigridCycle",
    # Smoother hierarchy
    "SmootherBase",
    "JacobiSmoother",
    # Cycles
    "VCycle",
    "WCycle",
    # Coarsening
    "AggregationCoarsening",
    "NeuralCoarseningStrategy",
    # Transfer operators
    "DenseTransferOperator",
    "NeuralTransferOperator",
    # Data
    "MultigridHierarchy",
    "MultigridLevel",
]
