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
        - AdaptiveSAPreconditioner: alpha-SA, a port of PyAMG's
          ``adaptive_sa_solver``; learns the near-null-space candidates from A
          (``adaptive_sa_hierarchy``) instead of assuming constants. Setup
          retains symmetric GS while the solve-time cycle defaults to Jacobi.
        - BootstrapAMGPreconditioner: Bootstrap AMG (BAMG); derives the C/F
          split, strength measure and interpolation weights from test
          vectors via compatible relaxation, algebraic distance and
          weighted least squares (``bootstrap.py``), instead of assuming an
          M-matrix sign structure. Its paper-aligned solve default remains GS.

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
        - GaussSeidelSmoother: Symmetric Gauss-Seidel smoother (SmootherBase).

    Cycles:
        - VCycle: Recursive V-cycle (gamma = 1).
        - WCycle: W-cycle with two coarse-grid corrections (gamma = 2).

    Coarsening:
        - AggregationCoarsening: Smoothed aggregation (SA-AMG).
        - TargetDimensionCoarsening: Aggregation coarsening driven by a
          target coarse dimension instead of theta, via exhaustive search.
        - BAMGCoarsening: Bootstrap AMG coarsening (compatible relaxation +
          algebraic distance + least-squares interpolation); the
          CoarseningStrategy extension point of the BAMG family.
        - NeuralCoarseningStrategy: Neural coarsening strategy (stub).

    Transfer operators:
        - DenseTransferOperator: P/R backed by a dense torch.Tensor.
        - NeuralTransferOperator: P/R backed by neural predictors (stub).

    Data:
        - MultigridHierarchy, MultigridLevel: Frozen dataclasses for the grid hierarchy.
"""

from .adaptive import AdaptiveSAPreconditioner, AdaptiveSAResult, adaptive_sa_hierarchy
from .amg import AMGPreconditioner
from .bootstrap import (
    BAMGCoarsening,
    BootstrapAMGPreconditioner,
    BootstrapAMGResult,
    BootstrapSetup,
)
from .coarsening import AggregationCoarsening, NeuralCoarseningStrategy, TargetDimensionCoarsening
from .cycle import VCycle, WCycle
from .hierarchy import MultigridHierarchy, MultigridLevel
from .protocols import CoarseningStrategy, MultigridCycle, MultigridSmoother, TransferOperator
from .smoothers import GaussSeidelSmoother, JacobiSmoother, SmootherBase
from .transfer import DenseTransferOperator, NeuralTransferOperator
from .variants import VCycleAMG, WCycleAMG

__all__ = [
    # Presets
    "VCycleAMG",
    "WCycleAMG",
    "AdaptiveSAPreconditioner",
    "AdaptiveSAResult",
    "adaptive_sa_hierarchy",
    "BootstrapAMGPreconditioner",
    "BootstrapAMGResult",
    "BootstrapSetup",
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
    "GaussSeidelSmoother",
    # Cycles
    "VCycle",
    "WCycle",
    # Coarsening
    "AggregationCoarsening",
    "TargetDimensionCoarsening",
    "BAMGCoarsening",
    "NeuralCoarseningStrategy",
    # Transfer operators
    "DenseTransferOperator",
    "NeuralTransferOperator",
    # Data
    "MultigridHierarchy",
    "MultigridLevel",
]
