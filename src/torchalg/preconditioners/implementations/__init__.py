"""Concrete preconditioner implementations.

Populated incrementally, stage by stage (see ``docs/plan.md``). So far
(Stages 3-5):

- ``Identity``: No preconditioning.
- ``JacobiPreconditioner``: Diagonal scaling.
- ``CallablePreconditioner``: Wrap an arbitrary function.
- ``LinearOperatorPreconditioner``: Wrap an arbitrary linear-operator
  callable.
- ``ScheduledPreconditioner``: Switch preconditioners based on iteration.
- ``ILUPreconditioner``: Dense masked incomplete LU (ILU(0)).
- ``IC0Preconditioner``: Dense masked zero-level incomplete Cholesky
  (IC(0)).
- ``ICholeskyPreconditioner``: Incomplete Cholesky using an externally
  supplied factor ``L``.
- ``AMGPreconditioner``: Algebraic Multigrid (smoothed aggregation or
  neural P/R); see ``.amg`` for the full public API (presets, protocols,
  cycles, smoothers, coarsening strategies, transfer operators).
- ``AdaptiveSAPreconditioner``: Adaptive smoothed aggregation (alpha-SA);
  test vectors learned from ``A``, see ``.amg.adaptive``.
- ``BootstrapAMGPreconditioner``: Bootstrap AMG (BAMG); C/F split, strength
  measure and interpolation all derived from test vectors via compatible
  relaxation, algebraic distance and weighted least squares, see
  ``.amg.bootstrap``.
- ``POD2GPreconditioner``: Proper Orthogonal Decomposition two-grid
  preconditioner, reusing the ``.amg`` engine with POD-basis coarsening; see
  ``.pod`` for the full public API.
- ``NeuralPreconditioner``: Wraps a caller-supplied predictor (via
  ``preconditioners.ports.PredictorAdapter``/``ExtraInputPredictorPort``)
  as a non-linear preconditioner.
"""

from .amg import AdaptiveSAPreconditioner, AMGPreconditioner, BootstrapAMGPreconditioner
from .callable import CallablePreconditioner
from .ic0 import IC0Preconditioner
from .icholesky import ICholeskyPreconditioner
from .identity import Identity
from .ilu import ILUPreconditioner
from .jacobi import JacobiPreconditioner
from .linear_operator import LinearOperatorPreconditioner
from .neural import NeuralPreconditioner
from .pod import POD2GPreconditioner
from .scheduled import ScheduledPreconditioner

__all__ = [
    "AdaptiveSAPreconditioner",
    "AMGPreconditioner",
    "BootstrapAMGPreconditioner",
    "CallablePreconditioner",
    "IC0Preconditioner",
    "ICholeskyPreconditioner",
    "ILUPreconditioner",
    "Identity",
    "JacobiPreconditioner",
    "LinearOperatorPreconditioner",
    "NeuralPreconditioner",
    "POD2GPreconditioner",
    "ScheduledPreconditioner",
]
