"""Format-agnostic multigrid engine, shared by the dense and sparse AMG trees.

Promoted out of ``torchalg.preconditioners.implementations.amg`` (see
``docs/plan.md``): after a full read, ``hierarchy.py``/``cycle.py``/``amg.py``/
``protocols.py`` were found to never commit to a dense or sparse matrix
format anywhere except the coarsest-level direct solve (which is itself
injectable via each cycle's ``coarse_solver`` parameter). ``build_hierarchy``
only calls the injected ``CoarseningStrategy.build_transfer``; ``VCycle``/
``WCycle`` only do ``matrix @ x`` (works for dense or sparse CSR) and call
the injected ``MultigridSmoother``/``TransferOperator``; ``AMGPreconditioner``
only treats its matrix as an opaque buffer. Both ``torchalg.preconditioners
.implementations.amg`` (dense) and ``torchalg.sparse.preconditioners.amg``
(sparse) consume this package instead of duplicating it; the dense tree's
old ``amg/{amg,cycle,hierarchy,protocols}.py`` modules are now thin
re-export shims pointing here, so existing imports keep working unchanged.
"""

from __future__ import annotations

from .amg import AMGPreconditioner
from .cycle import VCycle, WCycle, pseudo_inverse_solve
from .hierarchy import MultigridHierarchy, MultigridLevel, build_hierarchy
from .protocols import CoarseningStrategy, MultigridCycle, MultigridSmoother, TransferOperator

__all__ = [
    "AMGPreconditioner",
    "VCycle",
    "WCycle",
    "pseudo_inverse_solve",
    "MultigridHierarchy",
    "MultigridLevel",
    "build_hierarchy",
    "CoarseningStrategy",
    "MultigridCycle",
    "MultigridSmoother",
    "TransferOperator",
]
