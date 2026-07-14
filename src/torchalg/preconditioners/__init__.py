"""Preconditioner abstractions and implementations for iterative solvers.

Ported from ``neuralls.domain.solver.preconditioners`` (see
``docs/plan.md``), populated incrementally, stage by stage:

- ``base``: ``Preconditioner`` interface, ``PreconditionerContext``,
  ``BindableInputs``, ``LinearPreconditioner``, ``NonLinearPreconditioner``
  (Stage 3).
- ``implementations``: concrete preconditioners - ``Identity``,
  ``JacobiPreconditioner``, ``CallablePreconditioner``,
  ``LinearOperatorPreconditioner``, ``ScheduledPreconditioner`` (Stage 3),
  ``ILUPreconditioner``, ``IC0Preconditioner``, ``ICholeskyPreconditioner``
  (Stage 4, dense-only - no scipy/numba/sparse in ``src/``) so far;
  AMG/POD/Neural land in Stages 5-7.
- ``ports``: predictor adapter protocols for the neural preconditioner,
  populated in Stage 7.

Deliberately no re-exports here (unlike ``models``/``strategies``): ``base``
and ``implementations`` are separate ``tach`` modules with different
``depends_on`` lists (see ``docs/plan.md``'s "Dependency architecture
(tach)" section) - the DIP boundary between the preconditioner *interface*
and its *concrete implementations* is real, not just documented, so this
package's own ``__init__.py`` stays free of any import that would collapse
it. Import directly from the submodule you need, e.g.::

    from torchalg.preconditioners.base import Preconditioner
    from torchalg.preconditioners.implementations import Identity
"""
