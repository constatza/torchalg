"""Multigrid hierarchy data model.

Ported from ``neuralls.domain.solver.preconditioners.implementations.amg.hierarchy``
(see ``docs/plan.md``) with ``NDArray`` translated to ``torch.Tensor``. Stays
a plain frozen dataclass pair - not ``nn.Module`` - since the hierarchy is a
transient, lazily-(re)built cache owned by ``AMGPreconditioner``, not a
buffer registered directly on a module (see ``amg.py`` for the full
device/dtype-propagation reasoning).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

    from .coarse_solve import CoarseSolve, CoarseSolverFactory
    from .protocols import (
        CoarseningStrategy,
        MultigridSmoother,
        PreparedSmoother,
        TransferOperator,
    )


@dataclass(frozen=True, slots=True)
class MultigridLevel:
    """One level in a multigrid hierarchy.

    Attributes:
        matrix (torch.Tensor): System matrix on this grid level (n_k x n_k).
        transfer (TransferOperator | None): Operator that moves vectors to
            the next coarser level. ``None`` on the coarsest level - direct
            solve is used there.
        coarse_solve (CoarseSolve | None): Prepared direct solve of ``matrix``,
            attached to the coarsest level by ``cycle.prepare(hierarchy)``.
            ``None`` on every other level and on an unprepared hierarchy.
        smoother (PreparedSmoother | None): Smoother bound to ``matrix``, attached
            to every non-coarsest level by ``cycle.prepare(hierarchy)``. ``None``
            on the coarsest level and on an unprepared hierarchy.
    """

    matrix: torch.Tensor
    transfer: TransferOperator | None
    coarse_solve: CoarseSolve | None = None
    smoother: PreparedSmoother | None = None


@dataclass(frozen=True, slots=True)
class MultigridHierarchy:
    """Full multigrid hierarchy built by ``AMGPreconditioner``.

    Attributes:
        levels (tuple[MultigridLevel, ...]): Levels from finest (index 0) to
            coarsest (index -1).
    """

    levels: tuple[MultigridLevel, ...]


def build_hierarchy(
    matrix: torch.Tensor, coarsening: CoarseningStrategy, n_levels: int
) -> MultigridHierarchy:
    """Iteratively apply ``coarsening`` to build all levels.

    Pure function of its arguments (apart from any state the strategy itself
    keeps), so trial hierarchies can be built without instantiating an
    ``AMGPreconditioner``.

    Args:
        matrix (torch.Tensor): Finest-level system matrix.
        coarsening (CoarseningStrategy): Strategy that builds each coarse
            level.
        n_levels (int): Total number of levels, including the finest.

    Returns:
        MultigridHierarchy: Frozen hierarchy of levels from fine to coarse.
    """
    levels: list[MultigridLevel] = []
    current_matrix = matrix
    for _ in range(n_levels - 1):
        coarse_matrix, transfer = coarsening.build_transfer(current_matrix)
        levels.append(MultigridLevel(matrix=current_matrix, transfer=transfer))
        current_matrix = coarse_matrix
    levels.append(MultigridLevel(matrix=current_matrix, transfer=None))
    return MultigridHierarchy(levels=tuple(levels))


def prepare_hierarchy(
    hierarchy: MultigridHierarchy,
    smoother: MultigridSmoother,
    coarse_solver_factory: CoarseSolverFactory,
) -> MultigridHierarchy:
    """Attach prepared smoothers and the prepared coarse solve, once.

    Idempotent per level: a level that already carries its prepared state is kept as is,
    so repeated preparation never redoes factorizations or spectral-radius estimates.

    Args:
        hierarchy (MultigridHierarchy): Hierarchy to prepare.
        smoother (MultigridSmoother): Prepared against each non-coarsest level matrix.
        coarse_solver_factory (CoarseSolverFactory): Builds the solve from the coarsest
            matrix.

    Returns:
        MultigridHierarchy: Hierarchy whose levels carry their prepared state.
    """
    levels = tuple(
        _prepared_level(level, smoother, coarse_solver_factory) for level in hierarchy.levels
    )
    if all(new is old for new, old in zip(levels, hierarchy.levels, strict=True)):
        return hierarchy
    return MultigridHierarchy(levels=levels)


def _prepared_level(
    level: MultigridLevel, smoother: MultigridSmoother, factory: CoarseSolverFactory
) -> MultigridLevel:
    if level.transfer is None:
        if level.coarse_solve is not None:
            return level
        return replace(level, coarse_solve=factory(level.matrix))
    if level.smoother is not None:
        return level
    return replace(level, smoother=smoother.prepare(level.matrix))
