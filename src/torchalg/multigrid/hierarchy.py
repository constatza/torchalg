"""Multigrid hierarchy data model.

Ported from ``neuralls.domain.solver.preconditioners.implementations.amg.hierarchy``
with ``NDArray`` translated to ``torch.Tensor``. Stays
plain frozen dataclasses - not ``nn.Module`` - since the prepared hierarchy is
the state ``AMGPreconditioner.setup()`` builds and holds, not a buffer
registered directly on a module (see ``amg.py`` for the full
device/dtype-propagation reasoning).

Preparation is a type change, not a field fill: ``MultigridHierarchy`` is what a
coarsening strategy builds, ``PreparedHierarchy`` is what a cycle applies, so a
half-prepared hierarchy cannot be represented. ``prepare_hierarchy`` is the
only transition between the two.

Both hierarchy types hold their levels structurally: a tuple of fine levels
(each with a transfer to the next coarser level) plus exactly one coarsest
level, so a hierarchy without a coarsest level, or with one in the middle,
cannot be represented. ``levels`` is the read-only fine-then-coarsest view.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    import torch

    from .coarse_solve import CoarseSolve, CoarseSolverFactory
    from .protocols import (
        CoarseningStrategy,
        MultigridSmoother,
        PreparedSmoother,
        TransferOperator,
    )


@dataclass(frozen=True, slots=True)
class MultigridFineLevel:
    """An unprepared non-coarsest level, as a coarsening strategy builds it.

    Carries no per-matrix prepared state; ``prepare_hierarchy`` turns it into a
    ``PreparedFineLevel``.

    Attributes:
        matrix (torch.Tensor): System matrix on this grid level (n_k x n_k).
        transfer (TransferOperator): Operator that moves vectors to the next
            coarser level.
    """

    matrix: torch.Tensor
    transfer: TransferOperator


@dataclass(frozen=True, slots=True)
class MultigridCoarsestLevel:
    """An unprepared coarsest level: only its matrix; a direct solve is used there.

    Attributes:
        matrix (torch.Tensor): System matrix on the coarsest grid.
    """

    matrix: torch.Tensor


# The closed set of unprepared level kinds, in fine-then-coarsest order in ``levels``.
MultigridLevel = MultigridFineLevel | MultigridCoarsestLevel


@dataclass(frozen=True, slots=True)
class PreparedFineLevel:
    """A prepared non-coarsest level: its transfer and a smoother bound to its matrix.

    Attributes:
        matrix (torch.Tensor): System matrix on this grid level (n_k x n_k).
        transfer (TransferOperator): Operator that moves vectors to the next
            coarser level.
        smoother (PreparedSmoother): Smoother bound to ``matrix``.
    """

    matrix: torch.Tensor
    transfer: TransferOperator
    smoother: PreparedSmoother


@dataclass(frozen=True, slots=True)
class PreparedCoarsestLevel:
    """A prepared coarsest level: its matrix and the factorized direct solve.

    Attributes:
        matrix (torch.Tensor): System matrix on the coarsest grid.
        coarse_solve (CoarseSolve): Prepared direct solve of ``matrix``.
    """

    matrix: torch.Tensor
    coarse_solve: CoarseSolve


# The closed set of prepared level kinds; a cycle handles both exhaustively.
PreparedLevel = PreparedFineLevel | PreparedCoarsestLevel


def _require_kinds(
    fine: tuple[object, ...], coarsest: object, fine_type: type, coarsest_type: type
) -> None:
    """Reject, at construction, a coarsest level among the fine ones or the reverse.

    The annotations already forbid it; this makes the same mistake fail loudly where
    the type checker is not in the loop (``Any`` data, ``# type: ignore``).

    Raises:
        TypeError: If a ``fine`` entry is not a ``fine_type`` or ``coarsest`` is not a
            ``coarsest_type``.
    """
    if not isinstance(coarsest, coarsest_type):
        raise TypeError(
            f"coarsest must be a {coarsest_type.__name__}, got {type(coarsest).__name__}"
        )
    for position, level in enumerate(fine):
        if not isinstance(level, fine_type):
            raise TypeError(
                f"fine[{position}] must be a {fine_type.__name__}, got {type(level).__name__}"
            )


@dataclass(frozen=True, slots=True)
class MultigridHierarchy:
    """Unprepared multigrid hierarchy built by ``build_hierarchy``.

    Attributes:
        fine (tuple[MultigridFineLevel, ...]): Non-coarsest levels, finest first
            (empty for a single-level, direct-solve hierarchy).
        coarsest (MultigridCoarsestLevel): The one coarsest level.
    """

    fine: tuple[MultigridFineLevel, ...]
    coarsest: MultigridCoarsestLevel

    def __post_init__(self) -> None:
        """Validate the level kinds once; everything downstream trusts them."""
        _require_kinds(self.fine, self.coarsest, MultigridFineLevel, MultigridCoarsestLevel)

    @property
    def levels(self) -> tuple[MultigridLevel, ...]:
        """Read-only view: the fine levels followed by the coarsest one.

        Returns:
            tuple[MultigridLevel, ...]: Levels from finest (index 0) to coarsest (index -1).
        """
        return (*self.fine, self.coarsest)


@dataclass(frozen=True, slots=True)
class PreparedHierarchy:
    """Hierarchy whose every level is prepared; the only kind a cycle applies.

    Attributes:
        fine (tuple[PreparedFineLevel, ...]): Prepared non-coarsest levels, finest
            first (empty for a single-level, direct-solve hierarchy).
        coarsest (PreparedCoarsestLevel): The one prepared coarsest level.
    """

    fine: tuple[PreparedFineLevel, ...]
    coarsest: PreparedCoarsestLevel

    def __post_init__(self) -> None:
        """Validate the level kinds once; the cycles trust them."""
        _require_kinds(self.fine, self.coarsest, PreparedFineLevel, PreparedCoarsestLevel)

    @property
    def levels(self) -> tuple[PreparedLevel, ...]:
        """Read-only view: the prepared fine levels followed by the coarsest one.

        Returns:
            tuple[PreparedLevel, ...]: Levels from finest (index 0) to coarsest (index -1).
        """
        return (*self.fine, self.coarsest)


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
    fine: list[MultigridFineLevel] = []
    current_matrix = matrix
    for _ in range(n_levels - 1):
        coarse_matrix, transfer = coarsening.build_transfer(current_matrix)
        fine.append(MultigridFineLevel(matrix=current_matrix, transfer=transfer))
        current_matrix = coarse_matrix
    return MultigridHierarchy(
        fine=tuple(fine), coarsest=MultigridCoarsestLevel(matrix=current_matrix)
    )


def assemble_hierarchy(
    matrices: Sequence[torch.Tensor], transfers: Sequence[TransferOperator]
) -> MultigridHierarchy:
    """Hierarchy over level matrices and the transfers between consecutive levels.

    Args:
        matrices (Sequence[torch.Tensor]): Level matrices, finest first, at least one.
        transfers (Sequence[TransferOperator]): One transfer per level except the
            coarsest, finest first.

    Returns:
        MultigridHierarchy: Hierarchy sharing the given tensors.

    Raises:
        ValueError: If ``transfers`` is not exactly one shorter than ``matrices``.
    """
    *fine_matrices, coarsest = matrices
    if len(transfers) != len(fine_matrices):
        raise ValueError(
            f"{len(matrices)} level matrices need {len(fine_matrices)} transfers, "
            f"got {len(transfers)}"
        )
    return MultigridHierarchy(
        fine=tuple(
            MultigridFineLevel(matrix, transfer)
            for matrix, transfer in zip(fine_matrices, transfers, strict=True)
        ),
        coarsest=MultigridCoarsestLevel(coarsest),
    )


def prepare_hierarchy(
    hierarchy: MultigridHierarchy | PreparedHierarchy,
    smoother: MultigridSmoother,
    coarse_solver_factory: CoarseSolverFactory,
) -> PreparedHierarchy:
    """Bind smoothers to the non-coarsest levels and factorize the coarsest, once.

    Idempotent: an already prepared hierarchy is returned as is, so repeated
    preparation never redoes factorizations or spectral-radius estimates.

    Args:
        hierarchy (MultigridHierarchy | PreparedHierarchy): Hierarchy to prepare.
        smoother (MultigridSmoother): Prepared against each non-coarsest level matrix.
        coarse_solver_factory (CoarseSolverFactory): Builds the solve from the coarsest
            matrix.

    Returns:
        PreparedHierarchy: Hierarchy whose levels all carry their prepared state.
    """
    match hierarchy:
        case PreparedHierarchy():
            return hierarchy
        case MultigridHierarchy(fine=fine, coarsest=coarsest):
            return PreparedHierarchy(
                fine=tuple(
                    PreparedFineLevel(level.matrix, level.transfer, smoother.prepare(level.matrix))
                    for level in fine
                ),
                coarsest=PreparedCoarsestLevel(
                    coarsest.matrix, coarse_solver_factory(coarsest.matrix)
                ),
            )
