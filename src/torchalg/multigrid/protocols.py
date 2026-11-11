"""AMG protocols - minimal extension points for multigrid components.

Ported from ``neuralls.domain.solver.preconditioners.implementations.amg.protocols``
(see ``docs/plan.md``) with ``NDArray`` translated to ``torch.Tensor``. These
stay plain ``typing.Protocol`` definitions (structural typing, OCP extension
points), matching the reference exactly - no ``nn.Module`` involvement here,
since a ``Protocol`` never owns tensor state itself.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    import torch

    from .hierarchy import MultigridHierarchy


class TransferOperator(Protocol):
    """Moves vectors between a fine and its coarse grid.

    OCP hook: implement to add dense or neural P/R operators.
    """

    def prolongate(self, coarse: torch.Tensor) -> torch.Tensor:
        """Map coarse-grid vector to fine grid (interpolation).

        Args:
            coarse (torch.Tensor): Coarse-grid vector of length n_coarse.

        Returns:
            torch.Tensor: Fine-grid vector of length n_fine.
        """
        ...

    def restrict(self, fine: torch.Tensor) -> torch.Tensor:
        """Map fine-grid vector to coarse grid (restriction).

        Args:
            fine (torch.Tensor): Fine-grid vector of length n_fine.

        Returns:
            torch.Tensor: Coarse-grid vector of length n_coarse.
        """
        ...


class PreparedSmoother(Protocol):
    """A smoother bound to one level's matrix, with all matrix-dependent work done.

    Built once per level by ``MultigridSmoother.prepare`` (diagonal inverse, spectral-radius
    damping, triangular factors, level schedules, ...); every call afterwards only does
    sweeps.
    """

    def smooth(self, rhs: torch.Tensor, x: torch.Tensor, steps: int) -> torch.Tensor:
        """Apply ``steps`` smoothing iterations to ``A x = rhs`` from the iterate ``x``.

        Args:
            rhs (torch.Tensor): Right-hand side vector.
            x (torch.Tensor): Current iterate (not modified).
            steps (int): Number of smoothing steps.

        Returns:
            torch.Tensor: Updated iterate after smoothing.
        """
        ...

    def smooth_from_zero(self, rhs: torch.Tensor, steps: int) -> torch.Tensor:
        """Apply ``steps`` smoothing iterations from the zero initial guess.

        Equivalent to ``smooth(rhs, zeros_like(rhs), steps)`` but free to skip the
        residual matvec of the first sweep (``A @ 0``).

        Args:
            rhs (torch.Tensor): Right-hand side vector (or block of columns).
            steps (int): Number of smoothing steps.

        Returns:
            torch.Tensor: Iterate after smoothing.
        """
        ...


class MultigridSmoother(Protocol):
    """Damps high-frequency error on one grid level.

    OCP hook: implement to add Gauss-Seidel, polynomial, or other smoothers. A smoother
    object holds only configuration (damping override, ...); the per-matrix state lives in
    the ``PreparedSmoother`` it returns, so one smoother serves many matrices.
    """

    def prepare(self, A: torch.Tensor) -> PreparedSmoother:
        """Do every matrix-dependent one-off computation for ``A``.

        Args:
            A (torch.Tensor): System matrix on this level (dense or sparse CSR).

        Returns:
            PreparedSmoother: Smoother bound to ``A``.
        """
        ...


class CoarseningStrategy(Protocol):
    """Builds one coarse level from a fine-grid matrix.

    OCP hook: implement to add classical AMG, aggregation, or neural coarsening.
    Neural variants additionally implement ``BindableInputs`` so that spatial
    data (positions, parameters) can be bound before ``build_transfer`` is called.
    """

    def build_transfer(self, A: torch.Tensor) -> tuple[torch.Tensor, TransferOperator]:
        """Build a coarse grid and the associated transfer operator.

        The matrix A is the only argument because it is always available at
        hierarchy-construction time. Neural strategies pre-bind any other
        needed tensors (positions, theta) via ``bind_inputs`` before this call.

        Args:
            A (torch.Tensor): Fine-grid matrix (n x n).

        Returns:
            tuple[torch.Tensor, TransferOperator]: ``(A_coarse, transfer)``
                where ``A_coarse`` is the Galerkin coarse matrix
                (n_c x n_c) and ``transfer`` maps between the two grids.
        """
        ...


class MultigridCycle(Protocol):
    """Performs one multigrid correction cycle (V, W, F, ...).

    OCP hook: implement to add W-cycle, F-cycle, or custom cycles.
    """

    def prepare(self, hierarchy: MultigridHierarchy) -> MultigridHierarchy:
        """Do the one-off, hierarchy-dependent work ``apply`` needs (e.g. factor the coarse level).

        Idempotent. Must not store state on the cycle: one cycle instance serves many
        hierarchies.

        Args:
            hierarchy (MultigridHierarchy): Freshly built hierarchy.

        Returns:
            MultigridHierarchy: Hierarchy that ``apply`` accepts.
        """
        ...

    def apply(self, hierarchy: MultigridHierarchy, rhs: torch.Tensor) -> torch.Tensor:
        """Compute an approximate solution to the system on the finest grid.

        Args:
            hierarchy (MultigridHierarchy): Pre-built multigrid hierarchy.
            rhs (torch.Tensor): Right-hand side vector on the finest grid.

        Returns:
            torch.Tensor: Approximate solution vector on the finest grid.
        """
        ...
