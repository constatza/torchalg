"""Multigrid cycle implementations.

Ported from ``neuralls.domain.solver.preconditioners.implementations.amg.cycle``
(see ``docs/plan.md``) with ``NDArray`` translated to ``torch.Tensor`` and
``numpy.linalg.solve`` replaced by ``torch.linalg.solve`` for the coarsest
direct solve - both cycles remain dense-only throughout, matching the
hierarchy's dense matrix storage.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from .hierarchy import MultigridHierarchy

if TYPE_CHECKING:
    from .protocols import MultigridSmoother


class VCycle:
    """Standard V-cycle multigrid preconditioner (gamma = 1).

    Descends the hierarchy pre-smoothing, computes a coarse-grid correction,
    then ascends post-smoothing. One recursive call per level.

    Args:
        smoother (MultigridSmoother): Smoother applied before and after the
            coarse-grid correction.
        n_pre (int): Pre-smoothing steps.
        n_post (int): Post-smoothing steps.

    References:
        - Briggs, W. L., Henson, V. E., & McCormick, S. F. (2000).
          A Multigrid Tutorial (2nd ed.). SIAM. Algorithm 3.7, Section 3.2.
        - Trottenberg, U., Oosterlee, C. W., & Schuller, A. (2001).
          Multigrid. Academic Press. Section 2.2.1.
    """

    def __init__(
        self,
        smoother: MultigridSmoother,
        n_pre: int = 2,
        n_post: int = 2,
    ) -> None:
        """Store the smoother and pre/post-smoothing step counts.

        Args:
            smoother (MultigridSmoother): Smoother applied before and after
                the coarse-grid correction.
            n_pre (int): Pre-smoothing steps.
            n_post (int): Post-smoothing steps.
        """
        self._smoother = smoother
        self._n_pre = n_pre
        self._n_post = n_post

    def apply(self, hierarchy: MultigridHierarchy, rhs: torch.Tensor) -> torch.Tensor:
        """Compute one V-cycle starting from the finest level.

        Args:
            hierarchy (MultigridHierarchy): Pre-built multigrid hierarchy.
            rhs (torch.Tensor): Right-hand side vector on the finest grid.

        Returns:
            torch.Tensor: Approximate solution after one V-cycle (zero
                initial guess).
        """
        return self._vcycle(hierarchy, rhs, level_idx=0)

    def _vcycle(
        self,
        hierarchy: MultigridHierarchy,
        rhs: torch.Tensor,
        level_idx: int,
    ) -> torch.Tensor:
        level = hierarchy.levels[level_idx]
        matrix = level.matrix

        # Coarsest level: direct dense solve.
        if level.transfer is None:
            return torch.linalg.solve(matrix, rhs)

        # Pre-smooth.
        x = self._smoother.smooth(matrix, rhs, torch.zeros_like(rhs), self._n_pre)

        # Restrict residual to coarse grid.
        r_fine = rhs - matrix @ x
        r_coarse = level.transfer.restrict(r_fine)

        # Recursive coarse-grid correction.
        e_coarse = self._vcycle(hierarchy, r_coarse, level_idx + 1)

        # Prolongate and correct.
        x = x + level.transfer.prolongate(e_coarse)

        # Post-smooth.
        return self._smoother.smooth(matrix, rhs, x, self._n_post)


class WCycle:
    """W-cycle multigrid preconditioner (gamma = 2).

    Applies two sequential coarse-grid corrections per level. After the first
    correction the fine-grid residual is recomputed and restricted again before
    the second coarse solve. This doubles the coarse-grid work relative to
    V-cycle and yields a more robust convergence factor, especially for
    anisotropic or indefinite-leaning problems.

    Args:
        smoother (MultigridSmoother): Smoother applied before and after the
            coarse-grid corrections.
        n_pre (int): Pre-smoothing steps.
        n_post (int): Post-smoothing steps.

    References:
        - Briggs, W. L., Henson, V. E., & McCormick, S. F. (2000).
          A Multigrid Tutorial (2nd ed.). SIAM. Section 3.3, gamma-cycle
          definition.
        - Trottenberg, U., Oosterlee, C. W., & Schuller, A. (2001).
          Multigrid. Academic Press. Section 2.2.2, Algorithm 2.2 (mu = 2).
        - Stuben, K. (2001). A review of algebraic multigrid.
          J. Comput. Appl. Math., 128(1-2), 281-309.
    """

    def __init__(
        self,
        smoother: MultigridSmoother,
        n_pre: int = 2,
        n_post: int = 2,
    ) -> None:
        """Store the smoother and pre/post-smoothing step counts.

        Args:
            smoother (MultigridSmoother): Smoother applied before and after
                the coarse-grid corrections.
            n_pre (int): Pre-smoothing steps.
            n_post (int): Post-smoothing steps.
        """
        self._smoother = smoother
        self._n_pre = n_pre
        self._n_post = n_post

    def apply(self, hierarchy: MultigridHierarchy, rhs: torch.Tensor) -> torch.Tensor:
        """Compute one W-cycle starting from the finest level.

        Args:
            hierarchy (MultigridHierarchy): Pre-built multigrid hierarchy.
            rhs (torch.Tensor): Right-hand side vector on the finest grid.

        Returns:
            torch.Tensor: Approximate solution after one W-cycle (zero
                initial guess).
        """
        return self._wcycle(hierarchy, rhs, level_idx=0)

    def _wcycle(
        self,
        hierarchy: MultigridHierarchy,
        rhs: torch.Tensor,
        level_idx: int,
    ) -> torch.Tensor:
        level = hierarchy.levels[level_idx]
        matrix = level.matrix

        # Coarsest level: direct dense solve.
        if level.transfer is None:
            return torch.linalg.solve(matrix, rhs)

        # Pre-smooth from zero initial guess.
        x = self._smoother.smooth(matrix, rhs, torch.zeros_like(rhs), self._n_pre)

        # gamma = 2: two coarse-grid corrections, recomputing residual each time.
        for _ in range(2):
            r_fine = rhs - matrix @ x
            r_coarse = level.transfer.restrict(r_fine)
            e_coarse = self._wcycle(hierarchy, r_coarse, level_idx + 1)
            x = x + level.transfer.prolongate(e_coarse)

        # Post-smooth.
        return self._smoother.smooth(matrix, rhs, x, self._n_post)
