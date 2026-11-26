"""Multigrid cycle implementations.

Ported from ``neuralls.domain.solver.preconditioners.implementations.amg.cycle``
with ``NDArray`` translated to ``torch.Tensor`` and
``numpy.linalg.solve`` replaced by a prepared (factor-once) coarsest direct solve. Both cycles are format-agnostic: every op is delegated to
the injected ``MultigridSmoother``/``TransferOperator``/coarse-solver factory,
so a sparse ``CoarseningStrategy`` keeps every non-coarsest level sparse CSR
throughout the cycle - only the coarsest-level solve is format-committing
(``lu_coarse_solver`` by default; see
``torchalg.sparse.preconditioners.amg.coarse_solve`` for the sparse-aware
adapters). Smoothers and the coarse solve are prepared once per hierarchy by
``cycle.prepare``; ``apply`` does no matrix-dependent one-off work.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from .coarse_solve import CoarseSolverFactory, HierarchyNotPreparedError, lu_coarse_solver
from .hierarchy import MultigridHierarchy, PreparedHierarchy, prepare_hierarchy

if TYPE_CHECKING:
    from torchalg.preconditioners.base import Placement

    from .protocols import MultigridSmoother


_NOT_PREPARED = "Call cycle.prepare(hierarchy) after building a hierarchy and apply the result."


def _require_prepared(hierarchy: object) -> PreparedHierarchy:
    """Boundary check: an unprepared hierarchy can still arrive from a caller.

    Everything below ``apply`` works on ``PreparedHierarchy`` by type, so this is the
    one place that turns a forgotten ``prepare`` into a clear error.
    """
    if not isinstance(hierarchy, PreparedHierarchy):
        raise HierarchyNotPreparedError(_NOT_PREPARED)
    return hierarchy


class VCycle:
    """Standard V-cycle multigrid preconditioner (gamma = 1).

    Descends the hierarchy pre-smoothing, computes a coarse-grid correction,
    then ascends post-smoothing. One recursive call per level.

    Cost:
        Notation: ``N`` levels, ``l = 0 .. N - 2`` the non-coarsest ones with
        ``n_l`` rows, ``z_l`` stored entries (``z_l = n_l^2`` for a dense level) and
        ``p_l`` stored entries of the prolongator (``n_l n_{l+1}`` dense), ``m_l``
        the largest row count, ``c`` the coarsest dimension, ``a = n_pre``,
        ``b = n_post``, ``k`` right-hand-side columns, ``V = 8`` bytes per
        ``float64``. The ``sm`` terms are the smoother's own Cost.
        Setup (``prepare``, once per hierarchy): one ``smoother.prepare`` per
        non-coarsest level and one ``coarse_solver_factory`` call,
        ``W = sum_l W_sm_prep,l + W_c_prep`` (``2c^3 / 3`` for the default LU),
        ``D = max(max_l D_sm_prep,l, D_c_prep)`` (the levels are independent,
        though they are prepared one after another; ``D_c_prep = c`` for LU),
        ``L = sum_l L_sm_prep,l + 1`` and ``S = sum_l S_sm_prep,l + 1`` (the LU
        singularity read). A Jacobi smoother adds an Arnoldi estimate per level
        unless the coarsening already paid it on the same matrix.
        Apply (one cycle from the zero guess): per level ``a`` sweeps from zero, one
        residual, one restriction, one prolongation, one correction and ``b``
        sweeps, then one coarse solve:
        ``W = sum_l [W_sm,l(a, b) + W_res,l + 4 p_l k + n_l k] + 2c^2 k``. With
        weighted Jacobi (``PreparedJacobi``) the first sweep from zero is one
        scaling, so ``W = sum_l [2(a + b) z_l + 3(a + b) n_l + 4 p_l] k + 2c^2 k``,
        ``Q`` the SpMV bytes ``(a + b) z_l(V + w)`` and transfer bytes
        ``2 p_l(V + w)`` per level plus ``O((a + b) V n_l k)`` vector passes,
        ``D = sum_l (a + b + 3)(ceil(log2 m_l) + 4) + D_c`` (the cycle is one
        dependent chain through every level; ``D_c = 2c`` for the LU solve,
        ``ceil(log2 c) + 1`` for the pseudo-inverse), ``L = sum_l [4(a + b) + 2] + 1``
        and ``S = 0``. Gauss-Seidel sweeps replace ``ceil(log2 m_l)`` per sweep by
        the triangular depth ``lv_f,l + lv_b,l``.
        Roofline: about 0.15 flop per byte on sparse levels (memory-bound); a dense
        level is a GEMV, also memory-bound at ``k = 1``.
        Work-depth: ``W`` is dominated by the finest level (operator complexity
        ``sum_l z_l / z_0`` near 1 to 2); ``D`` grows with ``N`` and the coarse
        solve, so a cycle has far less parallel slack than a single SpMV.

    Args:
        smoother (MultigridSmoother): Smoother applied before and after the
            coarse-grid correction.
        n_pre (int): Pre-smoothing steps.
        n_post (int): Post-smoothing steps.
        coarse_solver_factory (CoarseSolverFactory): Builds the coarsest-level
            solve once per hierarchy (``prepare``); ``lu_coarse_solver`` by default.

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
        coarse_solver_factory: CoarseSolverFactory = lu_coarse_solver,
    ) -> None:
        """Store the smoother, pre/post-smoothing step counts and coarse solver.

        Args:
            smoother (MultigridSmoother): Smoother applied before and after
                the coarse-grid correction.
            n_pre (int): Pre-smoothing steps.
            n_post (int): Post-smoothing steps.
            coarse_solver_factory (CoarseSolverFactory): Builds the coarsest-level
                solve once per hierarchy.
        """
        self._smoother = smoother
        self.placement: Placement = smoother.placement
        self._n_pre = n_pre
        self._n_post = n_post
        self._coarse_solver_factory = coarse_solver_factory

    def prepare(self, hierarchy: MultigridHierarchy | PreparedHierarchy) -> PreparedHierarchy:
        """Prepare every level once and return the prepared hierarchy.

        Binds the smoother to each non-coarsest level (diagonal inverse, spectral-radius
        damping, triangular factors, ...) and factorizes the coarsest level. Idempotent.
        The prepared state lives on the returned hierarchy, not on this cycle, so one cycle
        instance can serve many matrices.

        Args:
            hierarchy (MultigridHierarchy | PreparedHierarchy): Hierarchy from
                ``build_hierarchy``; an already prepared one is returned as is.

        Returns:
            PreparedHierarchy: Hierarchy that ``apply`` accepts.
        """
        return prepare_hierarchy(hierarchy, self._smoother, self._coarse_solver_factory)

    def apply(self, hierarchy: PreparedHierarchy, rhs: torch.Tensor) -> torch.Tensor:
        """Compute one V-cycle starting from the finest level.

        Args:
            hierarchy (PreparedHierarchy): Hierarchy returned by ``prepare``.
            rhs (torch.Tensor): Right-hand side vector on the finest grid.

        Returns:
            torch.Tensor: Approximate solution after one V-cycle (zero
                initial guess).
        """
        return self._vcycle(_require_prepared(hierarchy), rhs, level_idx=0)

    def _vcycle(
        self,
        hierarchy: PreparedHierarchy,
        rhs: torch.Tensor,
        level_idx: int,
    ) -> torch.Tensor:
        if level_idx == len(hierarchy.fine):
            return hierarchy.coarsest.coarse_solve(rhs)
        level = hierarchy.fine[level_idx]
        transfer, smoother = level.transfer, level.smoother

        # Pre-smooth from the zero initial guess (skips the A @ 0 matvec).
        x = smoother.smooth_from_zero(rhs, self._n_pre)

        # Restrict residual to coarse grid.
        r_fine = smoother.residual(rhs, x)
        r_coarse = transfer.restrict(r_fine)

        # Recursive coarse-grid correction.
        e_coarse = self._vcycle(hierarchy, r_coarse, level_idx + 1)

        # Prolongate and correct.
        x = x + transfer.prolongate(e_coarse)

        # Post-smooth.
        return smoother.smooth(rhs, x, self._n_post)


class WCycle:
    """W-cycle multigrid preconditioner (gamma = 2).

    Applies two sequential coarse-grid corrections per level. After the first
    correction the fine-grid residual is recomputed and restricted again before
    the second coarse solve. This doubles the coarse-grid work relative to
    V-cycle and yields a more robust convergence factor, especially for
    anisotropic or indefinite-leaning problems.

    Cost:
        Notation as in ``VCycle``. Level ``l`` is visited ``2^l`` times and the
        coarsest level ``2^(N-1)`` times per cycle.
        Setup (``prepare``) is the same as ``VCycle``'s: one ``smoother.prepare``
        per non-coarsest level and one coarse factorization,
        ``W = sum_l W_sm_prep,l + W_c_prep``,
        ``D = max(max_l D_sm_prep,l, D_c_prep)``, ``L = sum_l L_sm_prep,l + 1``,
        ``S = sum_l S_sm_prep,l + 1``.
        Apply (one cycle from the zero guess): each visit runs ``a`` sweeps from
        zero, two (residual, restriction, recursion, prolongation, correction)
        rounds and ``b`` sweeps. With weighted Jacobi a visit costs
        ``W_l = [(2(a + b) + 2) z_l + (3(a + b) + 2) n_l + 8 p_l] k`` and
        ``L_l = 4(a + b) + 7``, so
        ``W = sum_l 2^l W_l + 2^(N-1) 2c^2 k``,
        ``D = sum_l 2^l (a + b + 5)(ceil(log2 m_l) + 4) + 2^(N-1) D_c`` (the two
        corrections are sequential: the second restricts the residual after the
        first), ``L = sum_l 2^l L_l + 2^(N-1)`` and ``S = 0``.
        Work-depth: with a coarsening ratio ``n_{l+1} / n_l`` below ``1/2`` the work
        stays dominated by the finest level, but the depth and the launch count
        double per level, so on a launch-bound device a W-cycle costs about
        ``2^(N-1)`` V-cycles of latency.

    Args:
        smoother (MultigridSmoother): Smoother applied before and after the
            coarse-grid corrections.
        n_pre (int): Pre-smoothing steps.
        n_post (int): Post-smoothing steps.
        coarse_solver_factory (CoarseSolverFactory): Builds the coarsest-level
            solve once per hierarchy (``prepare``); ``lu_coarse_solver`` by
            default (matches ``VCycle``'s equivalent parameter).

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
        coarse_solver_factory: CoarseSolverFactory = lu_coarse_solver,
    ) -> None:
        """Store the smoother, pre/post-smoothing step counts and coarse solver.

        Args:
            smoother (MultigridSmoother): Smoother applied before and after
                the coarse-grid corrections.
            n_pre (int): Pre-smoothing steps.
            n_post (int): Post-smoothing steps.
            coarse_solver_factory (CoarseSolverFactory): Builds the coarsest-level
                solve once per hierarchy.
        """
        self._smoother = smoother
        self.placement: Placement = smoother.placement
        self._n_pre = n_pre
        self._n_post = n_post
        self._coarse_solver_factory = coarse_solver_factory

    def prepare(self, hierarchy: MultigridHierarchy | PreparedHierarchy) -> PreparedHierarchy:
        """Prepare every level once and return the prepared hierarchy.

        Binds the smoother to each non-coarsest level (diagonal inverse, spectral-radius
        damping, triangular factors, ...) and factorizes the coarsest level. Idempotent.
        The prepared state lives on the returned hierarchy, not on this cycle, so one cycle
        instance can serve many matrices.

        Args:
            hierarchy (MultigridHierarchy | PreparedHierarchy): Hierarchy from
                ``build_hierarchy``; an already prepared one is returned as is.

        Returns:
            PreparedHierarchy: Hierarchy that ``apply`` accepts.
        """
        return prepare_hierarchy(hierarchy, self._smoother, self._coarse_solver_factory)

    def apply(self, hierarchy: PreparedHierarchy, rhs: torch.Tensor) -> torch.Tensor:
        """Compute one W-cycle starting from the finest level.

        Args:
            hierarchy (PreparedHierarchy): Hierarchy returned by ``prepare``.
            rhs (torch.Tensor): Right-hand side vector on the finest grid.

        Returns:
            torch.Tensor: Approximate solution after one W-cycle (zero
                initial guess).
        """
        return self._wcycle(_require_prepared(hierarchy), rhs, level_idx=0)

    def _wcycle(
        self,
        hierarchy: PreparedHierarchy,
        rhs: torch.Tensor,
        level_idx: int,
    ) -> torch.Tensor:
        if level_idx == len(hierarchy.fine):
            return hierarchy.coarsest.coarse_solve(rhs)
        level = hierarchy.fine[level_idx]
        transfer, smoother = level.transfer, level.smoother

        # Pre-smooth from the zero initial guess (skips the A @ 0 matvec).
        x = smoother.smooth_from_zero(rhs, self._n_pre)

        # gamma = 2: two coarse-grid corrections, recomputing residual each time.
        for _ in range(2):
            r_fine = smoother.residual(rhs, x)
            r_coarse = transfer.restrict(r_fine)
            e_coarse = self._wcycle(hierarchy, r_coarse, level_idx + 1)
            x = x + transfer.prolongate(e_coarse)

        # Post-smooth.
        return smoother.smooth(rhs, x, self._n_post)
