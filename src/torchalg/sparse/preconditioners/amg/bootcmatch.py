"""BootCMatch preconditioner, sparse-CSR sibling.

Sparse-CSR counterpart of
``preconditioners.implementations.amg.bootcmatch.BootCMatchPreconditioner``
(dense; kept unmodified for comparison) - see that module's docstring for
the full design rationale (composite-of-hierarchies construction, why this
does not subclass ``torchalg.multigrid.AMGPreconditioner``, and the
device-move handling). Every change from the dense class follows directly
from storage format, not a different algorithm: this tree's own
``BootCMatchCoarsening`` (sparse CSR, ``bootcmatch_coarsening.py``) and
``_presets.prebuilt_cycle``/``smoothers.resolve_jacobi_default`` (sparse
Jacobi default, ``dense_pseudo_inverse_solve`` coarse solver) in place of
the dense tree's equivalents.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn

from torchalg.multigrid import MultigridHierarchy, MultigridLevel, VCycle
from torchalg.multigrid.bootcmatch_setup import BootCMatchSetup
from torchalg.preconditioners.base import Preconditioner, PreconditionerContext

from ._presets import prebuilt_cycle, seeded_draw
from .bootcmatch_coarsening import BootCMatchCoarsening
from .smoothers import resolve_jacobi_default

if TYPE_CHECKING:
    from collections.abc import Callable

    from torchalg.multigrid.protocols import MultigridSmoother

__all__ = ["BootCMatchPreconditioner"]


def _move_hierarchy(hierarchy: MultigridHierarchy, like: torch.Tensor) -> MultigridHierarchy:
    """Move every level matrix in ``hierarchy`` to ``like``'s device/dtype.

    Sparse-CSR sibling of the dense ``bootcmatch._move_hierarchy`` - same
    rationale, ``.to()`` works identically for sparse CSR tensors.

    Args:
        hierarchy (MultigridHierarchy): Hierarchy to move.
        like (torch.Tensor): Tensor whose device/dtype to match.

    Returns:
        MultigridHierarchy: A new hierarchy sharing ``like``'s device/dtype.
    """
    return MultigridHierarchy(
        tuple(
            MultigridLevel(matrix=level.matrix.to(like), transfer=level.transfer)
            for level in hierarchy.levels
        )
    )


class BootCMatchPreconditioner(Preconditioner, nn.Module):
    """BootCMatch preconditioner (TOMS Sections 4-5), sparse-CSR sibling.

    Args:
        matrix (torch.Tensor): SPD sparse CSR system matrix A (n x n).
        seed (int): Seed of the default random-vector source.
        draw (Callable[[int], torch.Tensor] | None): Explicit uniform
            ``[0, 1)`` source overriding ``seed``.
        max_levels (int): Maximum number of levels per hierarchy.
        max_coarse (int): Stop coarsening at this many coarse nodes.
        k_max (int): Power-iteration sweeps per outer-loop round.
        rho_desired (float): Target asymptotic convergence rate.
        max_hierarchies (int): Hard cap on the number of hierarchies in the
            composite.
        smoother (MultigridSmoother | None): Explicit solve-time smoother;
            ``None`` selects weighted Jacobi.
        smoother_omega (float | None): Damping for the default
            weighted-Jacobi solve smoother.
        n_pre (int): Solve-time pre-smoothing sweeps.
        n_post (int): Solve-time post-smoothing sweeps.

    References:
        - D'Ambra, Filippone, Vassilevski (2018). BootCMatch: a Software
          Package for Bootstrap AMG Based on Graph Weighted Matching.
          ACM TOMS 44(4). Sections 4-5.
    """

    def __init__(
        self,
        matrix: torch.Tensor,
        seed: int = 0,
        draw: Callable[[int], torch.Tensor] | None = None,
        max_levels: int = 10,
        max_coarse: int = 10,
        k_max: int = 10,
        rho_desired: float = 0.7,
        max_hierarchies: int = 5,
        smoother: MultigridSmoother | None = None,
        smoother_omega: float | None = None,
        n_pre: int = 1,
        n_post: int = 1,
    ) -> None:
        """Run the BootCMatch bootstrap setup and store its composite hierarchies.

        Args:
            matrix (torch.Tensor): SPD sparse CSR system matrix A (n x n).
            seed (int): Seed of the default random-vector source.
            draw (Callable[[int], torch.Tensor] | None): Explicit uniform
                ``[0, 1)`` source overriding ``seed``.
            max_levels (int): Maximum number of levels per hierarchy.
            max_coarse (int): Stop coarsening at this many coarse nodes.
            k_max (int): Power-iteration sweeps per outer-loop round.
            rho_desired (float): Target asymptotic convergence rate.
            max_hierarchies (int): Hard cap on the number of hierarchies in
                the composite.
            smoother (MultigridSmoother | None): Explicit solve-time
                smoother, or ``None`` for weighted Jacobi.
            smoother_omega (float | None): Damping for the default
                weighted-Jacobi solve smoother.
            n_pre (int): Solve-time pre-smoothing sweeps.
            n_post (int): Solve-time post-smoothing sweeps.
        """
        nn.Module.__init__(self)
        cycle: VCycle = prebuilt_cycle(
            resolve_jacobi_default(smoother, smoother_omega), n_pre=n_pre, n_post=n_post
        )
        setup = BootCMatchSetup(
            coarsening_factory=BootCMatchCoarsening,
            cycle=cycle,
            draw=draw if draw is not None else seeded_draw(seed),
            max_levels=max_levels,
            max_coarse=max_coarse,
            k_max=k_max,
            rho_desired=rho_desired,
            max_hierarchies=max_hierarchies,
        )
        self._hierarchies: tuple[MultigridHierarchy, ...] = setup.run(matrix)
        self._cycle = cycle
        self._matrix: torch.Tensor
        self.register_buffer("_matrix", matrix)

    @property
    def hierarchies(self) -> tuple[MultigridHierarchy, ...]:
        """The composite's hierarchies, in application order.

        Returns:
            tuple[MultigridHierarchy, ...]: ``B_0, B_1, ..., B_{r-1}``.
        """
        return self._hierarchies

    @property
    def requires_flexible_cg(self) -> bool:
        """Whether Flexible CG is required.

        Returns:
            bool: ``False`` - a fixed linear composite of fixed linear
                cycles, same as ``BootstrapAMGPreconditioner``.
        """
        return False

    def apply(
        self,
        residual: torch.Tensor,
        context: PreconditionerContext | None = None,
    ) -> torch.Tensor:
        """Apply the multiplicative composite correction.

        Args:
            residual (torch.Tensor): Current residual vector r_k.
            context (PreconditionerContext | None): Ignored (the composite
                is stateless w.r.t. solver iteration).

        Returns:
            torch.Tensor: Approximate solution to ``A z = r`` from sweeping
            every hierarchy's cycle once, multiplicative-Schwarz style.
        """
        x = torch.zeros_like(residual)
        for hierarchy in self._hierarchies:
            moved = _move_hierarchy(hierarchy, self._matrix)
            x = x + self._cycle.apply(moved, residual - self._matrix @ x)
        return x

    def __str__(self) -> str:
        """Human-readable structural summary.

        Returns:
            str: e.g. ``"BootCMatch(n_hierarchies=2)"``.
        """
        return f"BootCMatch(n_hierarchies={len(self._hierarchies)})"
