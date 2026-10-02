"""BootCMatch's outer bootstrap loop (TOMS Section 5), shared by both trees.

Format-agnostic (see the package docstring): this module only ever calls
``CoarseningStrategy.build_transfer``/``MultigridCycle.apply`` and plain
``matrix @ x`` products - it never reads ``A.is_sparse_csr`` - so one copy
serves the dense ``BootCMatchCoarsening`` and the sparse-CSR
``BootCMatchCoarsening`` sibling identically, the same promotion precedent
as ``bootstrap_setup.py``.

Unlike ``BootstrapSetup`` (``BootstrapAMGPreconditioner``/CR-coarsening's
outer loop, which refines one hierarchy's test vectors in place and
rebuilds it repeatedly), BootCMatch's bootstrap process builds a sequence of
**independent** hierarchies, each seeded from a different smooth vector,
and the actual preconditioner is their multiplicative composition
``B = B_0 o B_1 o B_2 o ...`` - see ``D'Ambra, Filippone, Vassilevski,
"BootCMatch: a Software Package for Bootstrap AMG Based on Graph Weighted
Matching"`` (ACM TOMS 44(4), 2018), Section 5. ``run`` below implements that
outer loop; ``_build_hierarchy`` implements TOMS Algorithm 2's inner
per-hierarchy coarsening loop (simpler than ``BootstrapSetup``'s
``_build_levels``: no test-vector relaxation/level-vectors bookkeeping, just
one smooth vector carried level-to-level by the coarsening strategy itself).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import torch

from torchalg.utils.energy import energy_dot

from .hierarchy import MultigridHierarchy, MultigridLevel
from .protocols import MultigridCycle, TransferOperator


class _BootCMatchCoarseningStrategy(Protocol):
    """``CoarseningStrategy`` plus ``BootCMatchCoarsening``'s own public extensions.

    Narrower than the generic ``CoarseningStrategy`` protocol (Interface
    Segregation, same rationale as ``bootstrap_setup._BootstrapCoarseningStrategy``):
    this module only ever hands a caller-built ``BootCMatchCoarsening``
    (dense or sparse, both sharing this exact shape) around internally.
    """

    def build_transfer(self, A: torch.Tensor) -> tuple[torch.Tensor, TransferOperator]:
        """Build a coarse grid and the associated transfer operator."""
        ...


_CoarseningFactory = Callable[[torch.Tensor], _BootCMatchCoarseningStrategy]
"""``(w) -> _BootCMatchCoarseningStrategy`` - builds a fresh, finest-level-seeded
``BootCMatchCoarsening`` (dense or sparse) from a smooth vector."""


def _build_hierarchy(
    matrix: torch.Tensor,
    coarsening: _BootCMatchCoarseningStrategy,
    max_levels: int,
    max_coarse: int,
) -> MultigridHierarchy:
    """Coarsen ``matrix`` down to ``max_levels``/``max_coarse`` (TOMS Algorithm 2's outer while).

    Mirrors ``bootstrap_setup.BootstrapSetup._build_levels``'s degenerate-
    coarsening guard: a coarsening pass is discarded, and the loop stops
    with ``current`` standing as the final, coarsest level, whenever the
    produced coarse matrix is empty or no smaller than the current level.

    Args:
        matrix (torch.Tensor): Finest-level matrix.
        coarsening (_BootCMatchCoarseningStrategy): Coarsening strategy,
            already seeded with the finest-level smooth vector.
        max_levels (int): Maximum number of levels, including the finest.
        max_coarse (int): Stop coarsening at this many coarse nodes.

    Returns:
        MultigridHierarchy: Levels from finest to coarsest.
    """
    levels: list[MultigridLevel] = []
    current = matrix
    while len(levels) < max_levels - 1 and current.shape[0] > max_coarse:
        coarse_matrix, transfer = coarsening.build_transfer(current)
        if coarse_matrix.shape[0] == 0 or coarse_matrix.shape[0] >= current.shape[0]:
            break
        levels.append(MultigridLevel(matrix=current, transfer=transfer))
        current = coarse_matrix
    levels.append(MultigridLevel(matrix=current, transfer=None))
    return MultigridHierarchy(levels=tuple(levels))


def _a_norm(matrix: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    """``||vector||_A = sqrt(vector^T A vector)``, reusing the shared energy primitive.

    Args:
        matrix (torch.Tensor): SPD operator A.
        vector (torch.Tensor): Vector, shape ``(n,)``.

    Returns:
        torch.Tensor: 0-d A-norm.
    """
    return torch.sqrt(energy_dot(vector, vector, matrix).clamp_min(0.0))


@dataclass(frozen=True)
class BootCMatchSetup:
    """BootCMatch's outer bootstrap loop (TOMS Section 5).

    Builds a sequence of independent multigrid hierarchies, each seeded
    from a smooth vector obtained by power-iterating the homogeneous system
    through the composite of every hierarchy built so far, stopping once
    the composite's measured asymptotic convergence rate ``rho`` is good
    enough or ``max_hierarchies`` is reached.

    Args:
        coarsening_factory (_CoarseningFactory): Builds a fresh
            ``BootCMatchCoarsening`` (dense or sparse) seeded with a smooth
            vector ``w`` - the caller's own closure captures whatever
            constructor arguments it wants.
        cycle (MultigridCycle): Shared multigrid cycle applied to every
            hierarchy in the composite (one instance, many hierarchies -
            ``MultigridCycle.apply`` takes the hierarchy as a parameter).
        draw (Callable[[int], torch.Tensor]): Uniform ``[0, 1)``
            random-vector source for the seed smooth vector and each
            power-iteration restart.
        max_levels (int): Maximum number of levels per hierarchy.
        max_coarse (int): Stop coarsening at this many coarse nodes.
        k_max (int): Power-iteration sweeps per outer-loop round.
        rho_desired (float): Target asymptotic convergence rate; the loop
            stops adding hierarchies once the composite meets it. Default
            0.8 matches the paper's own composite-AMG experiments (Section
            7.3: "a bootstrap AMG when a convergence ratio p = 0.8 is
            prescribed"), looser than this package's prior 0.7 default -
            which was tighter than anything the paper itself used, and
            therefore tended to burn through ``max_hierarchies`` on harder
            problems instead of stopping early.
        max_hierarchies (int): Hard cap on the number of hierarchies in the
            composite, regardless of ``rho_desired``.

    References:
        - D'Ambra, Filippone, Vassilevski (2018). BootCMatch: a Software
          Package for Bootstrap AMG Based on Graph Weighted Matching.
          ACM TOMS 44(4). Sections 5, 7.3.
    """

    coarsening_factory: _CoarseningFactory
    cycle: MultigridCycle
    draw: Callable[[int], torch.Tensor]
    max_levels: int = 10
    max_coarse: int = 10
    k_max: int = 10
    rho_desired: float = 0.8
    max_hierarchies: int = 5

    def run(
        self, matrix: torch.Tensor, seed_vector: torch.Tensor | None = None
    ) -> tuple[MultigridHierarchy, ...]:
        """Run the BootCMatch bootstrap outer loop.

        Args:
            matrix (torch.Tensor): SPD system matrix, shape ``(n, n)``
                (dense or sparse CSR).
            seed_vector (torch.Tensor | None): Initial smooth vector
                ``w_0``; ``None`` draws one via ``self.draw``.

        Returns:
            tuple[MultigridHierarchy, ...]: The composite's hierarchies,
            ``B_0, B_1, ..., B_{r-1}``, in application order.
        """
        n = matrix.shape[0]
        w = (
            seed_vector
            if seed_vector is not None
            else self.draw(n).to(dtype=matrix.dtype, device=matrix.device)
        )
        hierarchies = [
            _build_hierarchy(matrix, self.coarsening_factory(w), self.max_levels, self.max_coarse)
        ]

        eps = torch.finfo(matrix.dtype).eps
        while True:
            x = self.draw(n).to(dtype=matrix.dtype, device=matrix.device)
            norm_before = _a_norm(matrix, x)
            for _ in range(self.k_max):
                for hierarchy in hierarchies:
                    x = x - self.cycle.apply(hierarchy, matrix @ x)

            norm_now = _a_norm(matrix, x)
            rho = (norm_now / norm_before.clamp_min(eps)) ** (1.0 / self.k_max)

            if rho <= self.rho_desired or len(hierarchies) >= self.max_hierarchies:
                break

            w = x
            hierarchies.append(
                _build_hierarchy(
                    matrix, self.coarsening_factory(w), self.max_levels, self.max_coarse
                )
            )

        return tuple(hierarchies)
