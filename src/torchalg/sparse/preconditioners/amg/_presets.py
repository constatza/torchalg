"""Shared building blocks for sparse prebuilt-hierarchy preconditioner presets.

Sparse-CSR sibling of
``preconditioners.implementations.amg._presets`` (dense; kept unmodified for
comparison - see ``docs/plan.md``'s "Correction: dense and sparse must be
separate implementations, not an internal branch"). ``adaptive.py``'s
sparse alpha-SA port follows the same preset shape as the dense one: run a
setup algorithm at construction, then wrap the resulting fixed hierarchy in
a ``torchalg.multigrid.AMGPreconditioner`` whose ``_make_hierarchy`` is
overridden, so the engine never asks a ``CoarseningStrategy`` for a level.
That shape needs the same shared seeded randomness, setup-cycle wiring,
solve-cycle wiring, and placeholder ``CoarseningStrategy`` - built here from
this package's own sparse ``GaussSeidelStage``/``dense_pseudo_inverse_
coarse_solver`` instead of the dense ``GaussSeidelSmoother``/
``pseudo_inverse_coarse_solver``.

No hierarchy-construction formula lives here, only wiring shared by the
sparse presets that need it.
"""

from __future__ import annotations

from collections.abc import Callable

import torch

from torchalg.multigrid import VCycle
from torchalg.multigrid.protocols import MultigridSmoother

from .coarse_solve import dense_pseudo_inverse_coarse_solver
from .smoothers import GaussSeidelStage
from .transfer import SparseTransferOperator


def gs_setup_cycle(stage: GaussSeidelStage) -> VCycle:
    """GS V(1,1) cycle used only inside sparse hierarchy setup, bound to one stage.

    PyAMG's adaptive candidate construction requires the algorithm-specific
    GS relaxation. The cycle's smoother is the setup stage's own
    ``GaussSeidelStage``, so preparing the many sub-hierarchies a stage cycles
    through builds each level matrix's level schedules once; a cycle built from
    a fresh stage per setup run therefore never shares prepared state across
    runs. Public solve-time cycles are built separately (``prebuilt_cycle``) so
    changing their smoother cannot perturb setup. ``dense_pseudo_inverse_coarse_solver``
    because the coarsest-level matrix here is still sparse CSR until that one
    deliberate densification.

    Args:
        stage (GaussSeidelStage): The stage's Gauss-Seidel preparation cache.

    Returns:
        VCycle: GS V(1,1) with a densifying pseudo-inverse coarse solve.
    """
    return VCycle(
        stage,
        n_pre=1,
        n_post=1,
        coarse_solver_factory=dense_pseudo_inverse_coarse_solver,
    )


def prebuilt_cycle(smoother: MultigridSmoother, n_pre: int = 1, n_post: int = 1) -> VCycle:
    """Build the solve-time cycle for a sparse prebuilt-hierarchy preset, V(1,1) by default.

    Sparse-CSR sibling of the dense ``_presets.prebuilt_cycle`` - same
    default and rationale, ``dense_pseudo_inverse_coarse_solver`` coarse solver.

    Args:
        smoother (MultigridSmoother): Solve-time smoothing strategy.
        n_pre (int): Pre-smoothing steps.
        n_post (int): Post-smoothing steps.

    Returns:
        VCycle: Symmetric-step cycle with a densifying pseudo-inverse
            coarse solve.
    """
    return VCycle(
        smoother,
        n_pre=n_pre,
        n_post=n_post,
        coarse_solver_factory=dense_pseudo_inverse_coarse_solver,
    )


def seeded_draw(seed: int) -> Callable[[int], torch.Tensor]:
    """Stateful uniform ``[0, 1)`` source seeded for reproducibility.

    Trivial duplicate of the dense ``_presets.seeded_draw`` - same precedent
    as this package's other small, entangled boilerplate duplication (see
    ``bootstrap.py``'s module docstring) rather than a cross-tree import for
    one function this small.

    Args:
        seed (int): Generator seed.

    Returns:
        Callable[[int], torch.Tensor]: ``n -> float64`` tensor of length ``n``.
    """
    generator = torch.Generator().manual_seed(seed)
    return lambda n: torch.rand(n, generator=generator, dtype=torch.float64)


class PrebuiltCoarsening:
    """Placeholder ``CoarseningStrategy``: the levels are prebuilt, never built by the engine.

    Sparse-CSR sibling of the dense ``_presets.PrebuiltCoarsening`` - same
    refusal behavior, own copy per this tree's "no dense import" rule.

    Args:
        algorithm (str): Name of the setup algorithm that owns the prebuilt
            levels; appears in the error message.
    """

    def __init__(self, algorithm: str) -> None:
        """Store the algorithm name used in the refusal message.

        Args:
            algorithm (str): Name of the setup algorithm that owns the
                prebuilt levels.
        """
        self._algorithm = algorithm

    def build_transfer(self, A: torch.Tensor) -> tuple[torch.Tensor, SparseTransferOperator]:
        """Always raises: the hierarchy is prebuilt by the preset's setup algorithm.

        Args:
            A (torch.Tensor): Unused.

        Raises:
            RuntimeError: Always.
        """
        raise RuntimeError(
            f"{self._algorithm} levels are prebuilt; the engine must not rebuild them"
        )
