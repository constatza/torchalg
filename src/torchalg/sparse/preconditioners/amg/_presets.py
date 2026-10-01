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
this package's own sparse ``GaussSeidelSmoother``/``dense_pseudo_inverse_
solve`` instead of the dense ``GaussSeidelSmoother``/``pseudo_inverse_solve``.

No hierarchy-construction formula lives here, only wiring shared by the
sparse presets that need it.
"""

from __future__ import annotations

from collections.abc import Callable

import torch

from torchalg.multigrid import VCycle
from torchalg.multigrid.protocols import MultigridSmoother

from .coarse_solve import dense_pseudo_inverse_solve
from .smoothers import GaussSeidelSmoother
from .transfer import SparseTransferOperator

GS_SETUP_CYCLE = VCycle(
    GaussSeidelSmoother(), n_pre=1, n_post=1, coarse_solver=dense_pseudo_inverse_solve
)
"""GS V(1,1) cycle used only inside sparse adaptive-SA hierarchy setup.

Mirrors the dense ``_presets.GS_SETUP_CYCLE``'s rationale: PyAMG's adaptive
candidate construction requires the algorithm-specific GS relaxation.
Public solve-time cycles are built separately so changing their smoother
cannot perturb setup. ``dense_pseudo_inverse_solve`` instead of the dense
``pseudo_inverse_solve`` since the coarsest-level matrix here is still
sparse CSR until that one deliberate densification.
"""


def prebuilt_cycle(smoother: MultigridSmoother, n_pre: int = 1, n_post: int = 1) -> VCycle:
    """Build the solve-time cycle for a sparse prebuilt-hierarchy preset, V(1,1) by default.

    Sparse-CSR sibling of the dense ``_presets.prebuilt_cycle`` - same
    default and rationale, ``dense_pseudo_inverse_solve`` coarse solver.

    Args:
        smoother (MultigridSmoother): Solve-time smoothing strategy.
        n_pre (int): Pre-smoothing steps.
        n_post (int): Post-smoothing steps.

    Returns:
        VCycle: Symmetric-step cycle with a densifying pseudo-inverse
            coarse solve.
    """
    return VCycle(smoother, n_pre=n_pre, n_post=n_post, coarse_solver=dense_pseudo_inverse_solve)


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
