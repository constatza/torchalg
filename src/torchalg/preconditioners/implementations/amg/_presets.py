"""Shared building blocks for the prebuilt-hierarchy preconditioner presets.

``adaptive.py`` (alpha-SA) and ``bootstrap.py`` (BAMG) follow the same
preset shape: run a setup algorithm at construction, then wrap the resulting
fixed hierarchy in an ``AMGPreconditioner`` whose ``_make_hierarchy`` is
overridden, so the engine never asks a ``CoarseningStrategy`` for a level.
That shape needs shared seeded randomness, setup-cycle wiring, solve-cycle
wiring, and a placeholder ``CoarseningStrategy`` that refuses to build
anything. They live here once instead of being copied per preset.

No hierarchy-construction formula lives here, only wiring shared by both
presets.
"""

from __future__ import annotations

from collections.abc import Callable

import torch

from .cycle import VCycle, pseudo_inverse_solve
from .protocols import MultigridSmoother
from .smoothers import GaussSeidelSmoother
from .transfer import DenseTransferOperator

GS_SETUP_CYCLE = VCycle(
    GaussSeidelSmoother(), n_pre=1, n_post=1, coarse_solver=pseudo_inverse_solve
)
"""GS V(1,1) cycle used only inside adaptive/bootstrap hierarchy setup.

PyAMG's adaptive candidate construction and [BAMG11]'s test-vector
improvement require the algorithm-specific GS relaxation. Public solve-time
cycles are built separately so changing their smoother cannot perturb setup.
"""


def prebuilt_cycle(smoother: MultigridSmoother) -> VCycle:
    """Build the solve-time V(1,1) cycle for a prebuilt hierarchy preset.

    Args:
        smoother (MultigridSmoother): Solve-time smoothing strategy.

    Returns:
        VCycle: Symmetric-step cycle with a pseudo-inverse coarse solve.
    """
    return VCycle(smoother, n_pre=1, n_post=1, coarse_solver=pseudo_inverse_solve)


def seeded_draw(seed: int) -> Callable[[int], torch.Tensor]:
    """Stateful uniform ``[0, 1)`` source seeded for reproducibility.

    Args:
        seed (int): Generator seed.

    Returns:
        Callable[[int], torch.Tensor]: ``n -> float64`` tensor of length ``n``.
    """
    generator = torch.Generator().manual_seed(seed)
    return lambda n: torch.rand(n, generator=generator, dtype=torch.float64)


class PrebuiltCoarsening:
    """Placeholder ``CoarseningStrategy``: the levels are prebuilt, never built by the engine.

    Presets that override ``_make_hierarchy`` never let the engine ask a
    strategy for a level, so this one always raises rather than silently
    returning something the preset did not compute.

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

    def build_transfer(self, A: torch.Tensor) -> tuple[torch.Tensor, DenseTransferOperator]:
        """Always raises: the hierarchy is prebuilt by the preset's setup algorithm.

        Args:
            A (torch.Tensor): Unused.

        Raises:
            RuntimeError: Always.
        """
        raise RuntimeError(
            f"{self._algorithm} levels are prebuilt; the engine must not rebuild them"
        )
