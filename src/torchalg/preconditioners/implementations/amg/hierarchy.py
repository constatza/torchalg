"""Multigrid hierarchy data model.

Ported from ``neuralls.domain.solver.preconditioners.implementations.amg.hierarchy``
(see ``docs/plan.md``) with ``NDArray`` translated to ``torch.Tensor``. Stays
a plain frozen dataclass pair - not ``nn.Module`` - since the hierarchy is a
transient, lazily-(re)built cache owned by ``AMGPreconditioner``, not a
buffer registered directly on a module (see ``amg.py`` for the full
device/dtype-propagation reasoning).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

    from .protocols import TransferOperator


@dataclass(frozen=True)
class MultigridLevel:
    """One level in a multigrid hierarchy.

    Attributes:
        matrix (torch.Tensor): System matrix on this grid level (n_k x n_k).
        transfer (TransferOperator | None): Operator that moves vectors to
            the next coarser level. ``None`` on the coarsest level - direct
            solve is used there.
    """

    matrix: torch.Tensor
    transfer: TransferOperator | None


@dataclass(frozen=True)
class MultigridHierarchy:
    """Full multigrid hierarchy built by ``AMGPreconditioner``.

    Attributes:
        levels (tuple[MultigridLevel, ...]): Levels from finest (index 0) to
            coarsest (index -1).
    """

    levels: tuple[MultigridLevel, ...]
