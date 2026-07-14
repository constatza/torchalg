"""POD-based coarsening strategy (POD-2G, Nikolopoulos et al. 2022)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from torch import nn

from ..amg.transfer import DenseTransferOperator
from .basis import compute_pod_basis

if TYPE_CHECKING:
    import torch


class PODCoarseningStrategy(nn.Module):
    """Coarsening via a POD basis fit to a snapshot ensemble (POD-2G).

    Subclasses ``nn.Module`` (buffer only, no learnable parameters - same
    rationale as ``JacobiPreconditioner``/``AMGPreconditioner``, see
    ``docs/plan.md``) purely so the fitted basis follows ``.to(device/dtype)``.
    Unlike ``AggregationCoarsening`` (plain class - it holds no tensor state,
    only scalar hyperparameters), this class owns a real tensor computed once
    at construction, which is exactly the case ``register_buffer`` is for.
    Assigning an ``nn.Module``-typed ``coarsening`` argument to
    ``AMGPreconditioner._coarsening`` auto-registers it as a submodule, so no
    changes to ``AMGPreconditioner`` are needed for this to participate in its
    owning preconditioner's ``.to()`` calls.

    Builds a single reduced level using a POD basis Phi_r fit to an ensemble
    of high-fidelity solution snapshots, instead of algebraic aggregation.
    The Galerkin coarse matrix K_r = Phi_r^T K Phi_r (eq. 28) has shape
    (rank x rank), with rank << n_dofs.

    Implements the same ``CoarseningStrategy`` protocol as
    ``amg.AggregationCoarsening`` (see
    ``preconditioners.implementations.amg.protocols``), so it plugs directly
    into the generic ``AMGPreconditioner`` engine - this class only decides
    *how* one coarse level is built, never how the multigrid cycle around it
    runs. The basis itself is built via the thin-SVD "snapshot method"
    rather than eigendecomposing a Gram matrix - see ``compute_pod_basis``
    for the full numerical-stability rationale (avoiding the
    condition-number squaring that a Gram-matrix formation would introduce).

    The snapshot ensemble is a plain tensor - where it comes from (FEM
    archive files today, a neural snapshot generator later) is entirely a
    composition-layer concern; this class does not care.

    Args:
        snapshots (torch.Tensor): Solution snapshot ensemble, shape
            (n_samples, n_dofs).
        rank (int | float): Fixed mode count (int) or minimum cumulative
            captured energy (float in (0, 1]) - see ``compute_pod_basis``.

    References:
        - Nikolopoulos, S., Kalogeris, I., Stavroulakis, G., & Papadopoulos,
          V. (2022). AI-enhanced iterative solvers for accelerating the
          solution of large-scale parametrized systems. arXiv:2207.02543.
    """

    def __init__(self, snapshots: torch.Tensor, rank: int | float) -> None:
        """Fit the POD basis from the snapshot ensemble.

        Args:
            snapshots (torch.Tensor): Solution snapshot ensemble, shape
                (n_samples, n_dofs).
            rank (int | float): Fixed mode count (int) or minimum cumulative
                captured energy (float in (0, 1]).
        """
        nn.Module.__init__(self)
        self._basis: torch.Tensor
        self.register_buffer("_basis", compute_pod_basis(snapshots, rank))

    def build_transfer(self, A: torch.Tensor) -> tuple[torch.Tensor, DenseTransferOperator]:
        """Build the POD-reduced coarse level from the fine-grid matrix A.

        Args:
            A (torch.Tensor): Fine-grid matrix (n_dofs x n_dofs). Named to
                match the ``CoarseningStrategy`` protocol's parameter name
                exactly (structural typing checks parameter names for
                positional-or-keyword parameters).

        Returns:
            tuple[torch.Tensor, DenseTransferOperator]: ``(A_coarse,
                transfer)`` - A_coarse is the Galerkin coarse matrix
                (rank x rank); transfer applies Phi_r/Phi_r^T. Reuses
                ``amg.DenseTransferOperator`` directly (see
                ``preconditioners.implementations.pod``'s module docstring
                for why this is a genuine reuse, not a coincidental name
                collision): its prolongate/restrict (``P @ x`` /
                ``P.T @ x``) is exactly the Phi_r/Phi_r^T pair the POD-2G
                paper defines, with Phi_r simply passed in as ``P``.
        """
        A_coarse = self._basis.T @ A @ self._basis
        return A_coarse, DenseTransferOperator(self._basis)
