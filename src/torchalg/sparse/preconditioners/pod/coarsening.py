"""Sparse-CSR POD coarsening, sparse-CSR sibling.

Sparse-CSR counterpart of
``preconditioners.implementations.pod.coarsening.PODCoarseningStrategy``
(dense; kept unmodified for comparison - see ``docs/plan.md``'s "Correction:
dense and sparse must be separate implementations, not an internal branch").
Same algorithm, same basis construction (``torchalg.utils.pod_basis
.compute_pod_basis``, a thin-SVD fit over a dense snapshot ensemble - the POD
basis Phi_r is fundamentally dense regardless of whether the fine-grid
matrix is sparse or dense), disambiguated by package: same class name
(``PODCoarseningStrategy``), same public shape, only ``build_transfer``
differs in how it forms the Galerkin coarse operator.

Mirrors the sparse ``AggregationCoarsening`` sibling
(``torchalg.sparse.preconditioners.amg.coarsening``) in spirit: the fine-grid
matrix ``A`` is accepted as sparse CSR and never densified. Unlike that
sibling, the transfer operator returned here is still
``torchalg.utils.dense_transfer.DenseTransferOperator`` (not a sparse
transfer operator) - prolongate/restrict apply the dense basis Phi_r itself,
which is unaffected by A's format.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from torch import nn

from torchalg.sparse.kernels.galerkin import form_sparse_dense
from torchalg.utils.dense_transfer import DenseTransferOperator
from torchalg.utils.pod_basis import compute_pod_basis

if TYPE_CHECKING:
    import torch


class PODCoarseningStrategy(nn.Module):
    """Coarsening via a POD basis fit to a snapshot ensemble (POD-2G), sparse-CSR sibling.

    Sparse-CSR sibling of
    ``preconditioners.implementations.pod.coarsening.PODCoarseningStrategy``
    - same name, disambiguated by package, same algorithm and same
    ``nn.Module``-buffer approach for the fitted basis (see the dense
    class's docstring for the full rationale: it follows ``.to(device/
    dtype)`` the same way ``JacobiPreconditioner``/``AMGPreconditioner`` do).
    The only behavioral difference is ``build_transfer``: for sparse CSR
    ``A``, the Galerkin coarse operator is formed via
    ``torchalg.sparse.kernels.galerkin.form_sparse_dense`` (SpMM,
    ``Phi_r.T @ (A @ Phi_r)``) instead of two dense matmuls, so the
    fine-grid matrix is never densified.

    Args:
        rank (int | float): Fixed mode count (int) or minimum cumulative
            captured energy (float in (0, 1]) - see ``compute_pod_basis``.
            Resolved to an actual mode count by ``fit()`` and exposed
            afterward via the ``rank`` property.

    References:
        - Nikolopoulos, S., Kalogeris, I., Stavroulakis, G., & Papadopoulos,
          V. (2022). AI-enhanced iterative solvers for accelerating the
          solution of large-scale parametrized systems. arXiv:2207.02543.
    """

    def __init__(self, rank: float) -> None:
        """Store the target rank; the basis is not fit until ``fit()`` runs.

        Args:
            rank (int | float): Fixed mode count (int) or minimum cumulative
                captured energy (float in (0, 1]).

        Raises:
            ValueError: If ``rank`` is a float outside (0, 1] - the one part
                of rank validation that needs no snapshot data, so it is
                checked eagerly here rather than deferred to ``fit()``.
        """
        if isinstance(rank, float) and not (0.0 < rank <= 1.0):
            raise ValueError(f"rank as an energy threshold must be in (0, 1], got {rank}")
        nn.Module.__init__(self)
        self._rank = rank

    def fit(self, snapshots: torch.Tensor, row_scales: torch.Tensor | None = None) -> None:
        """Fit the POD basis from the snapshot ensemble (one-shot closed-form SVD).

        Args:
            snapshots (torch.Tensor): Solution snapshot ensemble, shape
                (n_samples, n_dofs).
            row_scales (torch.Tensor | None): Optional per-snapshot scale,
                shape (n_samples,), forwarded to ``compute_pod_basis`` -
                see its docstring. ``None`` (default) fits the unweighted
                basis.

        Raises:
            ValueError: If ``rank`` is an int exceeding
                ``min(n_samples, n_dofs)`` - only knowable once snapshots
                exist, unlike the float-range check in ``__init__``.
        """
        self._basis: torch.Tensor
        self.register_buffer(
            "_basis", compute_pod_basis(snapshots, self._rank, row_scales=row_scales)
        )

    def is_fitted(self) -> bool:
        """Whether ``fit()`` has registered the basis buffer.

        Returns:
            bool: ``True`` once ``fit()`` (or a successful
                ``load_state_dict``) has populated ``_basis``.
        """
        return "_basis" in self._buffers and self._buffers["_basis"] is not None

    @property
    def rank(self) -> int:
        """The resolved mode count of the fitted basis.

        Returns:
            int: Resolved mode count.

        Raises:
            RuntimeError: If called before ``fit()``.
        """
        if not self.is_fitted():
            raise RuntimeError("PODCoarseningStrategy must be fit() before rank is available.")
        return self._basis.shape[1]

    def build_transfer(self, A: torch.Tensor) -> tuple[torch.Tensor, DenseTransferOperator]:
        """Build the POD-reduced coarse level from the sparse CSR fine-grid matrix A.

        Args:
            A (torch.Tensor): Fine-grid matrix ``A``, shape ``(n, n)``,
                sparse CSR. Named to match the ``CoarseningStrategy``
                protocol's parameter name exactly.

        Returns:
            tuple[torch.Tensor, DenseTransferOperator]: ``(A_coarse,
                transfer)`` - ``A_coarse`` is the dense Galerkin coarse
                matrix (rank x rank), formed via ``form_sparse_dense``
                without ever densifying ``A``; ``transfer`` applies
                Phi_r/Phi_r^T, identical to the dense sibling's.

        Raises:
            RuntimeError: If called before ``fit()``.
        """
        if not self.is_fitted():
            raise RuntimeError("PODCoarseningStrategy must be fit() before build_transfer().")
        A_coarse = form_sparse_dense(A, self._basis)
        return A_coarse, DenseTransferOperator(self._basis)

    def __str__(self) -> str:
        """Human-readable structural summary.

        Returns:
            str: ``"POD-2G(rank=<n>)"`` once fit, or
                ``"POD-2G(not yet fit)"`` before ``fit()`` has run.
        """
        if not self.is_fitted():
            return "POD-2G(not yet fit)"
        return f"POD-2G(rank={self.rank})"
