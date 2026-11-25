"""POD-based coarsening strategy (POD-2G, Nikolopoulos et al. 2022)."""

from __future__ import annotations

import torch
from torch import nn

from torchalg.utils.dense_transfer import DenseTransferOperator


class PODCoarseningStrategy(nn.Module):
    """Coarsening onto a fitted POD basis (POD-2G), for dense or sparse-CSR matrices.

    Built from an already fitted basis ``Phi_r`` (see ``compute_pod_basis``), so an
    instance is always ready: there is no unfitted state, and ``rank`` is always the
    basis column count. Fitting is the pure function ``compute_pod_basis``; this class
    only projects a fine-grid matrix onto the basis.

    Subclasses ``nn.Module`` (one buffer, no learnable parameters) purely so the basis
    follows ``.to(device/dtype)``: assigned to ``AMGPreconditioner._coarsening`` it is
    auto-registered as a submodule and moves with its owning preconditioner.

    One class serves both matrix layouts. The basis, the Galerkin coarse matrix
    ``K_r = Phi_r^T K Phi_r`` (Nikolopoulos et al. 2022, eq. 28) and the transfer
    operator are dense whatever the layout of ``A``; only the product ``A @ Phi_r``
    sees ``A``, and ``torch`` forms it for a strided or a sparse-CSR ``A`` alike (an
    SpMM for CSR). That is why POD has no sparse sibling, unlike aggregation, whose
    prolongation and coarse matrix are themselves sparse.

        >>> strategy = PODCoarseningStrategy(compute_pod_basis(snapshots, rank=10))
        >>> a_coarse, transfer = strategy.build_transfer(A)  # A strided or CSR

    Args:
        basis (torch.Tensor): Fitted basis ``Phi_r``, shape ``(n_dofs, r)``, strided.

    Raises:
        ValueError: If ``basis`` is not a strided 2-D tensor with at least one column.

    References:
        - Nikolopoulos, S., Kalogeris, I., Stavroulakis, G., & Papadopoulos, V. (2022).
          AI-enhanced iterative solvers for accelerating the solution of large-scale
          parametrized systems. arXiv:2207.02543.

    Complexity (``n`` = ``n_dofs``, ``r`` = ``rank``, ``nnz`` = stored entries of ``A``):
        - ``build_transfer()``: ``Phi_r.T @ (A @ Phi_r)``, once per matrix. Dense ``A``:
          ``O(r * n^2)``; CSR ``A``: ``O(r * nnz + r^2 * n)``. Evaluated right to left so
          the sparse product never needs a dense-times-sparse kernel.
        - Per application (``prolongate``/``restrict``): ``O(r * n)``.

    Cost:
        Per application (``prolongate`` or ``restrict``, one right-hand side),
        ``V = 8`` bytes per ``float64``: a dense matrix-vector product with ``Phi_r``,
        ``W = 2 n r``, ``Q = V (n r + n + r)``, ``D = ceil(log2 n) + 1`` for
        ``restrict`` and ``ceil(log2 r) + 1`` for ``prolongate`` (a row reduction),
        ``L = 1``, ``S = 0``. Roofline: ``W/Q`` is about 0.25 flop per byte
        (memory-bound). Work-depth: ``W/D`` is large, so the transfers parallelize
        well; a sparse aggregation transfer costs ``O(n)`` instead of ``O(n r)``
        because the basis is dense by construction, and these transfers are a large share
        of the V-cycle at ``r`` in the tens. The one-off ``build_transfer`` cost is on
        that method.
    """

    def __init__(self, basis: torch.Tensor) -> None:
        """Hold the fitted basis as a buffer.

        Args:
            basis (torch.Tensor): Fitted basis ``Phi_r``, shape ``(n_dofs, r)``, strided.

        Raises:
            ValueError: If ``basis`` is not a strided 2-D tensor with at least one column.
        """
        if basis.layout != torch.strided or basis.ndim != 2 or basis.shape[1] < 1:
            raise ValueError(
                "basis must be a strided (n_dofs, r) tensor with r >= 1, got layout "
                f"{basis.layout} and shape {tuple(basis.shape)}"
            )
        nn.Module.__init__(self)
        self._basis: torch.Tensor
        self.register_buffer("_basis", basis)

    @property
    def basis(self) -> torch.Tensor:
        """The fitted basis ``Phi_r``, shape ``(n_dofs, r)``.

        Returns:
            torch.Tensor: The basis buffer.
        """
        return self._basis

    @property
    def rank(self) -> int:
        """The mode count ``r``: the coarse dimension.

        Returns:
            int: Number of basis columns.
        """
        return self._basis.shape[1]

    def build_transfer(self, A: torch.Tensor) -> tuple[torch.Tensor, DenseTransferOperator]:
        """Build the POD-reduced coarse level from the fine-grid matrix ``A``.

        Cost:
            ``n`` = ``n_dofs``, ``r`` = ``rank``, ``z = nnz(A)`` (``n^2`` if dense),
            ``V = 8`` bytes per ``float64``, ``w`` the index width, once per matrix.
            Executed: ``W = 2 z r + 2 n r^2`` (one SpMM, one GEMM),
            ``Q = z (V + w) + (n + 1) w + 2 V n r`` for a CSR ``A``'s SpMM (``V (n^2 + 2 n r)`` for dense ``A``) plus ``V (2 n r + r^2)``
            for the GEMM, ``D = ceil(log2 m) + 1 + ceil(log2 n) + 1`` with ``m`` the
            largest row count of ``A``, ``L = 2`` and ``S = 0``. Algorithmic minimum for
            the GEMM: ``n r^2``, since ``Phi^T (A Phi)`` is symmetric for symmetric ``A``;
            a symmetric-panel product cuts ``W`` by a fraction but adds launches, and
            the full product stays until a triangular-output GEMM kernel exists.
            Roofline: the SpMM has ``W/Q`` of order 1 (memory-bound), the GEMM about
            ``r / 8`` flops per byte, so the step leans compute-bound for ``r`` in the tens.
            Work-depth: ``W/D`` is large (about ``10^6``), the most parallel stage of the
            POD setup; the spectral estimate for the smoother, not this step, is its
            long chain.

        Args:
            A (torch.Tensor): Fine-grid matrix, shape ``(n_dofs, n_dofs)``, strided or
                sparse CSR. Named to match the ``CoarseningStrategy`` protocol.

        Returns:
            tuple[torch.Tensor, DenseTransferOperator]: ``(A_coarse, transfer)`` -
                ``A_coarse`` is the dense Galerkin matrix ``(r, r)``; ``transfer``
                applies ``Phi_r`` / ``Phi_r^T``.
        """
        A_coarse = self._basis.mT @ (A @ self._basis)
        return A_coarse, DenseTransferOperator(self._basis)

    def __str__(self) -> str:
        """Human-readable structural summary.

        Returns:
            str: ``"POD-2G(rank=<r>)"``.
        """
        return f"POD-2G(rank={self.rank})"
