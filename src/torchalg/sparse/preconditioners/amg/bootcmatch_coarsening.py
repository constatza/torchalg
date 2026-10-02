"""BootCMatch coarsening strategy (one hierarchy level), sparse-CSR sibling.

Sparse-CSR counterpart of
``preconditioners.implementations.amg.bootcmatch_coarsening.BootCMatchCoarsening``
(dense; kept unmodified for comparison) - see that module's docstring for
the full per-level algorithm description (TOMS Algorithm 2: BootCMatch
edge weights, parallel pointer-based matching, pairwise-aggregation
prolongator construction). Every change from the dense class follows
directly from storage format, not a different algorithm:

- Edge weights/matching use this package's own sparse-CSR Components A/B
  (``kernels.bootcmatch_weights``/``kernels.bootcmatch_matching``).
- Component C (``torchalg.utils.bootcmatch_aggregation
  .bootcmatch_aggregates_from_matching``) is reused as-is, unchanged from
  the dense class's import - it is format-agnostic by construction (see
  its own module docstring).
- ``P`` is materialized sparse CSR directly from the Component C triples
  via ``kernels.prolongation.sparse_interpolation_prolongation`` (already
  existing, already used by this package's own ``BAMGCoarsening
  ._prolongation``) - never a dense intermediate.
- The Galerkin coarse operator reuses ``kernels.galerkin
  .form_sparse_sparse`` (``A_coarse = P.T @ A @ P`` via the
  CSR-transpose-via-COO workaround already documented there and already
  used by this package's own ``BAMGCoarsening.build_transfer``) - no new
  sparse-sparse product kernel.
- The smooth-vector restriction ``P.T @ w`` goes through COO for the
  transpose step, mirroring every other sparse-CSR transpose-needing
  product in this package (``BAMGCoarsening``'s own ``_current_T``/
  restricted-vectors code).
"""

from __future__ import annotations

import torch

from torchalg.sparse.kernels.bootcmatch_matching import bootcmatch_parallel_matching
from torchalg.sparse.kernels.bootcmatch_weights import bootcmatch_edge_weights
from torchalg.sparse.kernels.galerkin import form_sparse_sparse
from torchalg.sparse.kernels.prolongation import sparse_interpolation_prolongation
from torchalg.utils.bootcmatch_aggregation import bootcmatch_aggregates_from_matching

from .transfer import SparseTransferOperator

__all__ = ["BootCMatchCoarsening"]


class BootCMatchCoarsening:
    """One coarse level of BootCMatch (TOMS Algorithm 2), sparse-CSR sibling.

    Args:
        w (torch.Tensor): Smooth vector for this level, shape ``(n,)``.
    """

    def __init__(self, w: torch.Tensor) -> None:
        """Store the finest-level smooth vector.

        Args:
            w (torch.Tensor): Smooth vector, shape ``(n,)``.
        """
        self._vectors: dict[int, torch.Tensor] = {w.shape[0]: w}
        self._last_prolongation: torch.Tensor | None = None

    def smooth_vector_for(self, matrix: torch.Tensor) -> torch.Tensor:
        """Currently stored smooth vector for ``matrix``'s dimension.

        Args:
            matrix (torch.Tensor): Level matrix, shape ``(n, n)``.

        Returns:
            torch.Tensor: Smooth vector, shape ``(n,)``.
        """
        return self._vectors[matrix.shape[0]]

    def set_smooth_vector(self, matrix: torch.Tensor, w: torch.Tensor) -> None:
        """Overwrite the stored smooth vector for ``matrix``'s dimension.

        Args:
            matrix (torch.Tensor): Level matrix, shape ``(n, n)``.
            w (torch.Tensor): New smooth vector, shape ``(n,)``.
        """
        self._vectors[matrix.shape[0]] = w

    @property
    def last_prolongation(self) -> torch.Tensor:
        """Sparse CSR prolongation tensor ``P`` from the most recent ``build_transfer`` call.

        Returns:
            torch.Tensor: Sparse CSR prolongation matrix, shape ``(n, n_c)``.

        Raises:
            RuntimeError: If ``build_transfer`` has not been called yet.
        """
        if self._last_prolongation is None:
            raise RuntimeError("build_transfer has not been called yet")
        return self._last_prolongation

    def build_transfer(self, A: torch.Tensor) -> tuple[torch.Tensor, SparseTransferOperator]:
        """Build one BootCMatch coarse level from sparse CSR ``A``.

        Args:
            A (torch.Tensor): Sparse CSR fine-grid matrix, shape ``(n, n)``.

        Returns:
            tuple[torch.Tensor, SparseTransferOperator]: ``(A_coarse, transfer)``.
        """
        w = self._vectors[A.shape[0]]
        weights, fine_only_mask = bootcmatch_edge_weights(w, A)
        match_of = bootcmatch_parallel_matching(weights, fine_only_mask)
        row, col, value, n_coarse = bootcmatch_aggregates_from_matching(w, match_of)

        n = A.shape[0]
        prolongation = sparse_interpolation_prolongation(row, col, value.to(A.dtype), (n, n_coarse))

        coarse_matrix = form_sparse_sparse(prolongation, A)
        self._last_prolongation = prolongation

        p_coo = prolongation.to_sparse_coo()
        restricted_w = torch.sparse.mm(p_coo.t(), w.unsqueeze(1)).squeeze(1)
        self._vectors[coarse_matrix.shape[0]] = restricted_w
        return coarse_matrix, SparseTransferOperator(prolongation)
