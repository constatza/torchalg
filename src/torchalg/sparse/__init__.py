"""Sparse (``torch.sparse`` CSR) primitives for large-N AMG/POD preconditioners.

Companion to ``preconditioners.implementations``' dense-only kernels (see
``docs/plan.md``): every function here operates on ``torch.sparse_csr``
tensors directly, reusing ``torch.sparse_csr_tensor``/``torch.sparse.mm``
rather than a new custom tensor type. Functions run on whatever
device/dtype the input tensors already live on - no internal CPU<->CUDA
migration or size-based auto-dispatch (see ``docs/plan.md``'s "generic
CPU-vs-CUDA rule": that guidance is advisory, consulted by the caller before
constructing their tensors, never by these functions themselves).

Public API:
    - ``sparse_diagonal``: Extract a CSR matrix's diagonal without
      densifying it.
    - ``form_sparse_sparse``/``form_sparse_dense``: Sparse Galerkin coarse
      operator formation (AMG/POD shapes).
    - ``SparseTransferOperator``: Sparse-backed prolongation/restriction.
    - ``sparse_strength_of_connection``/``sparse_standard_aggregation``:
      Sparse-native SA-AMG coarsening kernels.
    - ``sparse_row_scale``: Per-row scaling of a sparse CSR matrix.
"""

from .aggregation import (
    sparse_piecewise_constant_prolongation,
    sparse_smoothed_prolongation,
    sparse_standard_aggregation,
    sparse_strength_of_connection,
)
from .diagonal import sparse_diagonal
from .galerkin import form_sparse_dense, form_sparse_sparse
from .rowscale import sparse_row_scale
from .transfer import SparseTransferOperator

__all__ = [
    "SparseTransferOperator",
    "form_sparse_dense",
    "form_sparse_sparse",
    "sparse_diagonal",
    "sparse_piecewise_constant_prolongation",
    "sparse_row_scale",
    "sparse_smoothed_prolongation",
    "sparse_standard_aggregation",
    "sparse_strength_of_connection",
]
