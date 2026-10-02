"""Sparse-CSR sibling of the dense alpha-SA ``make_bridge`` row-extension.

Sparse-CSR counterpart of
``preconditioners.implementations.amg._prolongation.make_bridge`` (dense;
kept unmodified for comparison - see ``docs/plan.md``'s "Correction: dense
and sparse must be separate implementations, not an internal branch"). The
dense version reshapes ``(n_nodes, dofs, m)`` and inserts one all-zero row
per node-block for the new candidate's dof. For sparse CSR every inserted
row is structurally empty (zero stored entries), so this reduces to a pure
CSR index remap - ``crow_indices`` rebuilt with one extra empty gap
inserted per node-block - with no dense intermediate and no change to
``col_indices()``/``values()`` at all.
"""

from __future__ import annotations

import torch

from torchalg.sparse.kernels.triangular import _require_csr


def sparse_make_bridge(tentative: torch.Tensor, dofs_per_node: int) -> torch.Tensor:
    """Extend ``T`` by one always-zero dof per node, sparse CSR input.

    Args:
        tentative (torch.Tensor): Sparse CSR ``T``, shape
            ``(n_nodes * dofs_per_node, n_coarse)``.
        dofs_per_node (int): Current dofs per node ``K``.

    Returns:
        torch.Tensor: Sparse CSR bridge of shape
            ``(n_nodes * (K + 1), n_coarse)`` whose new (last) dof rows per
            node are structurally empty.

    Raises:
        ValueError: If ``tentative`` is not sparse CSR.
    """
    _require_csr(tentative, "sparse_make_bridge")

    n = tentative.shape[0]
    n_nodes = n // dofs_per_node
    new_dofs = dofs_per_node + 1
    new_n = n_nodes * new_dofs

    crow = tentative.crow_indices()
    row_nnz = crow[1:] - crow[:-1]

    old_row = torch.arange(n, device=tentative.device)
    new_row_index = old_row + old_row // dofs_per_node

    new_row_nnz = torch.zeros(new_n, dtype=row_nnz.dtype, device=tentative.device)
    new_row_nnz[new_row_index] = row_nnz

    new_crow = torch.zeros(new_n + 1, dtype=torch.long, device=tentative.device)
    new_crow[1:] = torch.cumsum(new_row_nnz, dim=0)

    return torch.sparse_csr_tensor(
        new_crow,
        tentative.col_indices(),
        tentative.values(),
        size=(new_n, tentative.shape[1]),
        check_invariants=False,
    )
