"""Tentative prolongator fitted to near-null-space candidates (PyAMG ``fit_candidates``).

Dense torch port of ``amg_core::fit_candidates_common``. For each aggregate
the candidates restricted to its rows are orthonormalized column by column
with modified Gram-Schmidt; a column whose norm after orthogonalization is
not above ``tol`` times its norm before is set to zero (and ``R``'s diagonal
entry to 0). Every aggregate always contributes ``m`` (the candidate count)
columns, so the coarse dimension is ``m * n_aggregates``, and
``Q R = vectors`` on every non-isolated row.

Nodes may carry several dofs: ``vectors`` has ``dofs_per_node * n_nodes``
rows, node-major, and ``aggregate`` is defined on nodes (PyAMG's BSR layout).

References:
    - PyAMG 5.3.0, ``pyamg/aggregation/tentative.py`` and
      ``pyamg/amg_core/smoothed_aggregation.h``.
    - Vanek, P., Mandel, J., & Brezina, M. (1996). Computing, 56(3), 179-196.
"""

from __future__ import annotations

import torch


def _fit_aggregate(block: torch.Tensor, tol: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Modified Gram-Schmidt of one aggregate's candidate block, with PyAMG's drop rule.

    Args:
        block (torch.Tensor): Candidates restricted to the aggregate's dofs,
            shape ``(n_rows, m)``.
        tol (float): Relative drop threshold.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: ``(Q, R)`` of shapes ``(n_rows, m)``
            and ``(m, m)`` (upper triangular).
    """
    q = block.clone()
    r = torch.zeros(block.shape[1], block.shape[1], dtype=block.dtype, device=block.device)
    for j in range(block.shape[1]):
        threshold = tol * torch.linalg.norm(q[:, j])
        for i in range(j):
            r[i, j] = torch.dot(q[:, j], q[:, i])
            q[:, j] = q[:, j] - r[i, j] * q[:, i]
        norm = torch.linalg.norm(q[:, j])
        if norm > threshold:
            r[j, j] = norm
            q[:, j] = q[:, j] / norm
        else:
            q[:, j] = 0.0
    return q, r


def fit_candidates(
    aggregate: torch.Tensor, vectors: torch.Tensor, tol: float = 1e-10
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fit candidates to a tentative prolongator ``Q`` and coarse candidates ``R``.

    Args:
        aggregate (torch.Tensor): Long tensor of length ``n_nodes`` giving each
            node's aggregate, ``-1`` if isolated (row of ``Q`` left zero).
        vectors (torch.Tensor): Candidates, shape ``(dofs_per_node * n_nodes, m)``.
        tol (float): Relative threshold for dropping dependent columns.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: ``(Q, R)`` with shapes
            ``(n_dofs, m * n_aggregates)`` and ``(m * n_aggregates, m)``.
    """
    n_nodes, m = aggregate.shape[0], vectors.shape[1]
    dofs = vectors.shape[0] // n_nodes
    n_aggregates = max(int(aggregate.max().item()) + 1, 1)
    q = torch.zeros(vectors.shape[0], n_aggregates * m, dtype=vectors.dtype, device=vectors.device)
    r = torch.zeros(n_aggregates * m, m, dtype=vectors.dtype, device=vectors.device)
    dof_of_node = torch.arange(vectors.shape[0], device=vectors.device).reshape(n_nodes, dofs)
    for index in range(n_aggregates):
        rows = dof_of_node[aggregate == index].flatten()
        q_block, r_block = _fit_aggregate(vectors[rows], tol)
        q[rows, index * m : (index + 1) * m] = q_block
        r[index * m : (index + 1) * m] = r_block
    return q, r
