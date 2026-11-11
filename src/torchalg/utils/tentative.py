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

Unlike PyAMG's reference (which fits one aggregate at a time in a C++ loop),
``fit_candidates`` here pads every aggregate's row-block to the largest
aggregate's size and runs modified Gram-Schmidt as ONE batched computation
over the aggregate dimension: aggregates are independent, so a Python loop
over ``n_aggregates`` (thousands, for a real mesh) was launching that many
tiny CUDA kernels per candidate build and stalling on GPU almost entirely on
kernel-launch/dispatch overhead rather than compute. The batched form issues
``O(m^2)`` kernel launches total (``m`` = candidate count), independent of
``n_aggregates`` - see ``_fit_aggregates_batched``.

References:
    - PyAMG 5.3.0, ``pyamg/aggregation/tentative.py`` and
      ``pyamg/amg_core/smoothed_aggregation.h``.
    - Vanek, P., Mandel, J., & Brezina, M. (1996). Computing, 56(3), 179-196.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


def _fit_aggregates_batched(block: torch.Tensor, tol: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Modified Gram-Schmidt of every aggregate's candidate block, batched.

    Runs PyAMG's per-aggregate Gram-Schmidt (and its tolerance-drop rule) as
    one computation across the aggregate dimension instead of a Python loop
    over aggregates - only the ``m``-column loop remains in Python, so the
    kernel-launch count is ``O(m^2)`` regardless of how many aggregates
    there are.

    Args:
        block (torch.Tensor): Padded per-aggregate candidate blocks, shape
            ``(n_aggregates, max_size, m)``. Padded rows (an aggregate
            smaller than ``max_size``) must be exactly zero: a zero row
            contributes nothing to any dot product or norm here, so padding
            is mathematically transparent to the per-aggregate result.
        tol (float): Relative drop threshold.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: ``(Q, R)`` of shapes
            ``(n_aggregates, max_size, m)`` and ``(n_aggregates, m, m)``
            (each ``R[a]`` upper triangular).
    """
    m = block.shape[2]
    q = block.clone()
    r = torch.zeros(block.shape[0], m, m, dtype=block.dtype, device=block.device)
    for j in range(m):
        threshold = tol * torch.linalg.norm(q[:, :, j], dim=1)
        for i in range(j):
            coeff = (q[:, :, j] * q[:, :, i]).sum(dim=1)
            r[:, i, j] = coeff
            q[:, :, j] = q[:, :, j] - coeff.unsqueeze(1) * q[:, :, i]
        norm = torch.linalg.norm(q[:, :, j], dim=1)
        # torch.where, not a Python `if norm > threshold`: batched over every
        # aggregate at once, so a host sync here would still be paid only
        # once per column (m times per hierarchy level per candidate build),
        # not once per (aggregate, candidate) pair.
        keep = norm > threshold
        r[:, j, j] = torch.where(keep, norm, torch.zeros_like(norm))
        safe_norm = torch.where(keep, norm, torch.ones_like(norm))
        q[:, :, j] = torch.where(
            keep.unsqueeze(1), q[:, :, j] / safe_norm.unsqueeze(1), torch.zeros_like(q[:, :, j])
        )
    return q, r


@dataclass(frozen=True, slots=True)
class _TentativeEntries:
    """Stored entries of ``Q`` (one row per valid dof, ``m`` columns each) and ``R``."""

    rows: torch.Tensor
    cols: torch.Tensor
    values: torch.Tensor
    coarse: torch.Tensor
    shape: tuple[int, int]


def _tentative_entries(
    aggregate: torch.Tensor, vectors: torch.Tensor, tol: float
) -> _TentativeEntries:
    device, dtype = vectors.device, vectors.dtype
    n_nodes, m = aggregate.shape[0], vectors.shape[1]
    dofs = vectors.shape[0] // n_nodes
    n_aggregates = max(int(aggregate.max().item()) + 1, 1)
    shape = (vectors.shape[0], n_aggregates * m)
    empty_index = torch.empty(0, m, dtype=torch.long, device=device)

    # dof-major aggregate id per row of `vectors`, in the same node-major
    # row order `dof_of_node[aggregate == index]` would visit.
    dof_aggregate = torch.repeat_interleave(aggregate, dofs)
    valid_rows = torch.nonzero(dof_aggregate >= 0, as_tuple=False).flatten()
    if valid_rows.numel() == 0:
        return _TentativeEntries(
            rows=empty_index,
            cols=empty_index,
            values=torch.empty(0, m, dtype=dtype, device=device),
            coarse=torch.zeros(n_aggregates * m, m, dtype=dtype, device=device),
            shape=shape,
        )
    valid_agg = dof_aggregate[valid_rows]

    # Group rows by aggregate while preserving each aggregate's original
    # row order (stable sort), then scatter into a padded (n_aggregates,
    # max_size) index grid - one sync for the whole matrix (`counts.max()`),
    # not one per aggregate.
    order = torch.argsort(valid_agg, stable=True)
    sorted_rows, sorted_agg = valid_rows[order], valid_agg[order]
    counts = torch.bincount(sorted_agg, minlength=n_aggregates)
    offsets = torch.cumsum(counts, 0) - counts
    pos_in_group = torch.arange(sorted_rows.numel(), device=device) - offsets[sorted_agg]
    max_size = int(counts.max().item())

    padded_rows = torch.full((n_aggregates, max_size), -1, dtype=torch.long, device=device)
    padded_rows[sorted_agg, pos_in_group] = sorted_rows
    valid_pad = padded_rows >= 0

    block = vectors[padded_rows.clamp(min=0)] * valid_pad.unsqueeze(-1).to(dtype)
    q_batched, r_batched = _fit_aggregates_batched(block, tol)

    valid_a, valid_s = torch.nonzero(valid_pad, as_tuple=True)
    rows_out = padded_rows[valid_a, valid_s]
    col_idx = valid_a.unsqueeze(1) * m + torch.arange(m, device=device).unsqueeze(0)
    return _TentativeEntries(
        rows=rows_out.unsqueeze(1).expand(-1, m),
        cols=col_idx,
        values=q_batched[valid_a, valid_s],
        coarse=r_batched.reshape(n_aggregates * m, m),
        shape=shape,
    )


def fit_candidates(
    aggregate: torch.Tensor, vectors: torch.Tensor, tol: float = 1e-10
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fit candidates to a dense tentative prolongator ``Q`` and coarse candidates ``R``.

    Args:
        aggregate (torch.Tensor): Long tensor of length ``n_nodes`` giving each
            node's aggregate, ``-1`` if isolated (row of ``Q`` left zero).
        vectors (torch.Tensor): Candidates, shape ``(dofs_per_node * n_nodes, m)``.
        tol (float): Relative threshold for dropping dependent columns.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: ``(Q, R)`` with shapes
            ``(n_dofs, m * n_aggregates)`` and ``(m * n_aggregates, m)``.
    """
    entries = _tentative_entries(aggregate, vectors, tol)
    q = torch.zeros(entries.shape, dtype=vectors.dtype, device=vectors.device)
    q[entries.rows, entries.cols] = entries.values
    return q, entries.coarse


def fit_candidates_csr(
    aggregate: torch.Tensor, vectors: torch.Tensor, tol: float = 1e-10
) -> tuple[torch.Tensor, torch.Tensor]:
    """``fit_candidates`` with ``Q`` built directly as sparse CSR.

    Identical to ``fit_candidates(...)[0].to_sparse_csr()`` (same indices and values,
    exact zeros dropped) without the dense ``(n_dofs, m * n_aggregates)``
    intermediate, which is quadratic in n: at n = 90 000 the dense build and its
    conversion took 11.6 of a 15 s adaptive SA setup.

    Args:
        aggregate (torch.Tensor): Long tensor of length ``n_nodes`` giving each
            node's aggregate, ``-1`` if isolated (row of ``Q`` left empty).
        vectors (torch.Tensor): Candidates, shape ``(dofs_per_node * n_nodes, m)``.
        tol (float): Relative threshold for dropping dependent columns.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: ``(Q, R)``, ``Q`` sparse CSR of shape
            ``(n_dofs, m * n_aggregates)``, ``R`` dense ``(m * n_aggregates, m)``.
    """
    entries = _tentative_entries(aggregate, vectors, tol)
    rows, cols, values = (
        entries.rows.reshape(-1),
        entries.cols.reshape(-1),
        entries.values.reshape(-1),
    )
    stored = values != 0
    rows, cols, values = rows[stored], cols[stored], values[stored]
    # Each row's columns are already ascending (aggregate * m + j), so a stable sort
    # by row yields CSR order.
    order = torch.argsort(rows, stable=True)
    row_counts = torch.bincount(rows, minlength=entries.shape[0])
    crow = torch.zeros(entries.shape[0] + 1, dtype=torch.long, device=rows.device)
    crow[1:] = torch.cumsum(row_counts, 0)
    q = torch.sparse_csr_tensor(crow, cols[order], values[order], entries.shape)
    return q, entries.coarse
