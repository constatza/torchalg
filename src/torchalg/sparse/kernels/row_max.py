"""Per-row max reduction over a sparse CSR tensor's stored entries.

No standalone correctness/breakdown story of its own - a primitive consumed
by ``torchalg.sparse.kernels.strength.sparse_strength_graph`` (Bootstrap
AMG's pruning step), replacing dense ``distance.max(dim=1).values``.
"""

from __future__ import annotations

import torch

from .triangular import _expand_row_index, _require_csr


def sparse_row_max(matrix: torch.Tensor, *, default: float = 0.0) -> torch.Tensor:
    """Per-row max over a sparse CSR tensor's stored entries, 0 treated as an implicit candidate.

    Equal to ``matrix.to_dense().max(dim=1).values`` for any sparse CSR
    ``matrix``: every non-stored position reads back as ``default`` (0.0 by
    default, matching dense densification), so a row whose stored entries
    are all negative still reports ``default``, not its least-negative
    entry - the same semantics dense callers get "for free" from
    densifying. Implemented via ``scatter_reduce_(reduce="amax",
    include_self=True)``, which is exactly this: every row starts at
    ``default`` and only grows from there.

    Args:
        matrix (torch.Tensor): Sparse CSR matrix, shape ``(n, n)``.
        default (float): Value every row starts at - the implicit
            candidate a row with no stored entries (or only
            smaller-than-default entries) reports.

    Returns:
        torch.Tensor: Dense ``(n,)`` per-row max.

    Raises:
        ValueError: If ``matrix`` is not sparse CSR.
    """
    _require_csr(matrix, "sparse_row_max")
    n = matrix.shape[0]
    row = _expand_row_index(matrix)
    values = matrix.values()

    result = torch.full((n,), default, dtype=values.dtype, device=matrix.device)
    result.scatter_reduce_(0, row, values, reduce="amax", include_self=True)
    return result
