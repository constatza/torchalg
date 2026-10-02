"""Vectorized parallel approximate maximum-weight matching (sparse CSR).

Sparse twin of
``torchalg.preconditioners.implementations.amg._bootcmatch_matching
.bootcmatch_parallel_matching`` - same algorithm, same round-based
"locally dominant edge" rule (see that module's docstring for the full
derivation and termination argument), applied over a sparse CSR matrix's
*stored* entries only, instead of a densely masked ``(n, n)`` tensor.

**The sparse argmax-with-tie-break sub-problem.** This codebase's existing
``sparse_row_max`` (``torchalg.sparse.kernels.row_max``) gives the
row-grouped *maximum value* via ``scatter_reduce_(reduce="amax",
include_self=True)``, but a matching rule needs the *column index* that
achieved that max, with ties broken toward the lowest column index (the
thesis's own tie-break rule, Algorithm 20 line 5: "the lowest numbered
heaviest end-point... is chosen"). ``scatter_reduce`` has no "argmax"
reduction, so this is solved in two vectorized passes, mirroring
``sparse_row_max``'s own expand-and-scatter idiom:

1. Scatter-reduce (``amax``) the (masked) stored values into a per-row max
   value, exactly like ``sparse_row_max``.
2. Re-expand that per-row max back out to every stored entry (a plain
   gather by row index) and compare against each entry's own value to get
   a boolean "is this entry one of its row's achievers" mask. Among
   achievers, scatter-reduce (``amin``) each achiever's *column index*
   (non-achievers contributing a sentinel larger than any real column
   index) into a per-row result - the smallest column index among the
   row's maximum-value entries, i.e. exactly the tie-break rule.

Both passes are ``scatter_reduce_`` calls over the whole stored-entry
array at once; there is no Python loop over rows.

Both the "achieves the max" mask and the "must be active" mask (an edge is
only a candidate if both endpoints are currently active, and it is not a
self-loop) are computed fresh every round from the matrix's static
``(row, col, value)`` arrays - the active set changes, the sparsity
pattern never does - rather than mutating the sparse tensor's structure
in place, which is awkward and unnecessary when recomputation is this
cheap (not premature optimization to avoid).

References:
    - Halappanavar, M. (2009). Algorithms for Vertex-Weighted Matching in
      Graphs. PhD thesis, Old Dominion University. Algorithms 20-22 (the
      pointer-based "locally dominant edge" matching rule and its
      lowest-column-index tie-break, ported here in vectorized form).
    - D'Ambra, P., Filippone, S., & Vassilevski, P. S. (2018). BootCMatch: A
      Software Package for Bootstrap AMG based on Graph Weighted Matching.
      ACM Transactions on Mathematical Software, 44(4). Algorithm 3 (the
      same matching rule, confirmed identical in substance).
"""

from __future__ import annotations

import torch

from .triangular import _expand_row_index, _require_csr

_SYNC_CHECK_STRIDE = 4
"""Host-sync the round-loop's stop condition every this many rounds, not every round.

See the ``bootcmatch_parallel_matching`` round loop for why this is safe:
every round past true convergence is a provable no-op, so this only trades
a few extra cheap tensor-op rounds (at most ``_SYNC_CHECK_STRIDE - 1``) for
far fewer host-device round trips - the result is identical either way.
"""


def bootcmatch_parallel_matching(
    weights: torch.Tensor, fine_only_mask: torch.Tensor
) -> torch.Tensor:
    """Round-based parallel approximate max-weight matching (sparse CSR).

    Args:
        weights (torch.Tensor): Sparse CSR edge-weight matrix, shape
            ``(n, n)``. Non-stored positions are treated as non-edges
            (never a matching candidate); self-loop entries, if stored,
            are ignored regardless of their value.
        fine_only_mask (torch.Tensor): Bool, shape ``(n,)``. ``True`` where
            that vertex must never be offered as a matching candidate at
            all (and is therefore never matched).

    Returns:
        torch.Tensor: Long, shape ``(n,)``. ``match_of[v]`` is the vertex
        ``v`` is matched to, or ``-1`` if ``v`` ended up unmatched (either
        because ``fine_only_mask[v]`` was ``True``, or no valid partner
        remained). Guaranteed symmetric:
        ``match_of[match_of[v]] == v`` whenever ``match_of[v] != -1``.
    """
    _require_csr(weights, "bootcmatch_parallel_matching")
    n = weights.shape[0]
    device = weights.device
    arange = torch.arange(n, device=device)

    row = _expand_row_index(weights)
    col = weights.col_indices()
    val = weights.values()

    not_self_loop = row != col

    active = ~fine_only_mask
    matched = torch.zeros(n, dtype=torch.bool, device=device)
    match_of = torch.full((n,), -1, dtype=torch.long, device=device)

    sentinel_col = n
    neg_inf = float("-inf")

    # Host syncs (the `bool(...)` pull below) are themselves a measured
    # setup-time bottleneck at scale (see docs/plan.md): checking every
    # round forces one per round, and this loop runs once per coarsening
    # level per hierarchy. Checking only every `_SYNC_CHECK_STRIDE` rounds
    # cuts that by ~4x without changing `match_of`: once `active` is truly
    # empty, every further round body is a no-op (`active[row]`/`active[col]`
    # are all-False, so `valid` is all-False and nothing updates) - so the
    # worst case is up to `_SYNC_CHECK_STRIDE - 1` extra no-op rounds, never
    # a different result.
    round_index = 0
    while round_index % _SYNC_CHECK_STRIDE != 0 or bool(active.any()):
        valid = not_self_loop & active[row] & active[col]
        val_masked = torch.where(valid, val, torch.full_like(val, neg_inf))

        row_max = torch.full((n,), neg_inf, dtype=val.dtype, device=device)
        row_max.scatter_reduce_(0, row, val_masked, reduce="amax", include_self=True)
        has_valid = torch.isfinite(row_max)

        is_achiever = valid & (val_masked == row_max[row])
        col_for_tiebreak = torch.where(is_achiever, col, torch.full_like(col, sentinel_col))

        candidate = torch.full((n,), sentinel_col, dtype=torch.long, device=device)
        candidate.scatter_reduce_(0, row, col_for_tiebreak, reduce="amin", include_self=True)
        candidate = torch.where(has_valid, candidate, torch.full_like(candidate, -1))

        safe_candidate = candidate.clamp(min=0)
        points_back = candidate[safe_candidate] == arange
        mutual = active & has_valid & points_back

        match_of = torch.where(mutual, candidate, match_of)
        matched = matched | mutual
        # Deactivate both freshly matched vertices and vertices with no
        # valid candidate at all this round - see the dense kernel's
        # identical comment for why this is required for termination
        # (without it, a vertex with an exhausted neighborhood stays
        # `active` forever and the loop never ends).
        active = active & ~matched & has_valid
        round_index += 1

    return match_of
