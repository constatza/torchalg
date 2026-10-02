"""Vectorized parallel approximate maximum-weight matching (dense).

Port of the "locally dominant edge" / pointer-based matching algorithm
(Halappanavar's PhD thesis, Algorithms 20-22; identical in substance to
D'Ambra/Filippone/Vassilevski's TOMS paper, Algorithm 3 - the practical
basis for what later GPU literature calls "Suitor"). Unlike classical
greedy matching, which processes vertices/edges in a strict priority order
where decision ``N`` depends on decisions ``1..N-1``, this algorithm has no
such intra-round dependency: every round, every still-``active`` vertex
simultaneously computes its own best move using only the current
(unmatched) state, and only across rounds is there sequential structure (a
small, data-dependent number of rounds, not one Python iteration per
vertex).

Each round is a single batched tensor op over every remaining vertex at
once:

1. Every active vertex ``v`` picks its heaviest active, non-self neighbor
   as ``candidate[v]`` (``-1`` if none remain).
2. An edge ``(v, candidate[v])`` is a "locally dominant edge" - and thus
   gets matched this round - exactly when it is *mutual*: ``v``'s
   candidate also picked ``v`` back.
3. Vertices matched this round are removed from ``active``; every other
   active vertex's candidate is recomputed from scratch next round (no
   separate "reset pointer" bookkeeping is needed, unlike the thesis's
   distributed-memory version, which tracks this explicitly for
   messaging efficiency - recomputing the argmax fresh each round over the
   updated ``active`` set already has the identical effect).

This terminates because every round with any ``active`` vertex remaining
matches at least the single globally-heaviest remaining edge (that edge's
two endpoints are always mutually each other's argmax, by definition of
"heaviest"), so progress is guaranteed each round; the round count is
data-dependent, not a fixed small constant.

References:
    - Halappanavar, M. (2009). Algorithms for Vertex-Weighted Matching in
      Graphs. PhD thesis, Old Dominion University. Algorithms 20-22.
    - D'Ambra, P., Filippone, S., & Vassilevski, P. S. (2018). BootCMatch: A
      Software Package for Bootstrap AMG based on Graph Weighted Matching.
      ACM TOMS. Algorithm 3.
"""

from __future__ import annotations

import torch

_SYNC_CHECK_STRIDE = 4
"""Host-sync the round-loop's stop condition every this many rounds, not every round.

See the sparse sibling's identical constant/comment
(``torchalg.sparse.kernels.bootcmatch_matching``) for the no-op argument:
this only trades a few extra cheap tensor-op rounds for far fewer
host-device round trips, never a different result.
"""


def bootcmatch_parallel_matching(
    weights: torch.Tensor, fine_only_mask: torch.Tensor
) -> torch.Tensor:
    """Round-based parallel approximate max-weight matching (dense).

    Args:
        weights (torch.Tensor): Dense edge-weight matrix, shape ``(n, n)``.
            Degenerate/absent edges are already ``-inf``; higher is more
            desirable. The diagonal is ignored (masked to ``-inf``
            internally) regardless of its input value.
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
    n = weights.shape[0]
    device = weights.device
    arange = torch.arange(n, device=device)

    active = ~fine_only_mask
    matched = torch.zeros(n, dtype=torch.bool, device=device)
    match_of = torch.full((n,), -1, dtype=torch.long, device=device)

    neg_inf = torch.full_like(weights, float("-inf"))

    round_index = 0
    while round_index % _SYNC_CHECK_STRIDE != 0 or bool(active.any()):
        row_mask = active.unsqueeze(0) & active.unsqueeze(1)
        masked = torch.where(row_mask, weights, neg_inf)
        masked = masked.clone()
        masked.fill_diagonal_(float("-inf"))

        candidate = masked.argmax(dim=1)
        best_value = masked.gather(1, candidate.unsqueeze(1)).squeeze(1)
        has_valid = torch.isfinite(best_value)
        candidate = torch.where(has_valid, candidate, torch.full_like(candidate, -1))

        safe_candidate = candidate.clamp(min=0)
        points_back = candidate[safe_candidate] == arange
        mutual = active & has_valid & points_back

        match_of = torch.where(mutual, candidate, match_of)
        matched = matched | mutual
        # Deactivate both freshly matched vertices and vertices with no
        # valid candidate at all this round. The latter can never regain
        # one later: `active` only shrinks round over round, so a vertex
        # whose neighborhood is already exhausted of active partners stays
        # exhausted forever - without this, such a vertex would keep
        # `active` true indefinitely and the loop would never terminate.
        active = active & ~matched & has_valid
        round_index += 1

    return match_of
