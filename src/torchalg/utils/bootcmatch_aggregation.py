"""BootCMatch pairwise-aggregation / tentative-prolongator construction, shared by both trees.

Given a final matching (Component B's ``match_of``), builds the aggregates
and the ``(row, col, value)`` triples of the tentative prolongator ``P``
per D'Ambra, Filippone, Vassilevski, "AMG Preconditioners for Linear
Solvers towards Extreme Scale" (arXiv:2006.16147), eq. 3.2-3.3:

- Matched pair ``{i, j}`` (``match_of[i] == j``): ``P``'s shared coarse
  column gets ``w_i / sqrt(w_i**2 + w_j**2)`` at row ``i`` and
  ``w_j / sqrt(w_i**2 + w_j**2)`` at row ``j`` - the plain L2-orthonormal
  projection of ``w`` onto that pair.
- Unmatched/singleton vertex ``s`` (``match_of[s] == -1``): the entry at
  row ``s`` is ``sign(w_s)``.

Per this codebase's "one legitimate shared edge" precedent
(``torchalg.utils.ls_interpolation``/``torchalg.utils.test_vector_weights``):
this computation only ever touches the plain ``(n,)`` tensors ``w`` and
``match_of`` - it never reads the system matrix's storage format - so one
implementation serves both the dense and sparse ``BootCMatchCoarsening``
trees; each tree's own wrapper materializes the returned triples into its
own storage format using machinery that already exists in that tree
(sparse: ``torchalg.sparse.kernels.prolongation
.sparse_interpolation_prolongation``; dense: a plain scatter-assignment),
rather than this module doing any materialization itself.

Fully vectorized, no Python loop over vertices - see
``bootcmatch_aggregates_from_matching``'s own docstring for the derivation.

References:
    - D'Ambra, P., Filippone, S., & Vassilevski, P. S. (2020). AMG
      Preconditioners for Linear Solvers towards Extreme Scale.
      arXiv:2006.16147, eq. 3.2-3.3.
"""

from __future__ import annotations

import torch

__all__ = ["bootcmatch_aggregates_from_matching"]


def bootcmatch_aggregates_from_matching(
    w: torch.Tensor, match_of: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Flat ``(row, col, value)`` triples for the BootCMatch tentative prolongator ``P``.

    Args:
        w (torch.Tensor): Smooth vector, shape ``(n,)``.
        match_of (torch.Tensor): Long tensor, shape ``(n,)``, from
            ``bootcmatch_parallel_matching``: ``match_of[v]`` is ``v``'s
            partner, or ``-1`` if ``v`` is a singleton. Must be symmetric
            (``match_of[match_of[v]] == v`` whenever matched).

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]: ``(row,
        col, value, n_coarse)``. ``row``/``col``/``value`` are 1-D, length
        ``n`` (exactly one triple per vertex/row, since every vertex -
        matched or singleton - gets exactly one nonzero in ``P``).
        ``n_coarse`` is the total aggregate count (pairs + singletons).

    An "aggregate start" is a singleton or the lower-indexed half of a
    pair; a cumulative count over aggregate starts assigns each aggregate
    its coarse column index. Every vertex then reads its own coarse column
    as its own start index if it is one, else its partner's start index -
    both fully vectorized gathers, no Python loop over vertices.
    """
    n = w.shape[0]
    device = w.device
    arange = torch.arange(n, device=device)

    matched = match_of >= 0
    is_start = (~matched) | (arange < match_of.clamp(min=0))

    coarse_index_of_start = torch.cumsum(is_start.to(torch.long), dim=0) - 1
    n_coarse = int(coarse_index_of_start[-1].item()) + 1 if n > 0 else 0

    partner_start_index = torch.where(
        matched,
        coarse_index_of_start[match_of.clamp(min=0)],
        torch.zeros_like(coarse_index_of_start),
    )
    col = torch.where(is_start, coarse_index_of_start, partner_start_index)

    w_partner = w[match_of.clamp(min=0)]
    pair_norm = torch.sqrt(w**2 + w_partner**2)
    pair_value = w / pair_norm
    singleton_value = torch.sign(w)
    value = torch.where(matched, pair_value, singleton_value)

    row = arange
    return row, col, value, n_coarse
