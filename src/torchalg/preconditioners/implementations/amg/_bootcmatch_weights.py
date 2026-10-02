"""BootCMatch edge-weight kernel for graph-weighted-matching coarsening (Bootstrap AMG).

Implements eq. 10 of D'Ambra, Filippone, Vassilevski, "BootCMatch: a
Software Package for Bootstrap AMG Based on Graph Weighted Matching" (ACM
TOMS 44(4), 2018):

    a_hat_ij = 1 - 2 * a_ij * w_i * w_j / (a_ii * w_i**2 + a_jj * w_j**2)

computed once per stored (nonzero) edge ``(i, j)`` of the fine-grid matrix
``A``, given the current "smooth vector" ``w`` (a plain ``(n,)`` tensor as
far as this kernel is concerned - no special type). ``O(nnz)``, fully
vectorized: gather ``a_ii``, ``a_jj``, ``a_ij``, ``w_i``, ``w_j`` for every
edge and compute pointwise.

Eq. 6's log-transform of this weight is deliberately **not** implemented
here - it is the paper's own proof technique for the matching argument, not
part of the matching computation itself (``argmax`` is invariant to a
monotonic log-transform), so it is out of scope for this kernel.

Edge existence follows this file's own dense AMG convention, matching
``_graph.depth_neighborhood``'s adjacency definition exactly: ``matrix !=
0`` restricted to off-diagonal positions (the diagonal is never a
"matching" edge).

References:
    - D'Ambra, P., Filippone, S., & Vassilevski, P. S. (2018). BootCMatch: A
      Software Package for Bootstrap AMG Based on Graph Weighted Matching.
      ACM Transactions on Mathematical Software, 44(4), eq. 10 (edge
      weight), and its own stated degeneracy guards (TOL = machine epsilon):
      a degenerate edge ``sqrt(w_i**2/a_ii + w_j**2/a_jj) < TOL`` is excluded
      from matching entirely, and a vertex with ``|w_i| < TOL`` is a
      "fine-only" vertex never offered as a matching candidate.
"""

from __future__ import annotations

import torch

__all__ = ["bootcmatch_edge_weights"]


def bootcmatch_edge_weights(
    w: torch.Tensor, matrix: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-edge BootCMatch weights ``a_hat_ij`` ([BootCMatch18] eq. 10), dense.

    Args:
        w (torch.Tensor): Smooth vector, shape ``(n,)``.
        matrix (torch.Tensor): Dense fine-grid matrix ``A``, shape ``(n, n)``.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: ``(weights, fine_only_mask)``.
        ``weights`` has shape ``(n, n)``, equal to ``a_hat_ij`` at every
        off-diagonal stored (nonzero) edge of ``matrix``, and ``-inf`` at
        every non-edge position (including the diagonal) and at every
        degenerate edge - so a plain ``argmax`` over a row never selects a
        non-edge or a degenerate edge. ``fine_only_mask`` has shape ``(n,)``,
        ``True`` where ``|w_i| < TOL``.
    """
    n = matrix.shape[0]
    tol = torch.finfo(matrix.dtype).eps

    off_diagonal = ~torch.eye(n, dtype=torch.bool, device=matrix.device)
    is_edge = (matrix != 0) & off_diagonal

    diag = torch.diagonal(matrix)
    diag_i = diag.unsqueeze(1)
    diag_j = diag.unsqueeze(0)
    w_i = w.unsqueeze(1)
    w_j = w.unsqueeze(0)

    denominator = diag_i * w_i**2 + diag_j * w_j**2
    weights = 1.0 - 2.0 * matrix * w_i * w_j / denominator

    degenerate = torch.sqrt(w_i**2 / diag_i + w_j**2 / diag_j) < tol

    keep = is_edge & ~degenerate
    weights = torch.where(keep, weights, torch.full_like(weights, float("-inf")))

    fine_only_mask = w.abs() < tol

    return weights, fine_only_mask
