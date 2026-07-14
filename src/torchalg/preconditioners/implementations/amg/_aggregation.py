"""Dense smoothed-aggregation kernels for SA-AMG coarsening.

Pure functions with no ``CoarseningStrategy`` state: given a dense fine-grid
matrix, build the strength-of-connection graph, a greedy aggregation of its
nodes, and the resulting (tentative, then smoothed) prolongation matrix -
mirroring the reference's ``scipy.sparse``-backed SA-AMG kernels
(Vanek, Mandel & Brezina 1996) with dense ``torch.Tensor`` storage instead of
CSR, per ``docs/plan.md``'s dense-only directive.

Isolated here, separate from ``AggregationCoarsening`` (``coarsening.py``),
per the project's helpers-isolated-in-their-own-file convention established
by Stage 4's ``_masked_factorization.py``: the branchy, inherently-sequential
greedy graph traversal lives here, the strategy class stays declarative
orchestration.

References:
    - Vanek, P., Mandel, J., & Brezina, M. (1996). Algebraic multigrid by
      smoothed aggregation for second and fourth order elliptic problems.
      Computing, 56(3), 179-196.
    - Stuben, K. (2001). A review of algebraic multigrid.
      J. Comput. Appl. Math., 128(1-2), 281-309.
"""

from __future__ import annotations

import torch

_NEAR_ZERO_DIAGONAL_TOL = 1e-14
"""Diagonal entries with magnitude below this are treated as 1.0 when
normalizing strength-of-connection ratios (protects against division by
near-zero)."""


def strength_of_connection(matrix: torch.Tensor, theta: float) -> torch.Tensor:
    """Build a binary strength-of-connection matrix.

    Entry (i, j) is strong if ``|a_ij| >= theta * sqrt(|a_ii| * |a_jj|)`` and
    ``a_ij`` is non-zero (Stuben 2001, Section 2.1). Diagonal entries are
    excluded.

    Args:
        matrix (torch.Tensor): Dense fine-grid matrix ``A``, shape ``(n, n)``.
        theta (float): Strength-of-connection threshold theta in (0, 1).

    Returns:
        torch.Tensor: Boolean strength matrix, shape ``(n, n)``.
    """
    diag = torch.diagonal(matrix).abs()
    diag_safe = torch.where(diag > 0, diag, torch.ones_like(diag))
    normalizer = torch.sqrt(diag_safe.unsqueeze(1) * diag_safe.unsqueeze(0))
    strengths = matrix.abs() / normalizer
    strong = (strengths >= theta) & (matrix != 0)
    strong.fill_diagonal_(False)
    return strong


def greedy_aggregation(strength: torch.Tensor) -> torch.Tensor:
    """Assign each node to an aggregate via greedy graph traversal.

    Args:
        strength (torch.Tensor): Boolean strength-of-connection matrix,
            shape ``(n, n)``.

    Returns:
        torch.Tensor: Long tensor of length n where entry i is the aggregate
            index of node i.
    """
    n = strength.shape[0]
    aggregate = torch.full((n,), -1, dtype=torch.long, device=strength.device)
    n_aggregates = 0
    for i in range(n):
        if aggregate[i] >= 0:
            continue
        aggregate[i] = n_aggregates
        neighbors = torch.nonzero(strength[i], as_tuple=True)[0]
        for j in neighbors.tolist():
            if aggregate[j] < 0:
                aggregate[j] = n_aggregates
        n_aggregates += 1
    # Any isolated nodes get their own aggregate (defensive - unreachable in
    # practice since every index is visited by the loop above, kept for
    # fidelity with the reference).
    for i in range(n):
        if aggregate[i] < 0:
            aggregate[i] = n_aggregates
            n_aggregates += 1
    return aggregate


def piecewise_constant_prolongation(
    aggregate: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build the dense piecewise-constant tentative prolongation P0.

    Args:
        aggregate (torch.Tensor): Long tensor of length n mapping each fine
            node to its aggregate index.
        dtype (torch.dtype): Floating dtype for the returned matrix.

    Returns:
        torch.Tensor: Dense indicator matrix, shape ``(n, n_coarse)``.
    """
    n = aggregate.shape[0]
    n_coarse = int(aggregate.max().item()) + 1
    prolongation = torch.zeros(n, n_coarse, dtype=dtype, device=aggregate.device)
    prolongation[torch.arange(n, device=aggregate.device), aggregate] = 1.0
    return prolongation


def smoothed_prolongation(
    matrix: torch.Tensor,
    tentative: torch.Tensor,
    omega: float,
) -> torch.Tensor:
    """Apply one Jacobi smoothing step to the tentative prolongation P0.

    Computes ``P = (I - omega * D^{-1} A) P0`` (Vanek et al. 1996, Eq. 3.2).
    The damping factor omega should equal ``4 / (3 * rho(D^{-1}A))``. For
    isotropic SPD problems ``rho(D^{-1}A) ~= 2``, giving ``omega ~= 2/3`` -
    the fixed default used by ``AggregationCoarsening``. Anisotropic
    problems may need omega computed via power iteration on ``D^{-1}A``.

    Args:
        matrix (torch.Tensor): Fine-grid matrix ``A``, shape ``(n, n)``.
        tentative (torch.Tensor): Piecewise-constant tentative prolongation
            P0, shape ``(n, n_coarse)``.
        omega (float): Jacobi damping factor omega.

    Returns:
        torch.Tensor: Smoothed prolongation matrix ``P = (I - omega D^{-1}
            A) P0``, shape ``(n, n_coarse)``.
    """
    diag = torch.diagonal(matrix)
    diag_safe = torch.where(
        diag.abs() > _NEAR_ZERO_DIAGONAL_TOL,
        diag,
        torch.ones_like(diag),
    )
    return tentative - omega * ((matrix @ tentative) / diag_safe.unsqueeze(1))
