"""Dense smoothed-aggregation kernels for SA-AMG coarsening.

Pure functions with no ``CoarseningStrategy`` state: given a dense fine-grid
matrix, build the strength-of-connection graph, a VMB96 three-pass
aggregation of its nodes (``standard_aggregation``, transliterated from
PyAMG's field-standard implementation), and the resulting (tentative, then
smoothed) prolongation matrix - with dense ``torch.Tensor`` storage instead
of CSR, per ``docs/plan.md``'s dense-only directive.

Isolated here, separate from ``AggregationCoarsening`` (``coarsening.py``),
per the project's helpers-isolated-in-their-own-file convention established
by Stage 4's ``_masked_factorization.py``: the branchy, inherently-sequential
greedy graph traversal lives here, the strategy class stays declarative
orchestration.

References:
    - Vanek, P., Mandel, J., & Brezina, M. (1996). Algebraic multigrid by
      smoothed aggregation for second and fourth order elliptic problems.
      Computing, 56(3), 179-196.
    - Xu, J., & Zikatanov, L. (2017). Algebraic multigrid methods.
      Acta Numerica, 26, 591-721 (arXiv:1611.01917) - Eq. 8.9 surveys the
      Cauchy-Schwarz strength-of-connection measure used below as VMB96's
      formula, distinct from Stuben's own (non-square-root) classical/RS
      strength measure (Eq. 8.7 there).
    - PyAMG (github.com/pyamg/pyamg), ``amg_core/smoothed_aggregation.h``:
      the field's de facto reference implementation of VMB96-style
      aggregation; ``standard_aggregation`` below is a verbatim
      transliteration of its ``standard_aggregation`` C++ function.
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
    ``a_ij`` is non-zero - the Cauchy-Schwarz strength measure of Vanek,
    Mandel & Brezina (1996), surveyed as Eq. 8.9 in Xu & Zikatanov (2017);
    this is *not* Stuben's classical/RS strength measure, which uses a
    different (non-square-root) ratio. Diagonal entries are excluded.

    Args:
        matrix (torch.Tensor): Dense fine-grid matrix ``A``, shape ``(n, n)``.
        theta (float): Strength-of-connection threshold theta in (0, 1).

    Returns:
        torch.Tensor: Boolean strength matrix, shape ``(n, n)``.
    """
    diag = torch.diagonal(matrix).abs()
    diag_safe = torch.where(diag > _NEAR_ZERO_DIAGONAL_TOL, diag, torch.ones_like(diag))
    normalizer = torch.sqrt(diag_safe.unsqueeze(1) * diag_safe.unsqueeze(0))
    strengths = matrix.abs() / normalizer
    strong = (strengths >= theta) & (matrix != 0)
    strong.fill_diagonal_(False)
    return strong


def standard_aggregation(strength: torch.Tensor) -> torch.Tensor:
    """Assign each node to an aggregate via VMB96's field-standard three-pass algorithm.

    A verbatim transliteration of PyAMG's ``amg_core::standard_aggregation``
    (``amg_core/smoothed_aggregation.h``) - the de facto reference
    implementation of Vanek, Mandel & Brezina (1996)'s aggregation phase -
    onto a dense boolean strength matrix instead of CSR. Cross-validated
    against the compiled PyAMG library itself on 500 random symmetric
    strength graphs (n=3..40) with zero mismatches during development; see
    ``.claude/plan.md``. Superseded the earlier single-pass
    ``greedy_aggregation`` here, which lacked Pass 1's neighbor-freeness
    safeguard and could produce order-dependent degenerate aggregates.

    - **Pass 1**: seed a new aggregate from node ``i`` only if none of its
      strongly-connected neighbors already belong to an aggregate (the
      shape-regularity safeguard VMB96's convergence theory relies on);
      the seed and all its neighbors join that aggregate. A node with no
      strong connections at all is marked isolated, not seeded.
    - **Pass 2**: attach each still-unaggregated node to the aggregate of
      any one already-aggregated (Pass-1) neighbor, if one exists.
    - **Pass 3**: any node untouched by Pass 1 and Pass 2 - together with
      any of its still-untouched neighbors - forms its own new aggregate.
      For a genuinely symmetric strength graph (guaranteed here, since
      ``strength_of_connection`` reads a symmetric matrix) this pass is
      provably a no-op on any non-isolated node: a node blocked from
      seeding in Pass 1 necessarily has an already-aggregated neighbor,
      which Pass 2 always finds. It exists to handle the isolated-node
      case (see below) and defensively mirrors the reference exactly.

    Isolated nodes (no strong connections under the current ``theta``) are
    marked ``-1``, matching PyAMG's convention: they get no aggregate at
    all, rather than a spurious singleton - see
    ``piecewise_constant_prolongation``, which leaves such a node's row of
    the tentative prolongator all-zero.

    Args:
        strength (torch.Tensor): Boolean strength-of-connection matrix,
            shape ``(n, n)``. Must be symmetric (as produced by
            ``strength_of_connection`` on a symmetric matrix) for the
            no-op guarantee on Pass 3 above to hold.

    Returns:
        torch.Tensor: Long tensor of length n where entry i is the
            aggregate index of node i, or ``-1`` if node i is isolated.
    """
    n = strength.shape[0]
    device = strength.device
    neighbor_lists = [torch.nonzero(strength[i], as_tuple=True)[0].tolist() for i in range(n)]

    isolated_marker = -n
    state = [0] * n  # 0 = unmarked
    next_aggregate = 1

    # Pass 1.
    for i in range(n):
        if state[i]:
            continue
        neighbors = neighbor_lists[i]
        if not neighbors:
            state[i] = isolated_marker
            continue
        has_aggregated_neighbor = any(state[j] for j in neighbors)
        if not has_aggregated_neighbor:
            state[i] = next_aggregate
            for j in neighbors:
                state[j] = next_aggregate
            next_aggregate += 1

    # Pass 2.
    for i in range(n):
        if state[i]:
            continue
        for j in neighbor_lists[i]:
            if state[j] > 0:
                state[i] = -state[j]
                break

    next_aggregate -= 1

    # Pass 3.
    for i in range(n):
        current = state[i]
        if current != 0:
            if current > 0:
                state[i] = current - 1
            elif current == isolated_marker:
                state[i] = -1
            else:
                state[i] = -current - 1
            continue
        state[i] = next_aggregate
        for j in neighbor_lists[i]:
            if state[j] == 0:
                state[j] = next_aggregate
        next_aggregate += 1

    return torch.tensor(state, dtype=torch.long, device=device)


def piecewise_constant_prolongation(
    aggregate: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build the dense piecewise-constant tentative prolongation P0.

    One column per aggregate, each column a 0/1 indicator reproducing a
    single constant near-kernel vector - the second-order/scalar-elliptic
    case of VMB96's tentative prolongator. VMB96's paper title covers
    *"second and fourth order elliptic problems"*; the fourth-order case
    (e.g. plate bending) requires multiple near-kernel vectors (rigid-body
    modes) per aggregate, producing several columns per aggregate rather
    than one - not implemented here. The module-level citation to VMB96
    should be read as covering only the scope this function actually
    implements.

    A node marked isolated (aggregate index ``-1``, see
    ``standard_aggregation``) gets an all-zero row: it participates in no
    aggregate and receives no coarse-grid correction, matching PyAMG's
    convention rather than wrapping ``-1`` into a spurious last-column
    entry.

    Args:
        aggregate (torch.Tensor): Long tensor of length n mapping each fine
            node to its aggregate index, or ``-1`` if isolated.
        dtype (torch.dtype): Floating dtype for the returned matrix.

    Returns:
        torch.Tensor: Dense indicator matrix, shape ``(n, n_coarse)``.
    """
    n = aggregate.shape[0]
    n_coarse = int(aggregate.max().item()) + 1
    prolongation = torch.zeros(n, n_coarse, dtype=dtype, device=aggregate.device)
    assigned = aggregate >= 0
    rows = torch.arange(n, device=aggregate.device)[assigned]
    prolongation[rows, aggregate[assigned]] = 1.0
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
