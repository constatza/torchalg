"""Algebraic-distance strength of connection for Bootstrap AMG.

Pure functions with no coarsening-strategy state, following the module style
of ``_aggregation.py``/``_compatible_relaxation.py``: given a set of test
vectors and the fine-grid matrix, build the pairwise algebraic-distance
measure ``r_ij`` and the pruned strength graph ``M_d`` that feeds Task 4's
``BAMGCoarsening`` in place of ``_compatible_relaxation.py``'s placeholder
matrix-graph independent-set guide.

``docs/bootstrap-amg.md`` Sec. 2.2 is the authoritative spec (transcribed
directly from the primary paper, not a secondary summary); every equation
number cited below refers to it.

References:
    - Brandt, A., Brannick, J., Kahl, K., & Livshits, I. (2011). An algebraic
      distances measure of AMG strength of connection. arXiv:1106.5990.
      Cited as [AD11]: eq. 4.2 (algebraic d-neighborhood ``V_i``), eq. 4.3
      (the caliber-one LS distance ``r_ij``), eq. 4.4 (the pruned strength
      graph ``M_d``), Remark 4.3 (computing ``V_i`` from ``A``'s sparsity
      alone, without forming ``A^d``'s values).
    - Brandt, A., Brannick, J., Kahl, K., & Livshits, I. (2011). Bootstrap
      AMG. SIAM J. Sci. Comput. 33(2), 612-632. Cited as [BAMG11]: eq. 4.1
      (test-vector weights ``omega_kappa``, ``T = I`` reduction used here
      since no composite-interpolation operator exists yet at this stage),
      eq. 2.3 (the residual-correction/"adaptive relaxation" step that eq.
      4.3 below applies to the target test vectors).

``r_ij`` is a **one-sided, directional** measure, not a distance in the
metric sense: [AD11] describes eq. 4.3 explicitly as "the simplified variant
of the algebraic distance notion of strength of connection based on
one-sided interpolation" (the paper's own wording, right after eq. 4.3).
``r_ij != r_ji`` in general - directional in the same spirit as classical
AMG's own strength test, ``-a_ij >= theta * max_{k!=i}(-a_ik)``
(``docs/boomeramg.md`` Sec. 2.1), which is likewise a per-row threshold and
not symmetric in ``i, j``. Nothing in [AD11] claims or requires
``r_ij == r_ji``; callers must not assume symmetry.
"""

from __future__ import annotations

import torch

_NEAR_ZERO_ENERGY_TOL = 1e-14
"""Test-vector energies with magnitude below this are treated as 1.0 when
computing weights, protecting against division by near-zero."""

_MIN_RESIDUAL = 1e-14
"""Floor applied to the caliber-one LS residual before inversion, protecting
against division by zero/negative floating-point noise on a near-perfect fit."""

_NEAR_ZERO_DIAGONAL_TOL = 1e-14
"""Diagonal entries with magnitude below this are treated as 1.0 in the
residual-correction step, protecting against division by near-zero."""


def _residual_corrected_vectors(test_vectors: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
    """Apply one local Jacobi correction to every test vector ([BAMG11] eq. 2.3).

    ``v_i^(kappa) <- v_i^(kappa) - (A v^(kappa))_i / a_ii`` - the "adaptive
    relaxation" step eq. 4.3 fits its LS regression to, rather than the raw
    test vectors. Assumes ``a_ii != 0`` (the paper's own precondition),
    guarded against near-zero diagonals the same way ``_aggregation.py``
    guards its diagonal normalizer.

    Args:
        test_vectors (torch.Tensor): Test vectors ``V``, shape ``(n, k)``.
        matrix (torch.Tensor): Dense matrix ``A``, shape ``(n, n)``.

    Returns:
        torch.Tensor: Residual-corrected test vectors, shape ``(n, k)``.
    """
    diagonal = torch.diagonal(matrix)
    diagonal_safe = torch.where(
        diagonal.abs() > _NEAR_ZERO_DIAGONAL_TOL, diagonal, torch.ones_like(diagonal)
    )
    return test_vectors - (matrix @ test_vectors) / diagonal_safe.unsqueeze(1)


def depth_neighborhood(matrix: torch.Tensor, depth: int) -> torch.Tensor:
    """Boolean off-diagonal adjacency of ``matrix`` raised to ``depth`` ([AD11] eq. 4.2/Remark 4.3).

    ``depth=1`` is ``matrix``'s own off-diagonal nonzero pattern. Deeper
    values are reached by repeated boolean matrix multiplication of that
    base pattern with itself - only ever propagating sparsity, never
    ``A``'s numeric values, per Remark 4.3. The diagonal is excluded at
    every depth (including depths ``> 1``, where an even-length walk can
    reintroduce ``i``-to-``i`` reachability): a node is never its own
    caliber-one interpolation neighbor.

    Args:
        matrix (torch.Tensor): Dense matrix ``A``, shape ``(n, n)``.
        depth (int): Search depth ``d``, ``>= 1``.

    Returns:
        torch.Tensor: Boolean neighborhood matrix, shape ``(n, n)``.
    """
    n = matrix.shape[0]
    off_diagonal = ~torch.eye(n, dtype=torch.bool, device=matrix.device)
    adjacency = (matrix != 0) & off_diagonal
    reachable = adjacency
    for _ in range(depth - 1):
        reachable = (reachable.to(torch.float32) @ adjacency.to(torch.float32)) > 0
    return reachable & off_diagonal


def test_vector_weights(test_vectors: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
    """Per-test-vector weights ``omega_kappa`` ([BAMG11] eq. 4.1, ``T = I`` reduction).

    ``omega_kappa = <v^(kappa), v^(kappa)> / <A v^(kappa), v^(kappa)>`` - the
    pure ``A``-energy weighting the paper falls back to before any composite
    interpolation operator ``T`` exists, which is always the case at the
    strength-of-connection stage.

    Args:
        test_vectors (torch.Tensor): Test vectors ``V``, shape ``(n, k)``.
        matrix (torch.Tensor): Dense matrix ``A``, shape ``(n, n)``.

    Returns:
        torch.Tensor: Weight vector, shape ``(k,)``.
    """
    energy = (test_vectors * (matrix @ test_vectors)).sum(dim=0)
    energy_safe = torch.where(energy.abs() > _NEAR_ZERO_ENERGY_TOL, energy, torch.ones_like(energy))
    return (test_vectors**2).sum(dim=0) / energy_safe


def algebraic_distance(
    test_vectors: torch.Tensor, matrix: torch.Tensor, depth: int = 1
) -> torch.Tensor:
    """Pairwise caliber-one algebraic distance ``r_ij`` ([AD11] eq. 4.3).

    For every edge ``(i, j)`` of ``matrix``'s depth-``d`` graph, fits the
    caliber-one weighted LS regression of the residual-corrected ``v_i``
    (``_residual_corrected_vectors``, [BAMG11] eq. 2.3) onto the raw ``v_j``
    (minimizer ``p_ij``) and returns the reciprocal of the weighted residual
    sum of squares: large ``r_ij`` means ``j`` predicts ``i`` with small LS
    error, i.e. a strong connection. This is a **one-sided/directional**
    measure - see the module docstring - so ``r_ij != r_ji`` in general;
    pairs outside the depth-``d`` neighborhood (including the diagonal) are
    zero.

    Vectorized over every ``(i, j)`` pair at once via the weighted-LS
    residual identity ``SSE_ij = T_i - p_ij * S_ij`` (with
    ``S_ij = sum_kappa omega_kappa v_tilde_i^(kappa) v_j^(kappa)`` the
    corrected-target/raw-predictor cross term and ``T_i = sum_kappa
    omega_kappa v_tilde_i^(kappa)^2`` the corrected target's own weighted
    energy), avoiding both a Python loop over pairs and an ``(n, n, k)``
    intermediate.

    Args:
        test_vectors (torch.Tensor): Test vectors ``V``, shape ``(n, k)``.
        matrix (torch.Tensor): Dense fine-grid matrix ``A``, shape ``(n, n)``.
        depth (int): Search depth ``d`` defining the neighborhood ``V_i``
            (default ``1``).

    Returns:
        torch.Tensor: Dense algebraic-distance matrix ``r``, shape
        ``(n, n)``, zero outside the depth-``d`` neighborhood. Not
        symmetric in general.
    """
    neighborhood = depth_neighborhood(matrix, depth)
    weights = test_vector_weights(test_vectors, matrix)
    corrected = _residual_corrected_vectors(test_vectors, matrix)

    cross = (corrected * weights) @ test_vectors.T
    predictor_energy = (test_vectors**2 * weights).sum(dim=1)
    coefficients = cross / predictor_energy.unsqueeze(0)
    target_energy = (corrected**2 * weights).sum(dim=1)
    residual = target_energy.unsqueeze(1) - coefficients * cross
    residual = torch.clamp(residual, min=_MIN_RESIDUAL)
    return torch.where(neighborhood, 1.0 / residual, torch.zeros_like(residual))


def strength_graph(
    distance: torch.Tensor, fine_mask: torch.Tensor, theta_ad: float = 0.5
) -> torch.Tensor:
    """Prune the algebraic-distance matrix into the strength graph ``M_d`` ([AD11] eq. 4.4).

    Entry ``(i, j)`` is kept iff both ``i`` and ``j`` are fine points
    (``fine_mask[i] == fine_mask[j] == True`` means ``i, j in F``, matching
    this module's own naming) and ``distance[i, j]`` exceeds ``theta_ad``
    times ``i``'s strongest connection to any node.

    Args:
        distance (torch.Tensor): Algebraic-distance matrix ``r``, shape
            ``(n, n)`` (as returned by ``algebraic_distance``).
        fine_mask (torch.Tensor): Boolean mask, shape ``(n,)``, ``True`` at
            ``F``-points.
        theta_ad (float): Strength threshold theta_ad in (0, 1) (paper
            default ``0.5``).

    Returns:
        torch.Tensor: Boolean strength graph, shape ``(n, n)``.
    """
    strongest = distance.max(dim=1).values
    strong = distance > theta_ad * strongest.unsqueeze(1)
    both_fine = fine_mask.unsqueeze(1) & fine_mask.unsqueeze(0)
    return strong & both_fine
