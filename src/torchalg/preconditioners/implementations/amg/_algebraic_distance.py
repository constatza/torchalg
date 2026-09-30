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
      Cited as [AD11] (all four directly confirmed against
      ``docs/bamg/ad11_raw.md``): eq. 4.2 (algebraic d-neighborhood
      ``V_i := {j : (A^d)_ij != 0}`` - defined via the nonzero *pattern* of
      ``A^d``, not its values, which is what ``depth_neighborhood`` below
      computes), eq. 4.3 (the caliber-one LS distance ``r_ij``, whose own
      formula already embeds the residual-corrected target ``v_i^(kappa) -
      (1/a_ii) r_i^(kappa)`` inline), eq. 4.4 (the pruned strength graph
      ``M_d``). A prior read of this module additionally cited "Remark 4.3"
      for the sparsity-only claim above: **no Remark 4.3 exists in this
      paper** (its Sec. 4 has only Remarks 4.1, 4.2 and 4.4, none of which
      make this claim) - that citation was fabricated or misremembered and
      is dropped; eq. 4.2's own definition already supports the claim
      without needing a remark.
    - Brandt, A., Brannick, J., Kahl, K., & Livshits, I. (2015). Bootstrap
      algebraic multigrid: status report, open problems, and outlook. Numer.
      Math. Theor. Meth. Appl. 8(1). arXiv:1406.1819. Cited as [STATUS14]:
      eq. 3.2 (the residual-correction/"adaptive relaxation" step that eq.
      4.3 below applies to the target test vectors - implemented once, in
      ``_least_squares.lsr_correction``, which this module delegates to).
      Test-vector weights ``omega_kappa`` with the ``T = I`` reduction used
      here (no composite-interpolation operator exists yet at this stage):
      **not confirmed** against [STATUS14], which describes ``omega_kappa``
      only via ``||v||_A^2`` with no ``T`` term at all - see
      ``test_vector_weights``'s own TODO below and ``_mge.py``'s module
      docstring. A prior read of this module attributed this formula to
      "[BAMG11] eq. 4.1"; [BAMG11] could not be fetched this session to
      confirm that attribution either. See ``docs/bootstrap-amg.md`` Sec. 9.

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

from ._graph import depth_neighborhood
from ._least_squares import lsr_correction

_NEAR_ZERO_ENERGY_TOL = 1e-14
"""Test-vector energies with magnitude below this are treated as 1.0 when
computing weights, protecting against division by near-zero."""

_MIN_RESIDUAL_RTOL = 1e-14
"""Floor applied to the caliber-one LS residual before inversion, *relative*
to the target's own weighted energy ``T_i``, protecting against division by
zero/negative floating-point noise on a near-perfect fit. Relative rather
than absolute: ``T_i`` scales as ``||V||^2``, so an absolute floor saturates
on ordinary inputs (measured: 2 of 60 neighborhood edges hit a ``1e-14``
absolute floor exactly on N=31 with unrelaxed random test vectors), which
then corrupts ``strength_graph``'s per-row ``theta_ad * max`` normalization
by clamping unrelated edges to one common value."""


def _residual_corrected_vectors(test_vectors: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
    """Apply one local Jacobi correction to every test vector ([STATUS14] eq. 3.2).

    ``v_i^(kappa) <- v_i^(kappa) - (A v^(kappa))_i / a_ii`` - the "adaptive
    relaxation" step eq. 4.3 fits its LS regression to, rather than the raw
    test vectors. This is exactly ``_least_squares.lsr_correction`` applied
    to *every* row, so it delegates there rather than reimplementing eq. 3.2
    (including its near-zero-diagonal guard) a second time.

    Args:
        test_vectors (torch.Tensor): Test vectors ``V``, shape ``(n, k)``.
        matrix (torch.Tensor): Dense matrix ``A``, shape ``(n, n)``.

    Returns:
        torch.Tensor: Residual-corrected test vectors, shape ``(n, k)``.
    """
    all_rows = torch.arange(matrix.shape[0], device=matrix.device)
    return lsr_correction(test_vectors, matrix, all_rows)


def test_vector_weights(
    test_vectors: torch.Tensor, matrix: torch.Tensor, T: torch.Tensor | None = None
) -> torch.Tensor:
    """Per-test-vector weights ``omega_kappa``.

    ``omega_kappa = <T v^(kappa), v^(kappa)> / <A v^(kappa), v^(kappa)>``.
    ``T`` is the composite-interpolation Gram operator ``P_l^H P_l``
    (``docs/bootstrap-amg.md`` Sec. 4.2); ``T = None`` uses the ``T = I``
    reduction to a pure ``A``-energy weighting, valid "on the finest level,
    or before any MGE enrichment" (line 234) - i.e. whenever no composite
    interpolation has been built yet, which is every level when MGE
    (``_mge.py``, ``k_e``) is disabled, and the finest level regardless.
    **TODO(bamg-fidelity, needs-check):** the ``T``-weighted generalization
    used here for ``k_e > 0`` is not confirmed against [STATUS14], which
    states ``omega_kappa`` only via ``||v||_A^2`` (no ``T`` term); see
    ``_mge.py``'s module docstring for the full status of this gap.

    Args:
        test_vectors (torch.Tensor): Test vectors ``V``, shape ``(n, k)``.
        matrix (torch.Tensor): Dense matrix ``A``, shape ``(n, n)``.
        T (torch.Tensor | None): Composite-interpolation Gram operator,
            shape ``(n, n)``; ``None`` for the ``T = I`` reduction.

    Returns:
        torch.Tensor: Weight vector, shape ``(k,)``.
    """
    energy = (test_vectors * (matrix @ test_vectors)).sum(dim=0)
    energy_safe = torch.where(energy.abs() > _NEAR_ZERO_ENERGY_TOL, energy, torch.ones_like(energy))
    numerator = (
        (test_vectors**2).sum(dim=0)
        if T is None
        else (test_vectors * (T @ test_vectors)).sum(dim=0)
    )
    return numerator / energy_safe


def algebraic_distance(
    test_vectors: torch.Tensor, matrix: torch.Tensor, depth: int = 1, T: torch.Tensor | None = None
) -> torch.Tensor:
    """Pairwise caliber-one algebraic distance ``r_ij`` ([AD11] eq. 4.3).

    For every edge ``(i, j)`` of ``matrix``'s depth-``d`` graph, fits the
    caliber-one weighted LS regression of the residual-corrected ``v_i``
    (``_residual_corrected_vectors``, [STATUS14] eq. 3.2) onto the raw ``v_j``
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
    energy - unrelated to this function's own ``T`` parameter below despite
    the shared letter: ``T_i`` here is a per-row scalar intermediate of this
    SSE decomposition, while ``T`` is the composite-interpolation Gram
    *operator* that only enters through ``omega_kappa`` itself), avoiding
    both a Python loop over pairs and an ``(n, n, k)`` intermediate.

    Args:
        test_vectors (torch.Tensor): Test vectors ``V``, shape ``(n, k)``.
        matrix (torch.Tensor): Dense fine-grid matrix ``A``, shape ``(n, n)``.
        depth (int): Search depth ``d`` defining the neighborhood ``V_i``
            (default ``1``).
        T (torch.Tensor | None): Composite-interpolation Gram operator
            passed through to ``test_vector_weights``, shape ``(n, n)``;
            ``None`` for the ``T = I`` reduction (see that function's
            docstring).

    Returns:
        torch.Tensor: Dense algebraic-distance matrix ``r``, shape
        ``(n, n)``, zero outside the depth-``d`` neighborhood. Not
        symmetric in general.
    """
    neighborhood = depth_neighborhood(matrix, depth)
    weights = test_vector_weights(test_vectors, matrix, T=T)
    corrected = _residual_corrected_vectors(test_vectors, matrix)

    cross = (corrected * weights) @ test_vectors.T
    predictor_energy = (test_vectors**2 * weights).sum(dim=1)
    predictor_safe = torch.where(
        predictor_energy.abs() > _NEAR_ZERO_ENERGY_TOL,
        predictor_energy,
        torch.ones_like(predictor_energy),
    )
    coefficients = cross / predictor_safe.unsqueeze(0)
    target_energy = (corrected**2 * weights).sum(dim=1)
    residual = target_energy.unsqueeze(1) - coefficients * cross
    floor = (_MIN_RESIDUAL_RTOL * target_energy).clamp(min=torch.finfo(residual.dtype).tiny)
    residual = torch.maximum(residual, floor.unsqueeze(1))
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
