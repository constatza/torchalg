"""Weighted least-squares (LS/LSR) interpolation for Bootstrap AMG.

Pure functions with no coarsening-strategy state, following the module style
of ``_algebraic_distance.py``/``_compatible_relaxation.py``: given a set of
test vectors, fit the closed-form LS interpolation weights for a single fine
row (``ls_interpolation_row``), apply the residual-based "adaptive
relaxation" correction to test vectors before that fit (``lsr_correction``),
and grow a caliber-bounded interpolatory set greedily under the
algebraic-distance penalization rule (``select_interpolatory_set``).

``docs/bootstrap-amg.md`` Sec. 3 is the authoritative spec (transcribed
directly from the primary papers, not a secondary summary); every equation
number cited below refers to it.

References:
    - Brandt, A., Brannick, J., Kahl, K., & Livshits, I. (2011). Bootstrap
      AMG. SIAM J. Sci. Comput. 33(2), 612-632. Cited as [BAMG11]: eq. 2.1
      (the LS functional), eq. 2.2 and the closed-form minimizer stated
      immediately after it, Sec. 2.1's uniqueness condition
      ``rank(V_{C_i}) = |C_i|``, eq. 2.3 (the residual-correction/"adaptive
      relaxation" step, LSR), eq. 2.4 (the residual-based LS functional LSR
      fits).
    - Brandt, A., Brannick, J., Kahl, K., & Livshits, I. (2011). An algebraic
      distances measure of AMG strength of connection. arXiv:1106.5990.
      Cited as [AD11]: eq. 4.5 (the LS-ring candidate neighborhood ``C_i`` is
      drawn from, computed by ``BAMGCoarsening`` upstream of this module -
      see the ``candidates`` argument below), Sec. 4.3 (the caliber-growth
      penalization rule ``LS_{W''} < (LS_{W'})^{gamma(|W''|-|W'|)}``).

Real-SPD substitution: [BAMG11]'s formulas use Hermitian-transpose notation
(``V^H``) since the paper allows complex test vectors; torchalg is real-SPD
only throughout (no complex numbers anywhere in this codebase), so every
``V^H`` below is read as the plain transpose ``V^T`` - the same substitution
``adaptive.py``'s module docstring documents for its own PyAMG deviations.

``select_interpolatory_set``'s ``matrix`` parameter is accepted for
interface parity with Task 4's ``BAMGCoarsening`` calling convention (every
other per-row AMG kernel in this package takes ``matrix`` as context) but is
not read by the greedy criterion itself: [AD11] Sec. 4.3's ``LS_W`` is a
function of the test vectors and weights alone, not of ``matrix`` directly.
"""

from __future__ import annotations

import torch

_RANK_DEFICIENCY_RTOL = 1e-10
"""Relative tolerance for ``torch.linalg.matrix_rank`` when checking
[BAMG11] Sec. 2.1's uniqueness condition ``rank(V_{C_i}) = |C_i|`` on
``V_{C_i} W V_{C_i}^T``; below full rank, ``ls_interpolation_row`` falls back
to ``torch.linalg.lstsq`` instead of ``torch.linalg.solve``."""

_NEAR_ZERO_DIAGONAL_TOL = 1e-14
"""Diagonal entries with magnitude below this are treated as 1.0 in the
residual-correction step, protecting against division by near-zero (the
package-wide treat-as-1.0 diagonal-guard convention, shared with
``_aggregation.py``)."""


def ls_interpolation_row(
    test_vectors: torch.Tensor,
    target: int,
    interp_set: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Closed-form weighted-LS interpolation row ``p_i`` ([BAMG11] eq. 2.1-2.2).

    Solves ``p_i V_{C_i} W V_{C_i}^T = V_i W V_{C_i}^T`` for ``p_i`` (the
    normal equations of the weighted LS functional, eq. 2.1), returning the
    minimizer's transpose as a plain vector so ``p @ test_vectors[interp_set]``
    reconstructs the fitted approximation to ``test_vectors[target]``. Uses
    ``torch.linalg.solve`` on ``V_{C_i} W V_{C_i}^T`` when that Gram matrix is
    full rank; falls back to ``torch.linalg.lstsq`` when
    ``rank(V_{C_i} W V_{C_i}^T) < |interp_set|``, i.e. when [BAMG11] Sec.
    2.1's uniqueness condition ``rank(V_{C_i}) = |C_i|`` fails (typically
    ``|interp_set| > k``, or duplicate/collinear rows of ``V_{C_i}``).

    Args:
        test_vectors (torch.Tensor): Test vectors ``V``, shape ``(n, k)``.
        target (int): Fine row index ``i``.
        interp_set (torch.Tensor): Long tensor of interpolatory-set column
            indices ``C_i``, shape ``(|C_i|,)``.
        weights (torch.Tensor): Per-test-vector weights ``omega_kappa``,
            shape ``(k,)``.

    Returns:
        torch.Tensor: Interpolation row ``p_i``, shape ``(|C_i|,)``.
    """
    target_vector = test_vectors[target]
    restricted = test_vectors[interp_set]
    weighted = restricted * weights
    gram = weighted @ restricted.T
    rhs = weighted @ target_vector
    if torch.linalg.matrix_rank(gram, rtol=_RANK_DEFICIENCY_RTOL) < gram.shape[0]:
        return torch.linalg.lstsq(gram, rhs.unsqueeze(-1)).solution.squeeze(-1)
    return torch.linalg.solve(gram, rhs)


def lsr_correction(
    test_vectors: torch.Tensor,
    matrix: torch.Tensor,
    target_rows: torch.Tensor,
) -> torch.Tensor:
    """Residual-based "adaptive relaxation" correction ([BAMG11] eq. 2.3).

    ``v_i^(kappa) <- v_i^(kappa) - (A v^(kappa))_i / a_ii``, applied only at
    ``target_rows`` (the paper's practical schedule restricts this to a
    subset of TVs/points, [BAMG11] Sec. 4 - callers choose that subset;
    this function applies the correction to whichever rows they pass).
    Assumes ``a_ii != 0`` (the paper's own precondition), guarded against
    near-zero diagonals the same way ``_algebraic_distance.py`` guards its
    diagonal normalizer. Pure function: returns a new tensor, never mutates
    ``test_vectors``.

    Args:
        test_vectors (torch.Tensor): Test vectors ``V``, shape ``(n, k)``.
        matrix (torch.Tensor): Dense matrix ``A``, shape ``(n, n)``.
        target_rows (torch.Tensor): Long tensor of row indices to correct,
            shape ``(t,)``.

    Returns:
        torch.Tensor: New test vectors, shape ``(n, k)``, equal to
        ``test_vectors`` except at ``target_rows``.
    """
    diagonal = torch.diagonal(matrix)[target_rows]
    diagonal_safe = torch.where(
        diagonal.abs() > _NEAR_ZERO_DIAGONAL_TOL, diagonal, torch.ones_like(diagonal)
    )
    residual_at_targets = matrix[target_rows] @ test_vectors
    corrected = test_vectors.clone()
    corrected[target_rows] = test_vectors[
        target_rows
    ] - residual_at_targets / diagonal_safe.unsqueeze(1)
    return corrected


def _ls_residual(
    test_vectors: torch.Tensor,
    target: int,
    interp_set: torch.Tensor,
    weights: torch.Tensor,
) -> float:
    """Weighted LS functional value ``LS_{C_i}(p_i)`` at its minimizer ([BAMG11] eq. 2.1).

    Args:
        test_vectors (torch.Tensor): Test vectors ``V``, shape ``(n, k)``.
        target (int): Fine row index ``i``.
        interp_set (torch.Tensor): Long tensor of candidate-set indices,
            shape ``(|C_i|,)``.
        weights (torch.Tensor): Per-test-vector weights, shape ``(k,)``.

    Returns:
        float: ``sum_kappa omega_kappa (v_i^(kappa) - (p_i V_{C_i})^(kappa))^2``.
    """
    row = ls_interpolation_row(test_vectors, target, interp_set, weights)
    reconstructed = row @ test_vectors[interp_set]
    residual = test_vectors[target] - reconstructed
    return (weights * residual**2).sum().item()


def select_interpolatory_set(
    candidates: torch.Tensor,
    test_vectors: torch.Tensor,
    matrix: torch.Tensor,
    target: int,
    weights: torch.Tensor,
    caliber: int,
    gamma: float = 1.5,
) -> torch.Tensor:
    """Greedy caliber-bounded interpolatory-set selection ([AD11] Sec. 4.3/eq. 4.5).

    Grows ``C_i`` from the empty set: at each step, among the still-eligible
    ``candidates`` not yet chosen, picks the one whose addition minimizes the
    resulting LS functional value ``LS_{W''}(p_i)`` (``_ls_residual``, which
    delegates to ``ls_interpolation_row``). That addition is accepted only if
    it clears [AD11] Sec. 4.3's penalization test ``LS_{W''} <
    (LS_{W'})^{gamma * (|W''| - |W'|)}`` - here always ``gamma`` itself,
    since growth is always by exactly one candidate per step; otherwise
    growth stops early, before reaching ``caliber``. Because the accepted
    candidate is always the one minimizing ``LS_{W''}`` and the penalization
    threshold does not depend on which candidate is added, if the best
    candidate fails the test no other candidate would pass it either, so
    testing only the best candidate per step is sufficient.

    **Documented deviation: the penalization test is normalized.** [AD11]
    Sec. 4.3 states the rule on the raw functional value, which is only
    meaningful if ``LS`` is a dimensionless residual of order one: ``LS_W``
    scales as ``||V||^2``, so raising it to a power ``gamma > 1`` is not a
    scale-invariant operation. Callers whose test vectors have been driven
    to near-zero magnitude by many relaxation/bootstrap passes - exactly
    what ``BootstrapSetup.run`` does, since it relaxes on ``A x = 0``, whose
    exact solution is ``0`` - would otherwise see the threshold
    ``LS^gamma`` fall far below any attainable ``LS``, stalling growth
    immediately and frequently returning the empty set. Both sides are
    therefore divided by the empty-set functional value ``LS_0 = sum_kappa
    omega_kappa (v_i^(kappa))^2`` (the target row's own weighted energy,
    the loop's initial ``current_value``), captured once before the loop as
    a fixed normalization constant for this row's fit: the test becomes
    ``LS_{W''}/LS_0 < (LS_{W'}/LS_0)^gamma``, i.e. the identical rule
    applied to the *relative* LS residual. This is the same rule, made
    scale-invariant - not a different criterion. The guard on ``LS_0`` is a
    strict positivity test rather than the package's usual absolute
    ``1e-14`` near-zero tolerance, precisely because an absolute floor would
    reintroduce the scale dependence this normalization removes; ``LS_0`` is
    a sum of squares under positive weights, so it is zero only for an
    exactly-zero target row.

    ``candidates`` is expected to already be restricted to the LS-ring
    neighborhood ``N_{d_LS,i}`` ([AD11] eq. 4.5); computing that
    neighborhood (depth ``d_LS = d + 2`` graph connectivity from ``matrix``)
    is ``BAMGCoarsening``'s responsibility, not this function's.

    Args:
        candidates (torch.Tensor): Long tensor of eligible ``C``-indices
            within the LS-ring neighborhood, shape ``(m,)``.
        test_vectors (torch.Tensor): Test vectors ``V``, shape ``(n, k)``.
        matrix (torch.Tensor): Dense matrix ``A``, shape ``(n, n)``; accepted
            for interface parity only - see the module docstring.
        target (int): Fine row index ``i``.
        weights (torch.Tensor): Per-test-vector weights, shape ``(k,)``.
        caliber (int): Maximum interpolatory-set size ``c``.
        gamma (float): Penalization exponent (paper default ``1.5``).

    Returns:
        torch.Tensor: Long tensor of chosen interpolatory-set indices
        ``C_i``, shape ``(<= caliber,)``, ready for ``ls_interpolation_row``.
    """
    remaining = candidates.tolist()
    chosen: list[int] = []
    empty_set_value = (weights * test_vectors[target] ** 2).sum().item()
    normalizer = empty_set_value if empty_set_value > 0.0 else 1.0
    current_value = empty_set_value / normalizer
    while remaining and len(chosen) < caliber:
        trial_values = {
            candidate: _ls_residual(
                test_vectors,
                target,
                torch.tensor(chosen + [candidate], dtype=torch.long, device=test_vectors.device),
                weights,
            )
            for candidate in remaining
        }
        best_candidate = min(trial_values, key=lambda candidate: trial_values[candidate])
        best_value = trial_values[best_candidate] / normalizer
        if not best_value < current_value**gamma:
            break
        chosen.append(best_candidate)
        remaining.remove(best_candidate)
        current_value = best_value
    return torch.tensor(chosen, dtype=torch.long, device=test_vectors.device)
