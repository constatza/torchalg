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
    - Brandt, A., Brannick, J., Kahl, K., & Livshits, I. (2015). Bootstrap
      algebraic multigrid: status report, open problems, and outlook. Numer.
      Math. Theor. Meth. Appl. 8(1). arXiv:1406.1819. Cited as [STATUS14]:
      eq. 3.1 (the LS functional, the closed-form minimizer stated
      immediately after it, and Sec. 3's uniqueness condition
      ``rank(V_{C_i}) = |C_i|``), eq. 3.2 (the residual-correction/"adaptive
      relaxation" step, LSR), Table 1's caption (the practical 20%/
      largest-residual LSR schedule - see ``bootstrap.py``'s
      ``_LSR_TARGET_FRACTION``).
    - Brandt, A., Brannick, J., Kahl, K., & Livshits, I. (2011). An algebraic
      distances measure of AMG strength of connection. arXiv:1106.5990.
      Cited as [AD11]: eq. 4.5 (the LS-ring candidate neighborhood ``C_i`` is
      drawn from, computed by ``BAMGCoarsening`` upstream of this module -
      see the ``candidates`` argument below), Sec. 4.3 (the caliber-growth
      penalization rule ``LS_{W''} < (LS_{W'})^{gamma(|W''|-|W'|)}``).
    - Brandt, A., Brannick, J., Kahl, K., & Livshits, I. (2011). Bootstrap
      AMG. SIAM J. Sci. Comput. 33(2), 612-632. Cited as [BAMG11] where a
      prior read of this module attributed "eq. 2.1-2.4"/"Sec. 4" to the
      same material: the *content* matches what [STATUS14] confirms above,
      but [BAMG11] could not be fetched this session, so its own section/
      equation numbering (plausibly its own Sec. 2, differently numbered
      from [STATUS14]'s restatement in Sec. 3) is unconfirmed. See
      ``docs/bootstrap-amg.md`` Sec. 9.

Real-SPD substitution: [BAMG11]'s formulas use Hermitian-transpose notation
(``V^H``) since the paper allows complex test vectors; torchalg is real-SPD
only throughout (no complex numbers anywhere in this codebase), so every
``V^H`` below is read as the plain transpose ``V^T`` - the same substitution
``adaptive.py``'s module docstring documents for its own PyAMG deviations.

``ls_interpolation_row``/``select_interpolatory_set`` have since been
promoted to ``torchalg.utils.ls_interpolation`` (re-exported here for every
existing ``from ...amg._least_squares import ...`` call site) - both are
already fully format-agnostic: they operate only on dense ``(n, k)``
test-vector row-gathers and small dense Gram matrices built from those
gathers, and ``select_interpolatory_set``'s own ``matrix`` parameter is
accepted for call-signature parity only, never read by the greedy criterion
itself (see that module's docstring) - the same "one legitimate shared edge"
precedent as ``torchalg.utils.spectral``'s ``approximate_spectral_radius``.
``lsr_correction`` remains dense-only here, since it genuinely reads
``matrix``; its sparse sibling lives in ``torchalg.sparse.kernels``.
"""

from __future__ import annotations

import torch

from torchalg.utils.ls_interpolation import (
    batched_ls_interpolation_rows,
    batched_select_interpolatory_set,
    ls_interpolation_row,
    select_interpolatory_set,
)

__all__ = [
    "batched_ls_interpolation_rows",
    "batched_select_interpolatory_set",
    "ls_interpolation_row",
    "lsr_correction",
    "select_interpolatory_set",
]

_NEAR_ZERO_DIAGONAL_TOL = 1e-14
"""Diagonal entries with magnitude below this are treated as 1.0 in the
residual-correction step, protecting against division by near-zero (the
package-wide treat-as-1.0 diagonal-guard convention, shared with
``_aggregation.py``)."""


def lsr_correction(
    test_vectors: torch.Tensor,
    matrix: torch.Tensor,
    target_rows: torch.Tensor,
) -> torch.Tensor:
    """Residual-based "adaptive relaxation" correction ([STATUS14] eq. 3.2).

    ``v_i^(kappa) <- v_i^(kappa) - (A v^(kappa))_i / a_ii``, applied only at
    ``target_rows`` (the paper's practical schedule restricts this to a
    subset of TVs/points, [STATUS14] Table 1's caption - callers choose that subset;
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
