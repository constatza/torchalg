"""Weighted least-squares (LS) interpolation for Bootstrap AMG, shared by the dense and sparse trees.

Promoted out of ``preconditioners.implementations.amg._least_squares`` (see
``docs/plan.md``'s "Correction: dense and sparse must be separate
implementations" - same "one legitimate shared edge" precedent as
``torchalg.utils.spectral``'s ``approximate_spectral_radius``): every
function here operates only on dense ``(n, k)`` test-vector row-gathers and
small dense ``(|C_i|, |C_i|)`` Gram matrices built from those gathers -
``test_vectors`` is always dense regardless of the system matrix's format
(the same way a POD basis is always dense), and ``matrix`` itself is never
read by the actual computation (``select_interpolatory_set`` accepts it only
for call-signature parity with other per-row AMG kernels - see its own
docstring). So none of this needs a sparse sibling; both the dense and
sparse ``BAMGCoarsening`` import the same implementation.

``docs/bootstrap-amg.md`` Sec. 3 is the authoritative spec (transcribed
directly from the primary papers, not a secondary summary); every equation
number cited below refers to it.

References:
    - Brandt, A., Brannick, J., Kahl, K., & Livshits, I. (2015). Bootstrap
      algebraic multigrid: status report, open problems, and outlook. Numer.
      Math. Theor. Meth. Appl. 8(1). arXiv:1406.1819. Cited as [STATUS14]:
      eq. 3.1 (the LS functional, the closed-form minimizer stated
      immediately after it, and Sec. 3's uniqueness condition
      ``rank(V_{C_i}) = |C_i|``), Table 1's caption (the practical 20%/
      largest-residual LSR schedule).
    - Brandt, A., Brannick, J., Kahl, K., & Livshits, I. (2011). An algebraic
      distances measure of AMG strength of connection. arXiv:1106.5990.
      Cited as [AD11]: eq. 4.5 (the LS-ring candidate neighborhood ``C_i`` is
      drawn from, computed by ``BAMGCoarsening`` upstream of this module),
      Sec. 4.3 (the caliber-growth penalization rule ``LS_{W''} <
      (LS_{W'})^{gamma(|W''|-|W'|)}``).

Selection without per-row solves: the weighted LS residual of a candidate
set depends only on the span of its whitened columns ``sqrt(omega) * v_j``,
so it is unique even when the Gram matrix is singular. Growing the set one
candidate at a time therefore needs no Gram solve at all; it is orthogonal
matching pursuit (a Gram-Schmidt sweep). Keeping every candidate
orthogonalized against the already-selected ones, adding candidate ``j``
lowers the residual by ``(r . c_j)^2 / ||c_j||^2`` where ``r`` is the current
residual and ``c_j`` the orthogonalized candidate; the best candidate is the
argmax of that gain. This costs ``O(caliber * m * max_cand * k)`` for ``m``
rows, whereas solving a batched ``lstsq`` for every padded trial system at
every growth step dominated bootstrap AMG setup. Candidates that are
numerically dependent on the selected set (duplicate rows, rows of exactly
zero energy, padded duplicates) are detected with a relative norm test and get
zero gain, which is the same outcome ``lstsq`` reached through its
rank-revealing truncation.

Real-SPD substitution: formulas using Hermitian-transpose notation (``V^H``)
are read as the plain transpose ``V^T`` - torchalg is real-SPD only
throughout (no complex numbers anywhere in this codebase).
"""

from __future__ import annotations

import torch

_DEPENDENCE_ULPS = 1.0
"""A candidate whose orthogonalized squared norm falls to ``eps`` times its
original squared norm (i.e. its norm to ``sqrt(eps)`` of the original, the
resolution a Gram-matrix rank decision has) is numerically dependent on the
selected set and gets zero gain."""

_PIVOT_RATIO_ULPS = 1000.0
"""``batched_ls_interpolation_rows`` trusts a Cholesky factorization only when
``min(L_ii)^2 / max(L_ii)^2 > _PIVOT_RATIO_ULPS * eps``; rows below that, or
where the factorization fails, are re-solved with ``lstsq``."""

_RANK_DEFICIENCY_RTOL = 1e-10
"""Relative tolerance for ``torch.linalg.matrix_rank`` when checking
[STATUS14] Sec. 3's uniqueness condition ``rank(V_{C_i}) = |C_i|`` on
``V_{C_i} W V_{C_i}^T``; below full rank, ``ls_interpolation_row`` falls back
to ``torch.linalg.lstsq`` instead of ``torch.linalg.solve``."""


def ls_interpolation_row(
    test_vectors: torch.Tensor,
    target: int,
    interp_set: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Closed-form weighted-LS interpolation row ``p_i`` ([STATUS14] eq. 3.1).

    Solves ``p_i V_{C_i} W V_{C_i}^T = V_i W V_{C_i}^T`` for ``p_i`` (the
    normal equations of the weighted LS functional, eq. 3.1), returning the
    minimizer's transpose as a plain vector so ``p @ test_vectors[interp_set]``
    reconstructs the fitted approximation to ``test_vectors[target]``. Uses
    ``torch.linalg.solve`` on ``V_{C_i} W V_{C_i}^T`` when that Gram matrix is
    full rank; falls back to ``torch.linalg.lstsq`` when
    ``rank(V_{C_i} W V_{C_i}^T) < |interp_set|``, i.e. when [STATUS14] Sec.
    3's uniqueness condition ``rank(V_{C_i}) = |C_i|`` fails (typically
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


def _ls_residual(
    test_vectors: torch.Tensor,
    target: int,
    interp_set: torch.Tensor,
    weights: torch.Tensor,
) -> float:
    """Weighted LS functional value ``LS_{C_i}(p_i)`` at its minimizer ([STATUS14] eq. 3.1).

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
        matrix (torch.Tensor): Dense or sparse CSR matrix ``A``, shape
            ``(n, n)``; accepted for interface parity only - see the module
            docstring.
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


def batched_select_interpolatory_set(
    padded_candidates: torch.Tensor,
    candidate_mask: torch.Tensor,
    test_vectors: torch.Tensor,
    targets: torch.Tensor,
    weights: torch.Tensor,
    caliber: int,
    gamma: float = 1.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Batched sibling of ``select_interpolatory_set``, vectorized over ``m`` rows.

    Reproduces ``select_interpolatory_set``'s greedy caliber-bounded growth
    loop and its normalized penalization test (see that function's docstring
    for the rule's derivation) for all rows at once. The best candidate per
    step is found by orthogonal matching pursuit rather than by solving a
    least-squares system per trial set (see the module docstring): the
    whitened candidate columns ``sqrt(omega) * v_j`` are gathered once into
    an ``(m, max_cand, k)`` tensor, kept orthogonalized against the already
    selected candidates, and the candidate with the largest residual
    reduction ``(r . c)^2 / ||c||^2`` wins. The residual after each pick is
    computed directly from the updated residual vector, not by subtracting
    the gain, so the penalization test compares values free of cancellation.

    Rank-deficient rows need no special handling: a candidate dependent on
    the selected set (relative orthogonalized norm below ``sqrt(eps)``,
    which includes exactly-zero columns and a duplicate of a selected
    candidate) has zero gain and leaves the residual unchanged, so it either
    loses to any informative candidate or fails the penalization test and
    stops the row. Because the test is relative to each column's own norm,
    the selection is invariant to the overall scale of ``test_vectors``
    (relaxed bootstrap vectors reach ``~1e-8`` magnitude). Equal gains
    resolve to the lowest candidate slot (``argmax`` returns the first
    maximum), matching the single-row loop's first-minimum rule.

    Once a row's best candidate at some step fails the penalization test,
    that row is permanently deactivated (``active``) and never grows again
    on a later step - exactly matching the single-row loop's ``break``,
    not merely skipping one step.

    The only Python-level loop is ``for step in range(caliber)`` - a fixed,
    small (default 4) hyperparameter, not data-dependent.

    Args:
        padded_candidates (torch.Tensor): Long tensor, shape
            ``(m, max_cand)``, row ``r``'s eligible candidate indices
            left-aligned into a dense ``max_cand``-wide slot; padding
            slots (beyond that row's real candidate count) hold arbitrary
            values and must be marked ``False`` in ``candidate_mask``.
        candidate_mask (torch.Tensor): Bool tensor, shape
            ``(m, max_cand)``, ``True`` where ``padded_candidates`` at that
            slot is a real (not padding) candidate for its row.
        test_vectors (torch.Tensor): Test vectors ``V``, shape ``(n, k)``.
        targets (torch.Tensor): Long tensor, shape ``(m,)``, each row's
            fine target index ``i``.
        weights (torch.Tensor): Per-test-vector weights, shape ``(k,)``.
        caliber (int): Maximum interpolatory-set size ``c``; also the
            fixed iteration count of this function's outer loop.
        gamma (float): Penalization exponent (paper default ``1.5``).

    Returns:
        tuple[torch.Tensor, torch.Tensor]: ``(chosen, chosen_mask)``, both
        shape ``(m, caliber)``. ``chosen_mask`` is a left-aligned prefix
        mask per row (``True`` for the first ``final_len[row]`` slots,
        ``False`` after); ``chosen[row, :chosen_mask[row].sum()]`` holds
        that row's interpolatory-set indices in selection order.
        ``chosen`` at masked-``False`` positions is unspecified padding.
    """
    m, _ = padded_candidates.shape
    device = test_vectors.device
    dtype = test_vectors.dtype
    dependence_floor = _DEPENDENCE_ULPS * torch.finfo(dtype).eps

    whitener = weights.sqrt()
    residual = test_vectors[targets] * whitener
    orthogonal = test_vectors[padded_candidates] * whitener
    original_energy = orthogonal.square().sum(dim=-1)

    empty_set_value = residual.square().sum(dim=-1)
    normalizer = torch.where(empty_set_value > 0, empty_set_value, torch.ones_like(empty_set_value))
    current_value = empty_set_value / normalizer

    active = torch.ones(m, dtype=torch.bool, device=device)
    remaining_mask = candidate_mask.clone()
    chosen = torch.zeros(m, caliber, dtype=torch.long, device=device)
    chosen_mask = torch.zeros(m, caliber, dtype=torch.bool, device=device)
    no_gain = torch.full((), float("-inf"), dtype=dtype, device=device)

    for step in range(caliber):
        energy = orthogonal.square().sum(dim=-1)
        independent = energy > dependence_floor * original_energy
        correlation = torch.einsum("mck,mk->mc", orthogonal, residual)
        coefficient = torch.where(
            independent, correlation / energy.clamp(min=torch.finfo(dtype).tiny), 0.0
        )
        gain = torch.where(remaining_mask, coefficient * correlation, no_gain)

        best_gain, best_slot = gain.max(dim=-1)
        slot_index = best_slot.view(-1, 1)
        best_candidate = padded_candidates.gather(-1, slot_index).squeeze(-1)
        best_direction = orthogonal.gather(
            1, best_slot.view(-1, 1, 1).expand(-1, 1, orthogonal.shape[-1])
        ).squeeze(1)
        best_coefficient = coefficient.gather(-1, slot_index)

        updated_residual = residual - best_coefficient * best_direction
        best_value = updated_residual.square().sum(dim=-1) / normalizer
        has_valid_candidate = torch.isfinite(best_gain)
        accept = active & has_valid_candidate & (best_value < current_value**gamma)

        chosen[:, step] = torch.where(accept, best_candidate, chosen[:, step])
        chosen_mask[:, step] = accept
        current_value = torch.where(accept, best_value, current_value)
        residual = torch.where(accept.unsqueeze(-1), updated_residual, residual)

        clear = torch.zeros_like(remaining_mask)
        clear.scatter_(-1, slot_index, accept.unsqueeze(-1))
        remaining_mask = remaining_mask & ~clear
        active = active & accept

        if step + 1 < caliber:
            direction_energy = (best_direction * best_direction).sum(dim=-1, keepdim=True)
            usable = accept.unsqueeze(-1) & (direction_energy > 0)
            projection = torch.einsum("mck,mk->mc", orthogonal, best_direction) / torch.where(
                usable, direction_energy, torch.ones_like(direction_energy)
            )
            projection = torch.where(usable, projection, 0.0)
            orthogonal = orthogonal - projection.unsqueeze(-1) * best_direction.unsqueeze(1)

    return chosen, chosen_mask


def batched_ls_interpolation_rows(
    test_vectors: torch.Tensor,
    targets: torch.Tensor,
    chosen: torch.Tensor,
    chosen_mask: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Batched sibling of ``ls_interpolation_row``, vectorized over ``m`` rows.

    Each row ``r`` may have a different valid prefix length (``0`` to
    ``caliber``, per ``chosen_mask[r]``); this solves all ``m`` rows'
    normal equations at once via a "masked identity padding" trick: invalid
    slots are zeroed out of the Gram matrix's cross terms and given an
    identity on their own diagonal, so each invalid slot independently solves
    ``scale * p = 0`` (``p = 0`` for any nonzero ``scale``) and decouples
    exactly from the row's real sub-block instead of corrupting it - this
    holds for *any* nonzero diagonal pad value, since the cross terms between
    the valid and invalid blocks are already zeroed above.

    **The pad value matters for conditioning, even though it cannot change
    the exact answer.** A bare ``1.0`` pad is fine in infinite precision,
    but bootstrap test vectors routinely carry ``~1e-8`` magnitude (many
    relaxation/bootstrap passes toward ``A x = 0``), making the *real*
    sub-block's Gram entries ``~1e-16`` - next to a literal ``1.0`` pad,
    the combined per-row matrix's condition number is then dominated by
    that ``1e16`` scale mismatch, not by the real sub-block's own
    conditioning. Scaling the pad to each row's own real-block diagonal
    magnitude removes the mismatch without touching the exactness argument
    above, and keeps the padded matrix symmetric positive definite whenever
    the real block is.

    **Solver choice.** The padded ``(m, caliber, caliber)`` Gram is
    factorized with ``torch.linalg.cholesky_ex`` (one batched call that
    reports failure per row instead of raising). A row is accepted when the
    factorization succeeded and its pivot ratio ``min(L_ii)^2 /
    max(L_ii)^2`` exceeds ``_PIVOT_RATIO_ULPS * eps``; only the remaining
    rows - singular or nearly singular Grams, e.g. duplicate or collinear
    chosen vectors, or ``caliber > k`` - are re-solved with the
    rank-revealing ``torch.linalg.lstsq``, so the result agrees with a
    full-batch ``lstsq`` while the common well-conditioned case avoids it.

    Implemented independently from ``batched_select_interpolatory_set``
    (not reusing that function's per-step byproducts) - this mirrors the
    existing single-row call site, which already re-solves independently
    after ``select_interpolatory_set`` returns, so this is the established
    pattern here, not a regression.

    Args:
        test_vectors (torch.Tensor): Test vectors ``V``, shape ``(n, k)``.
        targets (torch.Tensor): Long tensor, shape ``(m,)``, each row's
            fine target index ``i``.
        chosen (torch.Tensor): Long tensor, shape ``(m, caliber)``, from
            ``batched_select_interpolatory_set``; values at masked-``False``
            slots may be arbitrary (never read as real indices below).
        chosen_mask (torch.Tensor): Bool tensor, shape ``(m, caliber)``,
            left-aligned prefix mask matching ``chosen``.
        weights (torch.Tensor): Per-test-vector weights, shape ``(k,)``.

    Returns:
        torch.Tensor: Fitted interpolation rows, shape ``(m, caliber)``.
        Values at ``chosen_mask == False`` positions are exactly ``0.0``,
        not garbage - callers may rely on this when flattening/scattering.
    """
    target_vectors = test_vectors[targets]
    restricted = test_vectors[chosen]
    weighted = restricted * weights
    gram = weighted @ restricted.transpose(-1, -2)

    valid = chosen_mask
    cross_valid = valid.unsqueeze(-1) & valid.unsqueeze(-2)
    gram = torch.where(cross_valid, gram, torch.zeros_like(gram))

    # Scale-match the padding diagonal to each row's own real-block
    # magnitude (see docstring) instead of a bare 1.0, so the combined
    # matrix's condition number reflects only the real sub-block's own
    # conditioning. Rows with no valid slots at all (chosen_mask all
    # False) have no real-block magnitude to match - pad with 1.0 there,
    # since the whole row is masked back to 0.0 at the end regardless.
    diag_values = gram.diagonal(dim1=-2, dim2=-1)
    valid_diag_magnitude = torch.where(valid, diag_values.abs(), torch.zeros_like(diag_values))
    valid_count = valid.sum(dim=-1).clamp(min=1)
    mean_scale = valid_diag_magnitude.sum(dim=-1) / valid_count
    has_valid = valid.any(dim=-1)
    pad_scale = torch.where(
        has_valid, mean_scale.clamp(min=torch.finfo(gram.dtype).tiny), torch.ones_like(mean_scale)
    )
    diag_fix = torch.diag_embed((~valid).to(gram.dtype)) * pad_scale.view(-1, 1, 1)
    gram = gram + diag_fix

    rhs = torch.einsum("mck,mk->mc", weighted, target_vectors)
    rhs = torch.where(valid, rhs, torch.zeros_like(rhs))

    p = _solve_padded_gram(gram, rhs)
    return torch.where(valid, p, torch.zeros_like(p))


def _solve_padded_gram(gram: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    """Solve ``gram @ p = rhs`` per row: Cholesky where well conditioned, ``lstsq`` elsewhere.

    Args:
        gram (torch.Tensor): Symmetric Gram matrices, shape ``(m, c, c)``.
        rhs (torch.Tensor): Right-hand sides, shape ``(m, c)``.

    Returns:
        torch.Tensor: Solutions, shape ``(m, c)``.
    """
    factor, info = torch.linalg.cholesky_ex(gram)
    pivots = factor.diagonal(dim1=-2, dim2=-1).abs()
    pivot_ratio = (pivots.amin(dim=-1) / pivots.amax(dim=-1)) ** 2
    trusted = (info == 0) & (pivot_ratio > _PIVOT_RATIO_ULPS * torch.finfo(gram.dtype).eps)

    # Rows that failed hold unspecified factors (possibly NaN); give them a benign
    # right-hand side so the batched solve stays finite, then overwrite them below.
    solution = torch.cholesky_solve(
        torch.where(trusted.unsqueeze(-1), rhs, torch.zeros_like(rhs)).unsqueeze(-1), factor
    ).squeeze(-1)
    if bool(trusted.all()):
        return solution
    fallback = ~trusted
    refit = torch.linalg.lstsq(gram[fallback], rhs[fallback].unsqueeze(-1)).solution.squeeze(-1)
    return solution.masked_scatter(fallback.unsqueeze(-1), refit)
