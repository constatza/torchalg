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

Real-SPD substitution: formulas using Hermitian-transpose notation (``V^H``)
are read as the plain transpose ``V^T`` - torchalg is real-SPD only
throughout (no complex numbers anywhere in this codebase).
"""

from __future__ import annotations

import torch

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

    Replicates ``select_interpolatory_set``'s greedy caliber-bounded growth
    loop and its normalized penalization test exactly (see that function's
    docstring for the rule's derivation), but evaluates every row's trial
    candidates in one batched tensor op per growth step instead of a nested
    Python loop per fine grid point - the actual scalability fix this
    function exists for (``BAMGCoarsening._prolongation`` calls the
    single-row path once per fine row, up to tens of thousands of times).

    The only Python-level loop is ``for step in range(caliber)`` - a fixed,
    small (default 4) hyperparameter, not data-dependent; every iteration
    body is pure batched tensor ops over all ``m`` rows and all
    ``max_cand`` candidate slots at once.

    **Documented deviation from the single-row path:** step (f) below
    always uses ``torch.linalg.lstsq``, never ``torch.linalg.solve`` with a
    rank check. A batched ``solve`` would error (or silently produce
    garbage) if even one matrix among the ``m * max_cand`` trial Gram
    matrices built per step were singular, whereas ``lstsq`` degrades
    gracefully per-slot; on well-conditioned full-rank systems (where the
    single-row path takes the ``solve`` branch) the two agree to numerical
    precision, so this changes performance characteristics only, never
    results.

    Once a row's best candidate at some step fails the penalization test,
    that row is permanently deactivated (``active``) and never grows again
    on a later step - exactly matching the single-row loop's ``break``,
    not merely skipping one step.

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
    m, max_cand = padded_candidates.shape
    device = test_vectors.device

    target_vectors = test_vectors[targets]
    empty_set_value = (weights * target_vectors**2).sum(dim=-1)
    normalizer = torch.where(empty_set_value > 0, empty_set_value, torch.ones_like(empty_set_value))
    current_value = empty_set_value / normalizer

    active = torch.ones(m, dtype=torch.bool, device=device)
    remaining_mask = candidate_mask.clone()
    chosen = torch.zeros(m, caliber, dtype=torch.long, device=device)
    chosen_mask = torch.zeros(m, caliber, dtype=torch.bool, device=device)

    for step in range(caliber):
        trial_indices = torch.empty(m, max_cand, step + 1, dtype=torch.long, device=device)
        if step > 0:
            trial_indices[:, :, :step] = chosen[:, :step].unsqueeze(1).expand(-1, max_cand, -1)
        trial_indices[:, :, step] = padded_candidates

        restricted = test_vectors[trial_indices]
        weighted = restricted * weights
        gram = weighted @ restricted.transpose(-1, -2)
        rhs = torch.einsum("mcpk,mk->mcp", weighted, target_vectors)
        p = torch.linalg.lstsq(gram, rhs.unsqueeze(-1)).solution.squeeze(-1)
        reconstructed = torch.einsum("mcp,mcpk->mck", p, restricted)
        residual = target_vectors.unsqueeze(1) - reconstructed
        trial_value = (weights * residual**2).sum(dim=-1)

        valid = remaining_mask
        trial_value = torch.where(valid, trial_value, torch.full_like(trial_value, float("inf")))

        best_value, best_slot = trial_value.min(dim=-1)
        best_value_normalized = best_value / normalizer
        best_candidate = padded_candidates.gather(-1, best_slot.unsqueeze(-1)).squeeze(-1)
        has_valid_candidate = torch.isfinite(best_value)
        accept = active & has_valid_candidate & (best_value_normalized < current_value**gamma)

        chosen[:, step] = torch.where(accept, best_candidate, chosen[:, step])
        chosen_mask[:, step] = accept
        current_value = torch.where(accept, best_value_normalized, current_value)

        clear = torch.zeros_like(remaining_mask)
        clear.scatter_(-1, best_slot.unsqueeze(-1), accept.unsqueeze(-1))
        remaining_mask = remaining_mask & ~clear

        active = active & accept

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
    normal equations in one batched ``torch.linalg.lstsq`` call via a
    "masked identity padding" trick: invalid slots are zeroed out of the
    Gram matrix's cross terms and given an identity on their own diagonal,
    so each invalid slot independently solves ``scale * p = 0`` (``p = 0``
    for any nonzero ``scale``) and decouples exactly from the row's real
    sub-block instead of corrupting it - this holds for *any* nonzero
    diagonal pad value, since the cross terms between the valid and
    invalid blocks are already zeroed above.

    **The pad value matters for conditioning, even though it cannot change
    the exact answer.** A bare ``1.0`` pad is fine in infinite precision,
    but bootstrap test vectors routinely carry ``~1e-8`` magnitude (many
    relaxation/bootstrap passes toward ``A x = 0``), making the *real*
    sub-block's Gram entries ``~1e-16`` - next to a literal ``1.0`` pad,
    the combined per-row matrix's condition number is then dominated by
    that ``1e16`` scale mismatch, not by the real sub-block's own
    conditioning, and ``torch.linalg.lstsq`` can return numerical zero for
    the real block instead of its correct value. Scaling the pad to each
    row's own real-block diagonal magnitude removes the mismatch without
    touching the exactness argument above.

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

    p = torch.linalg.lstsq(gram, rhs.unsqueeze(-1)).solution.squeeze(-1)
    return torch.where(valid, p, torch.zeros_like(p))
