"""Orthogonal-matching-pursuit selection versus the former per-step ``lstsq`` selection.

``reference_batched_select`` and ``reference_batched_fit`` are verbatim copies
of the previous batched implementations (one ``torch.linalg.lstsq`` over every
padded trial Gram matrix per growth step). They are the oracle: the production
selection must pick the same sets on well-conditioned data and must reach the
same weighted LS residual wherever candidates tie or are linearly dependent.
"""

from __future__ import annotations

from itertools import pairwise

import pytest
import torch

from tests.utils.selection_problem import SELECTION_CALIBER, SelectionProblem
from torchalg.utils import ls_interpolation
from torchalg.utils.ls_interpolation import (
    batched_ls_interpolation_rows,
    batched_select_interpolatory_set,
)


def reference_batched_select(
    padded_candidates: torch.Tensor,
    candidate_mask: torch.Tensor,
    test_vectors: torch.Tensor,
    targets: torch.Tensor,
    weights: torch.Tensor,
    caliber: int,
    gamma: float = 1.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Former implementation: batched ``lstsq`` over all trial Gram systems per step."""
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

        trial_value = torch.where(
            remaining_mask, trial_value, torch.full_like(trial_value, float("inf"))
        )

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


def reference_batched_fit(
    test_vectors: torch.Tensor,
    targets: torch.Tensor,
    chosen: torch.Tensor,
    chosen_mask: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Former implementation: one batched ``lstsq`` on the masked-identity-padded Gram."""
    target_vectors = test_vectors[targets]
    restricted = test_vectors[chosen]
    weighted = restricted * weights
    gram = weighted @ restricted.transpose(-1, -2)

    valid = chosen_mask
    cross_valid = valid.unsqueeze(-1) & valid.unsqueeze(-2)
    gram = torch.where(cross_valid, gram, torch.zeros_like(gram))

    diag_values = gram.diagonal(dim1=-2, dim2=-1)
    valid_diag_magnitude = torch.where(valid, diag_values.abs(), torch.zeros_like(diag_values))
    valid_count = valid.sum(dim=-1).clamp(min=1)
    mean_scale = valid_diag_magnitude.sum(dim=-1) / valid_count
    has_valid = valid.any(dim=-1)
    pad_scale = torch.where(
        has_valid, mean_scale.clamp(min=torch.finfo(gram.dtype).tiny), torch.ones_like(mean_scale)
    )
    gram = gram + torch.diag_embed((~valid).to(gram.dtype)) * pad_scale.view(-1, 1, 1)

    rhs = torch.einsum("mck,mk->mc", weighted, target_vectors)
    rhs = torch.where(valid, rhs, torch.zeros_like(rhs))
    p = torch.linalg.lstsq(gram, rhs.unsqueeze(-1)).solution.squeeze(-1)
    return torch.where(valid, p, torch.zeros_like(p))


def _select(problem: SelectionProblem, selector=batched_select_interpolatory_set):
    return selector(
        problem.padded_candidates,
        problem.candidate_mask,
        problem.test_vectors,
        problem.targets,
        problem.weights,
        SELECTION_CALIBER,
    )


def _tolerance(dtype: torch.dtype) -> float:
    return 1e-9 if dtype == torch.float64 else 2e-3


def _final_residuals(
    problem: SelectionProblem, chosen: torch.Tensor, chosen_mask: torch.Tensor
) -> torch.Tensor:
    """Weighted LS residual of each row's chosen prefix, normalised by the empty-set energy.

    Solved in float64 with a minimum-norm ``lstsq`` on the whitened columns,
    independent of both selectors' internals.
    """
    sqrt_w = problem.weights.double().sqrt()
    vectors = problem.test_vectors.double()
    residuals = []
    for row in range(chosen.shape[0]):
        target = vectors[problem.targets[row]] * sqrt_w
        count = int(chosen_mask[row].sum())
        if count == 0:
            residuals.append((target**2).sum())
            continue
        columns = (vectors[chosen[row, :count]] * sqrt_w).T
        solution = torch.linalg.lstsq(columns, target.unsqueeze(-1), driver="gelsd").solution
        residuals.append(((target - (columns @ solution).squeeze(-1)) ** 2).sum())
    empty = torch.stack(
        [(problem.test_vectors[t].double() * sqrt_w).pow(2).sum() for t in problem.targets]
    )
    return torch.stack(residuals) / empty.clamp(min=torch.finfo(torch.float64).tiny)


def _assert_valid_selection(
    problem: SelectionProblem, chosen: torch.Tensor, chosen_mask: torch.Tensor
) -> None:
    """Mask is a left-aligned prefix; picks come from the row's own unmasked, unrepeated slots."""
    assert torch.equal(chosen_mask, chosen_mask.cumprod(dim=-1).bool())
    for row in range(chosen.shape[0]):
        count = int(chosen_mask[row].sum())
        eligible = problem.padded_candidates[row][problem.candidate_mask[row]].tolist()
        picks = chosen[row, :count].tolist()
        assert all(pick in eligible for pick in picks), f"row {row}"
        assert len(set(picks)) == len(picks), f"row {row}"


def test_random_problem_selects_identical_sets(random_selection_problem: SelectionProblem) -> None:
    chosen, chosen_mask = _select(random_selection_problem)
    ref_chosen, ref_mask = _select(random_selection_problem, reference_batched_select)

    assert torch.equal(chosen_mask, ref_mask)
    assert torch.equal(chosen[chosen_mask], ref_chosen[ref_mask])
    assert int(chosen_mask.sum()) > 0


def test_degenerate_problem_matches_reference_residual_and_coverage(
    degenerate_selection_problem: SelectionProblem,
) -> None:
    problem = degenerate_selection_problem
    chosen, chosen_mask = _select(problem)
    ref_chosen, ref_mask = _select(problem, reference_batched_select)

    _assert_valid_selection(problem, chosen, chosen_mask)
    tolerance = _tolerance(problem.test_vectors.dtype)
    # Indices may differ only among exact ties (duplicate vectors), so compare what the
    # sets achieve: how many were taken and the weighted residual they leave behind.
    assert torch.equal(chosen_mask.sum(dim=-1), ref_mask.sum(dim=-1))
    torch.testing.assert_close(
        _final_residuals(problem, chosen, chosen_mask),
        _final_residuals(problem, ref_chosen, ref_mask),
        rtol=0.0,
        atol=tolerance,
    )


def test_rows_with_fewer_candidates_than_caliber(
    short_candidate_selection_problem: SelectionProblem,
) -> None:
    problem = short_candidate_selection_problem
    chosen, chosen_mask = _select(problem)
    ref_chosen, ref_mask = _select(problem, reference_batched_select)

    counts = problem.candidate_mask.sum(dim=-1)
    assert bool((counts < SELECTION_CALIBER).any())
    assert bool((chosen_mask.sum(dim=-1) <= counts).all())
    assert not chosen_mask[counts == 0].any()
    assert torch.equal(chosen_mask, ref_mask)
    assert torch.equal(chosen[chosen_mask], ref_chosen[ref_mask])


def test_selection_is_invariant_to_test_vector_scale(
    random_selection_problem: SelectionProblem,
) -> None:
    """Relaxed bootstrap vectors reach ~1e-8 magnitude (Gram entries ~1e-16); picks must not change."""
    problem = random_selection_problem
    scaled = SelectionProblem(
        padded_candidates=problem.padded_candidates,
        candidate_mask=problem.candidate_mask,
        test_vectors=problem.test_vectors * 1e-8,
        targets=problem.targets,
        weights=problem.weights,
    )
    chosen, chosen_mask = _select(problem)
    scaled_chosen, scaled_mask = _select(scaled)

    assert torch.equal(chosen_mask, scaled_mask)
    assert torch.equal(chosen[chosen_mask], scaled_chosen[scaled_mask])


def test_selection_does_not_call_lstsq(
    degenerate_selection_problem: SelectionProblem, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("selection must not solve per-row least-squares systems")

    monkeypatch.setattr(torch.linalg, "lstsq", forbidden)
    _select(degenerate_selection_problem)


def test_equal_gain_ties_pick_lowest_slot(float_dtype: torch.dtype) -> None:
    """Identical candidate vectors tie exactly; the lowest slot must win, as with ``argmin``."""
    base = torch.tensor([[1.0, 0.5, -0.25]], dtype=float_dtype)
    vectors = torch.cat([base, base, base, torch.tensor([[0.2, 0.1, 0.0]], dtype=float_dtype)])
    problem = SelectionProblem(
        padded_candidates=torch.tensor([[2, 1, 0]]),
        candidate_mask=torch.tensor([[True, True, True]]),
        test_vectors=vectors,
        targets=torch.tensor([3]),
        weights=torch.ones(3, dtype=float_dtype),
    )
    chosen, chosen_mask = _select(problem)
    assert int(chosen_mask.sum()) >= 1
    assert int(chosen[0, 0]) == 2


# --- final fit ------------------------------------------------------------------------


def _fit_inputs(problem: SelectionProblem) -> tuple[torch.Tensor, ...]:
    chosen, chosen_mask = _select(problem, reference_batched_select)
    return problem.test_vectors, problem.targets, chosen, chosen_mask, problem.weights


def test_fit_matches_lstsq_reference(random_selection_problem: SelectionProblem) -> None:
    args = _fit_inputs(random_selection_problem)
    rows = batched_ls_interpolation_rows(*args)
    reference = reference_batched_fit(*args)
    dtype = random_selection_problem.test_vectors.dtype
    torch.testing.assert_close(rows, reference, rtol=_tolerance(dtype), atol=_tolerance(dtype))
    assert torch.equal(rows[~args[3]], torch.zeros_like(rows[~args[3]]))


def test_fit_falls_back_to_lstsq_on_singular_gram_rows(
    degenerate_selection_problem: SelectionProblem,
) -> None:
    problem = degenerate_selection_problem
    vectors, targets, chosen, chosen_mask, weights = _fit_inputs(problem)
    # Force exactly singular Grams: repeat the first pick in the second slot and
    # point another row at an all-zero vector.
    chosen = chosen.clone()
    chosen_mask = chosen_mask.clone()
    chosen[0, 0] = chosen[0, 1] = 0
    chosen_mask[0, :2] = True
    chosen[1, 0] = 2 * 10  # an all-zero test vector
    chosen_mask[1, 0] = True

    rows = batched_ls_interpolation_rows(vectors, targets, chosen, chosen_mask, weights)
    reference = reference_batched_fit(vectors, targets, chosen, chosen_mask, weights)

    assert bool(torch.isfinite(rows).all())
    dtype = vectors.dtype
    torch.testing.assert_close(rows, reference, rtol=_tolerance(dtype), atol=_tolerance(dtype))


# --- width bucketing ------------------------------------------------------------------


def _bucketed_slots(chunks: list[tuple[int, int]], lengths: list[int]) -> int:
    return sum((hi - lo) * max(lengths[hi - 1], 1) for lo, hi in chunks)


@pytest.fixture
def lengths_with_wide_outlier() -> list[int]:
    """Ascending candidate counts: two big groups and one fallback-width row."""
    return sorted([3] * 400 + [5] * 400 + [200])


@pytest.fixture
def short_lengths() -> list[int]:
    return sorted([3] * 5 + [4] * 5)


def test_width_chunks_cover_rows_and_isolate_a_wide_outlier(
    lengths_with_wide_outlier: list[int],
) -> None:
    lengths = lengths_with_wide_outlier
    chunks = ls_interpolation._width_chunks(lengths, overhead=64)

    assert chunks[0][0] == 0 and chunks[-1][1] == len(lengths)
    assert all(prev[1] == nxt[0] for prev, nxt in pairwise(chunks))
    assert _bucketed_slots(chunks, lengths) < len(lengths) * lengths[-1] // 10
    assert chunks[-1] == (len(lengths) - 1, len(lengths))


def test_width_chunks_stay_whole_when_overhead_dominates(short_lengths: list[int]) -> None:
    lengths = short_lengths
    assert ls_interpolation._width_chunks(lengths, overhead=10**6) == [(0, len(lengths))]


def test_width_chunks_of_no_rows_is_empty() -> None:
    assert ls_interpolation._width_chunks([], overhead=64) == []


@pytest.mark.parametrize(
    "overhead", [0, 16, 10**9], ids=["every-length", "grouped", "single-chunk"]
)
def test_bucketed_selection_is_bit_identical_to_one_full_width_chunk(
    short_candidate_selection_problem: SelectionProblem,
    monkeypatch: pytest.MonkeyPatch,
    overhead: int,
) -> None:
    problem = short_candidate_selection_problem
    monkeypatch.setattr(ls_interpolation, "_CHUNK_OVERHEAD_SLOTS", 10**9)
    full_chosen, full_mask = _select(problem)
    monkeypatch.setattr(ls_interpolation, "_CHUNK_OVERHEAD_SLOTS", overhead)
    chosen, chosen_mask = _select(problem)

    assert torch.equal(chosen_mask, full_mask)
    assert torch.equal(chosen, full_chosen)


def test_bucketed_selection_runs_each_chunk_at_its_own_width(
    short_candidate_selection_problem: SelectionProblem, monkeypatch: pytest.MonkeyPatch
) -> None:
    problem = short_candidate_selection_problem
    monkeypatch.setattr(ls_interpolation, "_CHUNK_OVERHEAD_SLOTS", 0)
    widths: list[int] = []
    original = ls_interpolation._select_block

    def recording(
        padded_candidates: torch.Tensor,
        candidate_mask: torch.Tensor,
        test_vectors: torch.Tensor,
        targets: torch.Tensor,
        weights: torch.Tensor,
        caliber: int,
        gamma: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        widths.append(padded_candidates.shape[1])
        return original(
            padded_candidates, candidate_mask, test_vectors, targets, weights, caliber, gamma
        )

    monkeypatch.setattr(ls_interpolation, "_select_block", recording)
    _select(problem)

    assert len(widths) > 1
    assert max(widths) == int(problem.candidate_mask.sum(dim=-1).max())
    assert sum(widths) < problem.padded_candidates.shape[1] * len(widths)
