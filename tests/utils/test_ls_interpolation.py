"""Tests for the batched LS-interpolation functions in ``torchalg.utils.ls_interpolation``.

Every batched-function test case is cross-checked against the existing
single-row ``select_interpolatory_set``/``ls_interpolation_row`` functions
(looped in plain Python for the reference), per this module's "the batched
functions must reproduce the per-row loop exactly" contract - there is no
other oracle for this code.
"""

from __future__ import annotations

import torch

from torchalg.utils.ls_interpolation import (
    batched_ls_interpolation_rows,
    batched_select_interpolatory_set,
    ls_interpolation_row,
    select_interpolatory_set,
)

CALIBER = 4


def _pad_candidates(
    candidate_sets: list[torch.Tensor], max_cand: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Left-align each row's candidate indices into a dense ``(m, max_cand)`` pair.

    Args:
        candidate_sets: Per-row 1D long tensors of candidate indices.
        max_cand: Padded width; must be >= every row's candidate count.

    Returns:
        ``(padded_candidates, candidate_mask)``, both ``(m, max_cand)``.
    """
    m = len(candidate_sets)
    padded = torch.zeros(m, max_cand, dtype=torch.long)
    mask = torch.zeros(m, max_cand, dtype=torch.bool)
    for row, candidates in enumerate(candidate_sets):
        count = candidates.shape[0]
        padded[row, :count] = candidates
        mask[row, :count] = True
    return padded, mask


def _reference_row(
    test_vectors: torch.Tensor,
    target: int,
    candidates: torch.Tensor,
    weights: torch.Tensor,
    caliber: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Loop the existing single-row functions to build a reference (interp_set, row)."""
    interp_set = select_interpolatory_set(
        candidates, test_vectors, torch.empty(0), target, weights, caliber
    )
    if interp_set.shape[0] == 0:
        return interp_set, torch.empty(0, dtype=test_vectors.dtype)
    row = ls_interpolation_row(test_vectors, target, interp_set, weights)
    return interp_set, row


def test_single_row_batched_matches_single_row_reference(ls_test_vectors, ls_weights):
    target = 5
    candidates = torch.tensor([0, 1, 2, 3, 4, 6, 7, 8], dtype=torch.long)

    ref_interp_set, ref_row = _reference_row(
        ls_test_vectors, target, candidates, ls_weights, CALIBER
    )

    padded_candidates, candidate_mask = _pad_candidates([candidates], max_cand=8)
    targets = torch.tensor([target], dtype=torch.long)

    chosen, chosen_mask = batched_select_interpolatory_set(
        padded_candidates, candidate_mask, ls_test_vectors, targets, ls_weights, CALIBER
    )
    rows = batched_ls_interpolation_rows(ls_test_vectors, targets, chosen, chosen_mask, ls_weights)

    count = int(chosen_mask[0].sum())
    assert count == ref_interp_set.shape[0]
    torch.testing.assert_close(chosen[0, :count], ref_interp_set)
    torch.testing.assert_close(rows[0, :count], ref_row)


def test_multi_row_batching_matches_looped_single_row(ls_test_vectors, ls_weights):
    candidate_sets = [
        torch.tensor([0, 1, 2], dtype=torch.long),  # smaller than caliber
        torch.tensor([0, 1, 2, 3, 4, 5, 6, 8, 9, 10], dtype=torch.long),  # larger than caliber
        torch.tensor([1, 3, 5, 7], dtype=torch.long),  # exactly caliber-sized
        torch.tensor([2, 4, 6, 9, 10, 11], dtype=torch.long),
    ]
    targets_list = [3, 11, 0, 7]
    max_cand = max(c.shape[0] for c in candidate_sets)

    references = [
        _reference_row(ls_test_vectors, target, candidates, ls_weights, CALIBER)
        for target, candidates in zip(targets_list, candidate_sets, strict=True)
    ]

    padded_candidates, candidate_mask = _pad_candidates(candidate_sets, max_cand)
    targets = torch.tensor(targets_list, dtype=torch.long)

    chosen, chosen_mask = batched_select_interpolatory_set(
        padded_candidates, candidate_mask, ls_test_vectors, targets, ls_weights, CALIBER
    )
    rows = batched_ls_interpolation_rows(ls_test_vectors, targets, chosen, chosen_mask, ls_weights)

    for row_idx, (ref_interp_set, ref_row) in enumerate(references):
        count = int(chosen_mask[row_idx].sum())
        assert count == ref_interp_set.shape[0], f"row {row_idx}"
        torch.testing.assert_close(chosen[row_idx, :count], ref_interp_set)
        torch.testing.assert_close(rows[row_idx, :count], ref_row)


def test_row_with_zero_eligible_candidates_does_not_corrupt_other_rows(ls_test_vectors, ls_weights):
    candidate_sets = [
        torch.tensor([0, 1, 2, 3], dtype=torch.long),
        torch.tensor([], dtype=torch.long),  # zero eligible candidates
        torch.tensor([4, 5, 6, 7, 8], dtype=torch.long),
    ]
    targets_list = [9, 10, 11]
    max_cand = max(c.shape[0] for c in candidate_sets)

    references = [
        _reference_row(ls_test_vectors, target, candidates, ls_weights, CALIBER)
        for target, candidates in zip(targets_list, candidate_sets, strict=True)
    ]

    padded_candidates, candidate_mask = _pad_candidates(candidate_sets, max_cand)
    targets = torch.tensor(targets_list, dtype=torch.long)

    chosen, chosen_mask = batched_select_interpolatory_set(
        padded_candidates, candidate_mask, ls_test_vectors, targets, ls_weights, CALIBER
    )
    rows = batched_ls_interpolation_rows(ls_test_vectors, targets, chosen, chosen_mask, ls_weights)

    assert not chosen_mask[1].any()
    torch.testing.assert_close(rows[1], torch.zeros(CALIBER, dtype=ls_test_vectors.dtype))

    for row_idx in (0, 2):
        ref_interp_set, ref_row = references[row_idx]
        count = int(chosen_mask[row_idx].sum())
        assert count == ref_interp_set.shape[0], f"row {row_idx}"
        torch.testing.assert_close(chosen[row_idx, :count], ref_interp_set)
        torch.testing.assert_close(rows[row_idx, :count], ref_row)


def test_early_stopping_parity_across_rows(ls_test_vectors, ls_weights):
    """One row should stop before caliber (penalization test fails), another should reach it."""
    # A duplicated/near-duplicate row set tends to trigger early rank-deficiency stalling
    # via the normalized penalization test; a rich, independent candidate set reaches caliber.
    candidate_sets = [
        torch.tensor([0, 1], dtype=torch.long),  # fewer candidates than caliber: forced early stop
        torch.tensor([0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11], dtype=torch.long),  # reaches caliber
    ]
    targets_list = [11, 2]
    max_cand = max(c.shape[0] for c in candidate_sets)

    references = [
        _reference_row(ls_test_vectors, target, candidates, ls_weights, CALIBER)
        for target, candidates in zip(targets_list, candidate_sets, strict=True)
    ]

    padded_candidates, candidate_mask = _pad_candidates(candidate_sets, max_cand)
    targets = torch.tensor(targets_list, dtype=torch.long)

    chosen, chosen_mask = batched_select_interpolatory_set(
        padded_candidates, candidate_mask, ls_test_vectors, targets, ls_weights, CALIBER
    )

    ref_counts = [ref_interp_set.shape[0] for ref_interp_set, _ in references]
    assert ref_counts[0] < CALIBER
    assert ref_counts[1] == CALIBER
    for row_idx, ref_count in enumerate(ref_counts):
        assert int(chosen_mask[row_idx].sum()) == ref_count, f"row {row_idx}"
        ref_interp_set, _ = references[row_idx]
        torch.testing.assert_close(chosen[row_idx, :ref_count], ref_interp_set)
