"""Tests for ``torchalg.utils.bootcmatch_aggregation.bootcmatch_aggregates_from_matching``."""

from __future__ import annotations

import math

import torch

from torchalg.utils.bootcmatch_aggregation import bootcmatch_aggregates_from_matching


def test_mixed_pairs_and_singletons_build_expected_prolongator() -> None:
    # Vertices: 0-1 matched pair, 2 singleton, 3-4 matched pair.
    w = torch.tensor([3.0, 4.0, -2.0, 1.0, 1.0])
    match_of = torch.tensor([1, 0, -1, 4, 3])

    row, col, value, n_coarse = bootcmatch_aggregates_from_matching(w, match_of)

    assert n_coarse == 3
    assert row.tolist() == [0, 1, 2, 3, 4]

    # Pair {0, 1} shares a coarse column.
    assert col[0].item() == col[1].item()
    # Pair {3, 4} shares a different coarse column.
    assert col[3].item() == col[4].item()
    # Singleton 2 gets its own column, distinct from both pairs.
    assert col[2].item() not in {col[0].item(), col[3].item()}
    assert {col[0].item(), col[2].item(), col[3].item()} == {0, 1, 2}

    norm01 = math.sqrt(3.0**2 + 4.0**2)
    assert math.isclose(value[0].item(), 3.0 / norm01, rel_tol=1e-6)
    assert math.isclose(value[1].item(), 4.0 / norm01, rel_tol=1e-6)

    norm34 = math.sqrt(1.0**2 + 1.0**2)
    assert math.isclose(value[3].item(), 1.0 / norm34, rel_tol=1e-6)
    assert math.isclose(value[4].item(), 1.0 / norm34, rel_tol=1e-6)

    # Singleton value is sign(w_s).
    assert value[2].item() == -1.0


def test_matched_pair_column_reconstructs_original_smooth_vector_entries() -> None:
    """P's shared column, scaled back up, reproduces (w_i, w_j) exactly - eq. 3.2's guarantee."""
    w = torch.tensor([5.0, 12.0])
    match_of = torch.tensor([1, 0])

    _row, _col, value, n_coarse = bootcmatch_aggregates_from_matching(w, match_of)

    assert n_coarse == 1
    norm = math.sqrt(5.0**2 + 12.0**2)
    reconstructed_i = value[0].item() * norm
    reconstructed_j = value[1].item() * norm
    assert math.isclose(reconstructed_i, 5.0, rel_tol=1e-6)
    assert math.isclose(reconstructed_j, 12.0, rel_tol=1e-6)


def test_all_singletons_gives_one_aggregate_per_vertex() -> None:
    w = torch.tensor([1.0, -2.0, 3.0])
    match_of = torch.tensor([-1, -1, -1])

    _row, col, value, n_coarse = bootcmatch_aggregates_from_matching(w, match_of)

    assert n_coarse == 3
    assert sorted(col.tolist()) == [0, 1, 2]
    assert value.tolist() == [1.0, -1.0, 1.0]
