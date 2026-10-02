"""Tests for ``_bootcmatch_matching.bootcmatch_parallel_matching`` (dense).

Spec: round-based parallel approximate max-weight matching via mutually
agreeing ("locally dominant") heaviest-edge pointers - see that module's
docstring for the full algorithm. ``-inf`` marks a non-edge/degenerate
edge; every weighted adjacency fixture here is built directly inside a
fixture function, never inline in a test body, per project convention.
"""

from __future__ import annotations

import pytest
import torch

from torchalg.preconditioners.implementations.amg._bootcmatch_matching import (
    bootcmatch_parallel_matching,
)

_NEG_INF = float("-inf")


@pytest.fixture
def hand_verified_graph(torch_dtype: torch.dtype) -> torch.Tensor:
    """4-vertex weighted adjacency with a hand-verifiable best matching.

    Edges (symmetric): 0-1 (5), 0-2 (1), 1-2 (1), 2-3 (10); all other
    off-diagonal entries are ``-inf`` (no edge). The unique maximum-weight
    matching is ``{0-1, 2-3}`` (total 15): vertex 3's only edge is to 2
    (weight 10, the single heaviest edge in the graph), so 2-3 is always a
    locally dominant edge in round 1 regardless of 2's other neighbors;
    with 2 and 3 removed, 0 and 1's only remaining mutual option is each
    other.
    """
    dense = torch.full((4, 4), _NEG_INF, dtype=torch_dtype)
    edges = {(0, 1): 5.0, (0, 2): 1.0, (1, 2): 1.0, (2, 3): 10.0}
    for (i, j), w in edges.items():
        dense[i, j] = w
        dense[j, i] = w
    return dense


@pytest.fixture
def no_fine_only(torch_dtype: torch.dtype) -> torch.Tensor:
    """All-``False`` ``fine_only_mask`` over 4 vertices."""
    return torch.zeros(4, dtype=torch.bool)


def _random_weighted_adjacency(n: int, seed: int, dtype: torch.dtype) -> torch.Tensor:
    """Symmetric random dense weighted adjacency, ``-inf`` diagonal, seeded.

    Args:
        n (int): Vertex count.
        seed (int): Explicit RNG seed for determinism.
        dtype (torch.dtype): Target dtype.

    Returns:
        torch.Tensor: Dense ``(n, n)`` symmetric weights, diagonal ``-inf``.
    """
    generator = torch.Generator().manual_seed(seed)
    raw = torch.rand(n, n, generator=generator, dtype=dtype)
    symmetric = (raw + raw.T) / 2
    symmetric.fill_diagonal_(_NEG_INF)
    return symmetric


@pytest.fixture(params=[0, 1, 2, 3])
def random_small_graph(request: pytest.FixtureRequest, torch_dtype: torch.dtype) -> torch.Tensor:
    """Several seeded 6-vertex random symmetric weighted adjacencies."""
    return _random_weighted_adjacency(6, seed=request.param, dtype=torch_dtype)


@pytest.fixture
def no_fine_only_small(torch_dtype: torch.dtype) -> torch.Tensor:
    """All-``False`` ``fine_only_mask`` over 6 vertices."""
    return torch.zeros(6, dtype=torch.bool)


@pytest.fixture
def fine_only_graph(torch_dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    """4-vertex graph where vertex 1 is marked ``fine_only`` (never a candidate).

    Vertex 1 is otherwise the heaviest possible partner for both 0 and 2
    (weights 100), so if ``fine_only_mask`` were ignored, 1 would clearly
    get matched; this isolates that the mask is actually respected.
    """
    dense = torch.full((4, 4), _NEG_INF, dtype=torch_dtype)
    edges = {(0, 1): 100.0, (1, 2): 100.0, (2, 3): 1.0}
    for (i, j), w in edges.items():
        dense[i, j] = w
        dense[j, i] = w
    mask = torch.tensor([False, True, False, False])
    return dense, mask


@pytest.fixture
def isolated_vertex_graph(torch_dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    """4-vertex graph where vertex 3 has no valid (non-``-inf``) neighbor at all."""
    dense = torch.full((4, 4), _NEG_INF, dtype=torch_dtype)
    dense[0, 1] = dense[1, 0] = 1.0
    mask = torch.zeros(4, dtype=torch.bool)
    return dense, mask


def test_matches_brute_force_reference(
    hand_verified_graph: torch.Tensor, no_fine_only: torch.Tensor
) -> None:
    match_of = bootcmatch_parallel_matching(hand_verified_graph, no_fine_only)
    expected = torch.tensor([1, 0, 3, 2])
    assert torch.equal(match_of, expected)


def test_symmetry_invariant(
    random_small_graph: torch.Tensor, no_fine_only_small: torch.Tensor
) -> None:
    match_of = bootcmatch_parallel_matching(random_small_graph, no_fine_only_small)
    matched = match_of != -1
    assert torch.equal(match_of[match_of[matched]], torch.arange(match_of.shape[0])[matched])


def test_fine_only_mask_respected(fine_only_graph: tuple[torch.Tensor, torch.Tensor]) -> None:
    weights, mask = fine_only_graph
    match_of = bootcmatch_parallel_matching(weights, mask)
    assert match_of[1].item() == -1
    assert not torch.any(match_of == 1)


def test_no_valid_candidate_leaves_vertex_unmatched(
    isolated_vertex_graph: tuple[torch.Tensor, torch.Tensor],
) -> None:
    weights, mask = isolated_vertex_graph
    match_of = bootcmatch_parallel_matching(weights, mask)
    assert match_of[3].item() == -1
    assert match_of[0].item() == 1
    assert match_of[1].item() == 0
