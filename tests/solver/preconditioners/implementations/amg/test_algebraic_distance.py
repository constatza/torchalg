"""Algebraic-distance strength-of-connection tests (Bootstrap AMG, docs/bootstrap-amg.md Sec. 2.2).

Fixtures live in this module, matching ``test_compatible_relaxation.py``'s
precedent: each ``implementations/amg/`` test module keeps its fixtures
local rather than pre-emptively factoring them into a shared conftest before
a second consumer exists.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest
import torch

from torchalg.preconditioners.implementations.amg._algebraic_distance import (
    algebraic_distance,
    strength_graph,
)


@pytest.fixture
def identity_test_vectors_6(torch_dtype: torch.dtype) -> torch.Tensor:
    """Standard-basis test vectors, shape ``(6, 6)`` - a full-rank set for neighborhood-masking checks."""
    return torch.eye(6, dtype=torch_dtype)


@pytest.fixture
def identity_test_vectors_5(torch_dtype: torch.dtype) -> torch.Tensor:
    """Standard-basis test vectors, shape ``(5, 5)`` - for the depth-two neighborhood check."""
    return torch.eye(5, dtype=torch_dtype)


@pytest.fixture
def fine_mask_excludes_node_0() -> torch.Tensor:
    """Boolean mask of length 6, ``False`` at node 0 and ``True`` elsewhere."""
    return torch.tensor([False, True, True, True, True, True])


@pytest.fixture
def all_fine_mask_6() -> torch.Tensor:
    """Boolean mask of length 6, every entry ``True`` (every point is fine)."""
    return torch.ones(6, dtype=torch.bool)


@pytest.fixture
def one_strong_one_weak_distance_6(torch_dtype: torch.dtype) -> torch.Tensor:
    """Symmetric distance matrix, row 0 strongly connected to 1, weakly to 2."""
    row0 = torch.tensor([0.0, 1.0, 0.1, 0.0, 0.0, 0.0], dtype=torch_dtype)
    distance = torch.zeros(6, 6, dtype=torch_dtype)
    distance[0] = row0
    return distance + distance.T


@pytest.fixture
def small_test_vectors(torch_dtype: torch.dtype) -> torch.Tensor:
    """Non-trivial full-column-rank test vectors, shape ``(3, 2)``, for the direct-fit check."""
    return torch.tensor([[1.0, 0.3], [0.4, 1.0], [0.7, -0.5]], dtype=torch_dtype)


def test_algebraic_distance_is_directional_not_symmetric(
    poisson_1d_factory: Callable[[int], torch.Tensor],
    small_test_vectors: torch.Tensor,
) -> None:
    """``r_ij`` is a one-sided measure ([AD11]'s own description of eq. 4.3): ``r != r.T`` in general.

    Node 0 of a 1D Poisson matrix has a single graph neighbor (node 1), so
    its residual-corrected test vectors reduce to an exact multiple of
    node 1's raw test vectors - a (near-)perfect one-sided fit, hence a very
    large ``r[0, 1]``. Node 1 has two neighbors (0 and 2), so fitting its
    residual-corrected vectors from node 0's raw vector alone is imperfect,
    giving a modest, finite ``r[1, 0]``. This regression guards against
    reintroducing the false ``r == r.T`` assumption Task 2's original plan
    incorrectly required.
    """
    matrix = poisson_1d_factory(3)
    r = algebraic_distance(small_test_vectors, matrix, depth=1)

    assert not torch.isclose(r[0, 1], r[1, 0])
    assert r[0, 1] > 1e6
    assert r[1, 0] < 100.0


def test_algebraic_distance_zero_outside_depth_one_neighborhood(
    poisson_1d_factory: Callable[[int], torch.Tensor],
    identity_test_vectors_6: torch.Tensor,
) -> None:
    """Non-adjacent pairs and the diagonal are exactly zero at depth 1."""
    matrix = poisson_1d_factory(6)
    r = algebraic_distance(identity_test_vectors_6, matrix, depth=1)
    assert r[0, 2] == 0.0
    assert r[0, 0] == 0.0
    assert r[0, 1] != 0.0


def test_algebraic_distance_depth_two_reaches_two_hop_neighbor(
    poisson_1d_factory: Callable[[int], torch.Tensor],
    identity_test_vectors_5: torch.Tensor,
) -> None:
    """Depth 2 connects node 0 to node 2 (a two-hop path in the 1D Poisson graph) but not node 1."""
    matrix = poisson_1d_factory(5)

    r_depth_one = algebraic_distance(identity_test_vectors_5, matrix, depth=1)
    r_depth_two = algebraic_distance(identity_test_vectors_5, matrix, depth=2)

    assert r_depth_one[0, 2] == 0.0
    assert r_depth_two[0, 2] != 0.0
    assert r_depth_two[0, 1] == 0.0


def test_algebraic_distance_matches_direct_weighted_ls_fit(
    poisson_1d_factory: Callable[[int], torch.Tensor],
    small_test_vectors: torch.Tensor,
) -> None:
    """``r[1, 0]`` matches an independently (loop-based) computed weighted, residual-corrected LS fit.

    Uses the ``(1, 0)`` pair rather than ``(0, 1)``: node 0 has a single
    graph neighbor, so its residual-corrected target is an exact multiple
    of node 1's raw predictor (SSE = 0, a degenerate edge case handled by
    ``algebraic_distance``'s residual clamp but not a useful numeric check
    here). Node 1 has two neighbors, giving a well-conditioned fit to check
    the closed-form vectorization against.
    """
    matrix = poisson_1d_factory(3)
    r = algebraic_distance(small_test_vectors, matrix, depth=1)

    energies = torch.stack(
        [
            (matrix @ small_test_vectors[:, kappa]) @ small_test_vectors[:, kappa]
            for kappa in range(small_test_vectors.shape[1])
        ]
    )
    weights = (small_test_vectors**2).sum(dim=0) / energies

    i, j = 1, 0
    a_ii = matrix[i, i]
    residual_i = matrix[i] @ small_test_vectors
    corrected_i = small_test_vectors[i] - residual_i / a_ii

    numerator = torch.stack(
        [
            weights[kappa] * corrected_i[kappa] * small_test_vectors[j, kappa]
            for kappa in range(small_test_vectors.shape[1])
        ]
    ).sum()
    denominator = torch.stack(
        [
            weights[kappa] * small_test_vectors[j, kappa] ** 2
            for kappa in range(small_test_vectors.shape[1])
        ]
    ).sum()
    p_ij = numerator / denominator
    sse = torch.stack(
        [
            weights[kappa] * (corrected_i[kappa] - p_ij * small_test_vectors[j, kappa]) ** 2
            for kappa in range(small_test_vectors.shape[1])
        ]
    ).sum()
    expected = 1.0 / sse

    assert torch.isclose(r[i, j], expected, atol=1e-10)


def test_strength_graph_prunes_below_threshold(
    one_strong_one_weak_distance_6: torch.Tensor,
    all_fine_mask_6: torch.Tensor,
) -> None:
    """A connection below ``theta_ad`` times the row's strongest connection is pruned."""
    graph = strength_graph(one_strong_one_weak_distance_6, all_fine_mask_6, theta_ad=0.5)
    assert graph[0, 1]
    assert not graph[0, 2]


def test_strength_graph_excludes_coarse_points(
    one_strong_one_weak_distance_6: torch.Tensor,
    fine_mask_excludes_node_0: torch.Tensor,
) -> None:
    """A strong connection is still pruned if either endpoint is not a fine point."""
    graph = strength_graph(one_strong_one_weak_distance_6, fine_mask_excludes_node_0, theta_ad=0.5)
    assert not graph[0, 1]
    assert not graph[1, 0]


def test_strength_graph_is_boolean(
    one_strong_one_weak_distance_6: torch.Tensor,
    all_fine_mask_6: torch.Tensor,
) -> None:
    """The returned strength graph has boolean dtype."""
    graph = strength_graph(one_strong_one_weak_distance_6, all_fine_mask_6, theta_ad=0.5)
    assert graph.dtype == torch.bool
