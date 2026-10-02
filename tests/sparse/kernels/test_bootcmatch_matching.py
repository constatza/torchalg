"""Tests for ``torchalg.sparse.kernels.bootcmatch_matching.bootcmatch_parallel_matching``.

Spec: identical algorithm and tie-break rule (lowest column index) to the
dense sibling in
``torchalg.preconditioners.implementations.amg._bootcmatch_matching`` -
see that module's docstring for the full algorithm, and this module's own
docstring for the sparse-specific argmax-with-tie-break construction.
Every weighted-adjacency fixture is sparse CSR, built directly inside a
fixture function, never inline in a test body.
"""

from __future__ import annotations

import pytest
import torch

from torchalg.preconditioners.implementations.amg._bootcmatch_matching import (
    bootcmatch_parallel_matching as dense_bootcmatch_parallel_matching,
)
from torchalg.sparse.kernels.bootcmatch_matching import (
    bootcmatch_parallel_matching as sparse_bootcmatch_parallel_matching,
)

_NEG_INF = float("-inf")


def _dense_to_sparse_edges(dense: torch.Tensor) -> torch.Tensor:
    """Convert a dense ``-inf``-padded weighted adjacency to sparse CSR.

    Only finite entries are stored - ``-inf``/non-edge positions become
    structurally absent, matching the input contract (Component A's
    producer is expected to emit the same structural-absence convention).
    """
    finite = torch.isfinite(dense)
    sparse_dense = torch.where(finite, dense, torch.zeros_like(dense))
    return (sparse_dense * finite).to_sparse_csr()


@pytest.fixture
def hand_verified_graph_dense(torch_dtype: torch.dtype) -> torch.Tensor:
    """4-vertex weighted adjacency with a hand-verifiable best matching.

    Identical graph to the dense kernel's own hand-verified fixture: edges
    0-1 (5), 0-2 (1), 1-2 (1), 2-3 (10); unique maximum-weight matching
    ``{0-1, 2-3}``.
    """
    dense = torch.full((4, 4), _NEG_INF, dtype=torch_dtype)
    edges = {(0, 1): 5.0, (0, 2): 1.0, (1, 2): 1.0, (2, 3): 10.0}
    for (i, j), w in edges.items():
        dense[i, j] = w
        dense[j, i] = w
    return dense


@pytest.fixture
def hand_verified_graph_csr(hand_verified_graph_dense: torch.Tensor) -> torch.Tensor:
    return _dense_to_sparse_edges(hand_verified_graph_dense)


@pytest.fixture
def no_fine_only(torch_dtype: torch.dtype) -> torch.Tensor:
    return torch.zeros(4, dtype=torch.bool)


def _random_weighted_adjacency(n: int, seed: int, dtype: torch.dtype) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    raw = torch.rand(n, n, generator=generator, dtype=dtype)
    symmetric = (raw + raw.T) / 2
    symmetric.fill_diagonal_(_NEG_INF)
    return symmetric


@pytest.fixture(params=[0, 1, 2, 3])
def random_small_graph_dense(
    request: pytest.FixtureRequest, torch_dtype: torch.dtype
) -> torch.Tensor:
    return _random_weighted_adjacency(6, seed=request.param, dtype=torch_dtype)


@pytest.fixture
def random_small_graph_csr(random_small_graph_dense: torch.Tensor) -> torch.Tensor:
    return _dense_to_sparse_edges(random_small_graph_dense)


@pytest.fixture
def no_fine_only_small(torch_dtype: torch.dtype) -> torch.Tensor:
    return torch.zeros(6, dtype=torch.bool)


@pytest.fixture
def fine_only_graph_dense(torch_dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    dense = torch.full((4, 4), _NEG_INF, dtype=torch_dtype)
    edges = {(0, 1): 100.0, (1, 2): 100.0, (2, 3): 1.0}
    for (i, j), w in edges.items():
        dense[i, j] = w
        dense[j, i] = w
    mask = torch.tensor([False, True, False, False])
    return dense, mask


@pytest.fixture
def fine_only_graph_csr(
    fine_only_graph_dense: tuple[torch.Tensor, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    dense, mask = fine_only_graph_dense
    return _dense_to_sparse_edges(dense), mask


@pytest.fixture
def isolated_vertex_graph_dense(torch_dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    dense = torch.full((4, 4), _NEG_INF, dtype=torch_dtype)
    dense[0, 1] = dense[1, 0] = 1.0
    mask = torch.zeros(4, dtype=torch.bool)
    return dense, mask


@pytest.fixture
def isolated_vertex_graph_csr(
    isolated_vertex_graph_dense: tuple[torch.Tensor, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    dense, mask = isolated_vertex_graph_dense
    return _dense_to_sparse_edges(dense), mask


def test_matches_brute_force_reference(
    hand_verified_graph_csr: torch.Tensor, no_fine_only: torch.Tensor
) -> None:
    match_of = sparse_bootcmatch_parallel_matching(hand_verified_graph_csr, no_fine_only)
    expected = torch.tensor([1, 0, 3, 2])
    assert torch.equal(match_of, expected)


def test_symmetry_invariant(
    random_small_graph_csr: torch.Tensor, no_fine_only_small: torch.Tensor
) -> None:
    match_of = sparse_bootcmatch_parallel_matching(random_small_graph_csr, no_fine_only_small)
    matched = match_of != -1
    assert torch.equal(match_of[match_of[matched]], torch.arange(match_of.shape[0])[matched])


def test_fine_only_mask_respected(fine_only_graph_csr: tuple[torch.Tensor, torch.Tensor]) -> None:
    weights, mask = fine_only_graph_csr
    match_of = sparse_bootcmatch_parallel_matching(weights, mask)
    assert match_of[1].item() == -1
    assert not torch.any(match_of == 1)


def test_no_valid_candidate_leaves_vertex_unmatched(
    isolated_vertex_graph_csr: tuple[torch.Tensor, torch.Tensor],
) -> None:
    weights, mask = isolated_vertex_graph_csr
    match_of = sparse_bootcmatch_parallel_matching(weights, mask)
    assert match_of[3].item() == -1
    assert match_of[0].item() == 1
    assert match_of[1].item() == 0


def test_dense_sparse_parity(
    random_small_graph_dense: torch.Tensor, no_fine_only_small: torch.Tensor
) -> None:
    dense_result = dense_bootcmatch_parallel_matching(random_small_graph_dense, no_fine_only_small)
    sparse_result = sparse_bootcmatch_parallel_matching(
        _dense_to_sparse_edges(random_small_graph_dense), no_fine_only_small
    )
    assert torch.equal(dense_result, sparse_result)


@pytest.fixture
def random_large_graph_dense(torch_dtype: torch.dtype) -> torch.Tensor:
    """100-vertex random weighted adjacency.

    Large enough that the round-based matching needs more rounds than
    ``_SYNC_CHECK_STRIDE``, exercising the batched host-sync path (not just
    its first, always-checked round).
    """
    return _random_weighted_adjacency(100, seed=0, dtype=torch_dtype)


@pytest.fixture
def no_fine_only_large(torch_dtype: torch.dtype) -> torch.Tensor:
    return torch.zeros(100, dtype=torch.bool)


def test_dense_sparse_parity_multi_round(
    random_large_graph_dense: torch.Tensor, no_fine_only_large: torch.Tensor
) -> None:
    """Batched sync-check must not change the result once it actually spans multiple checks."""
    dense_result = dense_bootcmatch_parallel_matching(random_large_graph_dense, no_fine_only_large)
    sparse_result = sparse_bootcmatch_parallel_matching(
        _dense_to_sparse_edges(random_large_graph_dense), no_fine_only_large
    )
    assert torch.equal(dense_result, sparse_result)
    matched = sparse_result != -1
    assert torch.equal(sparse_result[sparse_result[matched]], torch.arange(100)[matched])
