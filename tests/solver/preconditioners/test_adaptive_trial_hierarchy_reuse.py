"""Adaptive SA prepares each trial hierarchy once per candidate, dense and sparse trees.

The general setup stage cycles on the whole hierarchy and then on each nested
sub-hierarchy. Between two of those trial solves only the level just refitted changes,
so the coarsest inverse (a ``pinv``) is computed once per candidate and every level's
setup smoother once per distinct level matrix (the fine matrix, shared by every
candidate, once per candidate).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from types import ModuleType
from typing import TYPE_CHECKING, Any

import pytest
import torch

from torchalg.preconditioners.implementations.amg import adaptive as dense_adaptive
from torchalg.preconditioners.implementations.amg import smoothers as dense_smoothers
from torchalg.sparse.preconditioners.amg import adaptive as sparse_adaptive

if TYPE_CHECKING:
    from collections.abc import Callable

POISSON_SIZE = 300
NUM_CANDIDATES = 3
SEED = 0


@dataclass(frozen=True, slots=True)
class Tree:
    """One implementation tree: its adaptive module and operand conversion."""

    adaptive: ModuleType
    to_operand: Callable[[torch.Tensor], torch.Tensor]


TREES = {
    "dense": Tree(dense_adaptive, lambda a: a),
    "sparse": Tree(sparse_adaptive, lambda a: a.to_sparse_csr()),
}


@pytest.fixture(params=list(TREES.values()), ids=list(TREES))
def tree(request: pytest.FixtureRequest) -> Tree:
    return request.param


@pytest.fixture
def operand(tree: Tree, poisson_1d_factory: Callable[[int], torch.Tensor]) -> torch.Tensor:
    return tree.to_operand(poisson_1d_factory(POISSON_SIZE).to(torch.float64))


@pytest.fixture
def pinv_sizes(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Orders of every ``torch.linalg.pinv`` call (the coarsest inverses below 128 rows)."""
    sizes: list[int] = []
    original = torch.linalg.pinv

    def counting(matrix: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        sizes.append(matrix.shape[0])
        return original(matrix, *args, **kwargs)

    monkeypatch.setattr(torch.linalg, "pinv", counting)
    return sizes


@pytest.fixture
def dense_cycle_preparations(monkeypatch: pytest.MonkeyPatch) -> list[torch.Tensor]:
    """Every matrix the dense setup cycle's Gauss-Seidel smoother is prepared on (kept alive)."""
    matrices: list[torch.Tensor] = []
    original = dense_smoothers.prepare_symmetric_gauss_seidel

    def recording(matrix: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
        matrices.append(matrix)
        return original(matrix, *args, **kwargs)

    monkeypatch.setattr(dense_smoothers, "prepare_symmetric_gauss_seidel", recording)
    return matrices


def test_coarsest_inverse_is_computed_once_per_later_candidate(
    tree: Tree, operand: torch.Tensor, pinv_sizes: list[int]
) -> None:
    tree.adaptive.adaptive_sa_hierarchy(operand, num_candidates=NUM_CANDIDATES, seed=SEED)

    assert len(pinv_sizes) == NUM_CANDIDATES - 1


def test_dense_setup_cycle_prepares_each_level_matrix_once_per_candidate(
    poisson_1d_factory: Callable[[int], torch.Tensor],
    dense_cycle_preparations: list[torch.Tensor],
) -> None:
    matrix = poisson_1d_factory(POISSON_SIZE).to(torch.float64)

    dense_adaptive.adaptive_sa_hierarchy(matrix, num_candidates=NUM_CANDIDATES, seed=SEED)

    counts = Counter(id(prepared) for prepared in dense_cycle_preparations)
    assert counts.pop(id(matrix)) == NUM_CANDIDATES - 1
    assert counts
    assert set(counts.values()) == {1}
